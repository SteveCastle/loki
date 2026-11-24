// Prompt enhancement: the server's text model (loki-scribe) rewrites a Transform prompt the way the target model
// reads best (POST /api/transform/enhance-prompt, guides in media-server/tasks/prompt_guides/). Only ever on
// request, two ways: the Enhance button puts the rewrite in the prompt box, where it can be reviewed, edited or
// undone; the review step's "Enhance first" box (settings.autoEnhance, withAutoEnhance) sends every queued run
// through its own rewrite, seeded with the run's seed, and the server queues the run with it. Either way the rewrite
// is an enhance-prompt job in the server's queue; the button follows it on the /stream job events.
import { mediaServerBase } from '../../platform';
import { subscribeStream } from '../../stream-bus';
import { effectiveDreamModel, loadedDreamCatalog, type DreamCatalog } from '../dream/dream-models';
import { mediaKind, referenceTokens, type IntentId, type JobRequest, type TransformSettings } from './intents';

export type EnhanceTask = 'generate' | 'reimagine' | 'edit' | 'video';

export interface EnhanceTarget {
  /** The server's guide key. */
  model: string;
  task: EnhanceTask;
  /** The model's name, for the button's tooltip. */
  label: string;
}

export interface EnhanceRequestBody {
  prompt: string;
  model: string;
  task: EnhanceTask;
  references: Array<{ token: string; kind: string; path?: string }>;
  image?: string;
  imageToken?: string;
  /** The text model's sampling seed: another seed, another rewrite (absent = the engine's default). */
  seed?: number;
}

export interface EnhanceResult {
  prompt: string;
  model: string;
  missingTokens?: string[];
}

// loki-dream architecture → guide key.
const DREAM_GUIDES: Record<string, string> = {
  moss: 'moss-image',
  krea2: 'krea2',
  flux2: 'flux2-klein',
  anima: 'anima',
  sdxl: 'sdxl',
};

/** Which guide a prompt is rewritten with, or null when the intent's prompt can't be enhanced. */
export function enhanceTarget(intent: IntentId, s: TransformSettings, catalog: DreamCatalog | undefined): EnhanceTarget | null {
  if (intent === 'generate' || intent === 'reimagine') {
    const m = catalog ? effectiveDreamModel(catalog, s.dreamModel) : undefined;
    const arch = m?.arch || 'moss';
    return { model: DREAM_GUIDES[arch] || 'moss-image', task: intent, label: m?.archName || 'MOSS-Image' };
  }
  if (intent === 'edit') {
    return s.retouchModel === 'h3'
      ? { model: 'h3-image', task: 'edit', label: 'MiniMax H3 Image' }
      : { model: 'qwen-image-edit', task: 'edit', label: 'Qwen Image Edit' };
  }
  // Every video model (H3, Fast, the extra files) is MiniMax H3.
  if (intent === 'direct') return { model: 'h3-video', task: 'video', label: 'MiniMax H3' };
  return null;
}

/**
 * The request for a prompt: its reference tokens, and the picture the text model looks at (the image being
 * edited or redrawn, else Direct's first picture).
 */
export function enhanceBody(
  target: EnhanceTarget,
  prompt: string,
  refs: Array<{ token: string; path: string }>,
  paths: string[]
): EnhanceRequestBody {
  const body: EnhanceRequestBody = {
    prompt,
    model: target.model,
    task: target.task,
    // The server shows the text model every image and (a frame of) every video here.
    references: refs.map((r) => ({ token: r.token, kind: mediaKind(r.path), path: r.path })),
  };
  if (target.task === 'reimagine') {
    // the picture being redrawn has no token
    const src = paths.find((p) => mediaKind(p) === 'image');
    if (src) body.image = src;
  } else if (target.task === 'edit' && refs.length === 0) {
    // a single-image edit lists no tokens, but the image is still <image1>
    const src = paths.find((p) => mediaKind(p) === 'image');
    if (src) {
      body.image = src;
      body.imageToken = '<image1>';
    }
  }
  return body;
}

/** A seed for a rewrite nothing else seeds (the Enhance button: each press asks for another take). */
export function randomEnhanceSeed(): number {
  return 1 + Math.floor(Math.random() * 2147483646);
}

// The paths a job runs on: the last, quoted token of its input (see JobRequest).
function jobPaths(job: JobRequest): string[] {
  const m = /"([^"]*)"\s*$/.exec(job.input);
  return m ? m[1].split('\n').filter(Boolean) : [];
}

/**
 * "Enhance first": give every job that sends a prompt its own rewrite request, seeded with the job's seed, so runs
 * queued with a random seed get different prompts and a fixed seed gets the same one again. The server queues each
 * job once its rewrite is done (store.ts submitJobs). Jobs are returned as they were when the setting is off or the
 * intent's prompt has no guide.
 */
export function withAutoEnhance(
  jobs: JobRequest[],
  intent: IntentId,
  s: TransformSettings,
  catalog: DreamCatalog | undefined = loadedDreamCatalog()
): JobRequest[] {
  if (!s.autoEnhance) return jobs;
  const target = enhanceTarget(intent, s, catalog);
  if (!target) return jobs;
  const manual = s.manualPrompt !== null;
  return jobs.map((job) => {
    const prompt = job.fields.prompt;
    if (!prompt || !prompt.trim()) return job;
    const paths = jobPaths(job);
    // the tokens as this job's run numbers them (each pinned-references run has its own paths)
    const tokens = referenceTokens(intent, paths, manual);
    const seen = new Set<string>();
    const refs: Array<{ token: string; path: string }> = [];
    Object.keys(tokens).forEach((p) => {
      const t = tokens[p];
      if (!t.startsWith('<') || seen.has(t)) return;
      seen.add(t);
      refs.push({ token: t, path: p });
    });
    const body = enhanceBody(target, prompt, refs, paths);
    const seed = Number(job.fields.seed);
    if (Number.isFinite(seed) && seed > 0) body.seed = seed;
    return { ...job, enhance: body, label: `${job.label} · enhanced` };
  });
}

/** Where a queued rewrite is: waiting its turn in the job queue, or running. */
export type EnhancePhase = 'queued' | 'running';

interface EnhanceJobState {
  state: 'pending' | 'in_progress' | 'completed' | 'error' | 'cancelled' | 'paused';
  result?: EnhanceResult;
  error?: string;
}

// The stream is the signal; this re-check only covers a dropped connection.
const ENHANCE_RECHECK_MS = 15000;

/**
 * Queue the rewrite as an enhance-prompt job (so the text model takes its turn with the other model work) and follow
 * it on the /stream job events until it finishes. `onPhase` hears queued / running and, while it runs, how far along it is
 * (0-1, the job's progress: model load, then the writing); aborting `signal` cancels the job.
 */
export async function requestEnhance(
  body: EnhanceRequestBody,
  authToken: string | undefined,
  signal: AbortSignal,
  onPhase?: (phase: EnhancePhase, progress?: number) => void
): Promise<EnhanceResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (authToken) headers.Authorization = `Bearer ${authToken}`;
  const url = `${mediaServerBase}/api/transform/enhance-prompt`;
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: anySignal(signal, AbortSignal.timeout(15000)) });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  const jobId: string = data.jobId;
  onPhase?.('queued');

  return new Promise<EnhanceResult>((resolve, reject) => {
    let settled = false;
    let checking = false;
    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      unsubscribe();
      window.clearInterval(recheck);
      signal.removeEventListener('abort', onAbort);
      fn();
    };
    let progress: number | undefined;
    const phaseOf = (state: string) => {
      if (state === 'in_progress') onPhase?.('running', progress);
      else if (state === 'pending' || state === 'paused') onPhase?.('queued');
    };
    const progressOf = (done?: number, total?: number) => {
      if (!total || done === undefined) return;
      progress = Math.max(0, Math.min(1, done / total));
      onPhase?.('running', progress);
    };
    // the job's own status: its phase while it waits or runs, else the rewrite or why there is none
    const check = async () => {
      if (settled || checking) return;
      checking = true;
      try {
        const r = await fetch(`${url}?job=${encodeURIComponent(jobId)}`, { headers, signal: AbortSignal.timeout(10000) });
        const st = (await r.json().catch(() => null)) as (EnhanceJobState & { error?: string }) | null;
        if (!r.ok || !st) {
          finish(() => reject(new Error((st && st.error) || `HTTP ${r.status}`)));
          return;
        }
        if (st.state === 'completed' && st.result) finish(() => resolve(st.result as EnhanceResult));
        else if (st.state === 'error') finish(() => reject(new Error(st.error || 'the enhance job failed')));
        else if (st.state === 'cancelled') finish(() => reject(new Error('the enhance job was cancelled')));
        else phaseOf(st.state);
      } catch {
        // a missed check: the next event or re-check tries again
      } finally {
        checking = false;
      }
    };
    const unsubscribe = subscribeStream((type, event) => {
      if (type !== 'create' && type !== 'update' && type !== 'delete' && type !== 'progress') return;
      try {
        if (type === 'progress') {
          const p = JSON.parse(event.data) as { id?: string; done?: number; total?: number };
          if (p.id === jobId) progressOf(p.done, p.total);
          return;
        }
        const { job } = JSON.parse(event.data) as { job?: { id: string; state: string; progress_done?: number; progress_total?: number } };
        if (!job || job.id !== jobId) return;
        if (type === 'delete' || ['completed', 'error', 'cancelled'].includes(job.state)) check();
        else {
          if (job.state === 'in_progress') progressOf(job.progress_done, job.progress_total);
          phaseOf(job.state);
        }
      } catch {
        // not a job event
      }
    });
    const onAbort = () => {
      finish(() => reject(new DOMException('Aborted', 'AbortError')));
      fetch(`${mediaServerBase}/job/${jobId}/cancel`, { method: 'POST', headers, signal: AbortSignal.timeout(10000) }).catch(() => undefined);
    };
    signal.addEventListener('abort', onAbort);
    const recheck = window.setInterval(check, ENHANCE_RECHECK_MS);
    if (signal.aborted) onAbort();
    // it may have moved on before the subscription started
    else check();
  });
}

// Aborts when either signal does.
function anySignal(a: AbortSignal, b: AbortSignal): AbortSignal {
  const ctl = new AbortController();
  const stop = () => ctl.abort();
  if (a.aborted || b.aborted) stop();
  a.addEventListener('abort', stop);
  b.addEventListener('abort', stop);
  return ctl.signal;
}
