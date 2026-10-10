// Tiny external store for the Transform flow: the palette (which unmounts its
// contents when it hides) opens the full options through here, and a root-level host
// renders it. Also persists the last-used settings per intent.
import { useSyncExternalStore } from 'react';
import type { IntentId, TransformSettings, JobRequest } from './intents';
import { EngineSetupDeferred, ensureEngines } from './engine-setup';

export type FlowPhase = 'choose' | 'shape' | 'review';

export interface FlowRequest {
  /** The working set: right-clicked file plus any multi-selection, in order. */
  paths: string[];
  /** Playback position of a single video target, seconds (frame sampling). */
  videoTime?: number;
  intent?: IntentId;
  phase?: FlowPhase;
  /** A saved prompt to start from (custom intents). */
  prompt?: string;
}

let current: FlowRequest | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

export function openTransformFlow(req: FlowRequest): void {
  current = req;
  emit();
}

export function closeTransformFlow(): void {
  if (current === null) return;
  current = null;
  emit();
}

export function getFlowRequest(): FlowRequest | null {
  return current;
}

export function useFlowRequest(): FlowRequest | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
    () => null
  );
}

// --- remembered settings -----------------------------------------------------

const STORAGE_KEY = 'loki.transform.settings.v1';
type Remembered = Partial<Record<IntentId, Partial<TransformSettings>>>;

// Per-intent text (prompts, descriptions) is deliberately NOT remembered: a
// stale "make it night" must never ride along into the next image.
const REMEMBER_KEYS: Array<keyof TransformSettings> = [
  'sizeMode',
  'scale',
  'wallpaperTarget',
  'quality',
  'shake',
  'duration',
  'refSize',
  'noAudio',
  'keepSoundtrack',
];

export function loadRemembered(id: IntentId): Partial<TransformSettings> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const all = JSON.parse(raw) as Remembered;
    return all[id] || {};
  } catch {
    return {};
  }
}

export function saveRemembered(id: IntentId, s: TransformSettings): void {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const all: Remembered = raw ? JSON.parse(raw) : {};
    const picked: Record<string, unknown> = {};
    REMEMBER_KEYS.forEach((k) => {
      picked[k] = s[k];
    });
    all[id] = picked as Partial<TransformSettings>;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: settings just don't stick
  }
}

// --- saved prompt templates ---------------------------------------------------

// Prompts the user saved to start from again, per intent: tags like <image1> or
// <Picture 1> only mean something to the engine they were written for.
const TEMPLATES_KEY = 'loki.transform.templates.v1';

export interface PromptTemplate {
  id: string;
  name: string;
  prompt: string;
}

type TemplateStore = Partial<Record<IntentId, PromptTemplate[]>>;

const NO_TEMPLATES: PromptTemplate[] = [];
let templates: TemplateStore | null = null;
const templateListeners = new Set<() => void>();

function readTemplates(): TemplateStore {
  if (templates) return templates;
  try {
    const raw = window.localStorage.getItem(TEMPLATES_KEY);
    templates = raw ? (JSON.parse(raw) as TemplateStore) : {};
  } catch {
    templates = {};
  }
  return templates;
}

function writeTemplates(next: TemplateStore): void {
  templates = next;
  try {
    window.localStorage.setItem(TEMPLATES_KEY, JSON.stringify(next));
  } catch {
    // storage unavailable: templates last for this session only
  }
  templateListeners.forEach((l) => l());
}

export function loadTemplates(id: IntentId): PromptTemplate[] {
  return readTemplates()[id] || NO_TEMPLATES;
}

/** Saves `prompt` under `name`; a template with the same name (case-insensitive) is replaced. */
export function saveTemplate(id: IntentId, name: string, prompt: string): PromptTemplate | null {
  const n = name.trim();
  const p = prompt.trim();
  if (!n || !p) return null;
  const tpl: PromptTemplate = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, name: n, prompt: p };
  const list = loadTemplates(id).filter((t) => t.name.toLowerCase() !== n.toLowerCase());
  writeTemplates({ ...readTemplates(), [id]: [...list, tpl] });
  return tpl;
}

export function deleteTemplate(id: IntentId, templateId: string): void {
  writeTemplates({ ...readTemplates(), [id]: loadTemplates(id).filter((t) => t.id !== templateId) });
}

export function useTemplates(id: IntentId): PromptTemplate[] {
  return useSyncExternalStore(
    (cb) => {
      templateListeners.add(cb);
      return () => templateListeners.delete(cb);
    },
    () => loadTemplates(id),
    () => NO_TEMPLATES
  );
}

/** Tests: forget the in-memory copy so the next read goes back to localStorage. */
export function resetTemplateCache(): void {
  templates = null;
}

// --- submission ---------------------------------------------------------------

export interface SubmitContext {
  mediaServerBase: string;
  authToken: string | null;
}

/**
 * POST each job to the media server's /create. Resolves with the ids; rejects on the first failure.
 * A first-ever job for an engine that is not installed asks before downloading it (see engine-setup);
 * if that is declined or left running in the background nothing is queued and EngineSetupDeferred is thrown.
 */
export async function submitJobs(
  jobs: JobRequest[],
  ctx: SubmitContext
): Promise<string[]> {
  const setup = await ensureEngines(jobs, ctx.mediaServerBase);
  if (setup !== 'ready') throw new EngineSetupDeferred(setup);
  const ids: string[] = [];
  for (const job of jobs) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (ctx.authToken) headers.Authorization = `Bearer ${ctx.authToken}`;
    // eslint-disable-next-line no-await-in-loop
    const res = await fetch(`${ctx.mediaServerBase}/create`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ input: job.input, fields: job.fields }),
      signal: AbortSignal.timeout(10000),
      redirect: 'error',
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    try {
      // eslint-disable-next-line no-await-in-loop
      const body = (await res.json()) as { id?: string };
      if (body && body.id) ids.push(body.id);
    } catch {
      // older servers may not echo an id
    }
  }
  return ids;
}
