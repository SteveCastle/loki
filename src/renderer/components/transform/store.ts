// Tiny external store for the Transform flow: the palette (which unmounts its
// contents when it hides) opens the full options through here, and a root-level host
// renders it. Also persists the last-used settings per intent.
import { useSyncExternalStore } from 'react';
import { INTENTS, intentKind, pinnable, runSummary, settingsFor, type IntentId, type TransformSettings, type JobRequest } from './intents';
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
  /** A whole configuration to start from (the last run), instead of the intent's defaults. */
  settings?: TransformSettings;
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
  'cameraAzimuth',
  'cameraElevation',
  'quality',
  'shake',
  'duration',
  'refSize',
  'reshootModel',
  'noAudio',
  'keepSoundtrack',
];

export function loadRemembered(id: IntentId): Partial<TransformSettings> {
  const pins = pinnable(id) ? { pinned: loadPins(id) } : {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return pins;
    const all = JSON.parse(raw) as Remembered;
    return { ...(all[id] || {}), ...pins };
  } catch {
    return pins;
  }
}

// --- pinned references ------------------------------------------------------------

// Per intent (Edit, Direct), saved the moment they change: a pinned reference is there the next time the flow
// opens, whatever is selected.
const PINS_KEY = 'loki.transform.pins.v1';

export function loadPins(id: IntentId): string[] {
  try {
    const raw = window.localStorage.getItem(PINS_KEY);
    const all = raw ? (JSON.parse(raw) as Partial<Record<IntentId, string[]>>) : {};
    return Array.isArray(all[id]) ? (all[id] as string[]) : [];
  } catch {
    return [];
  }
}

export function savePins(id: IntentId, pins: string[]): void {
  try {
    const raw = window.localStorage.getItem(PINS_KEY);
    const all = raw ? (JSON.parse(raw) as Partial<Record<IntentId, string[]>>) : {};
    all[id] = pins;
    window.localStorage.setItem(PINS_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: pins last for this flow only
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

// Configurations the user saved to start from again ("saved tasks"), per intent: tags like <image1> or
// <Picture 1> only mean something to the engine they were written for. A custom intent's is its prompt (plus the
// settings it goes with); a preset's (Restore, Upscale, New angle…) is its settings alone, saved from a result's
// metadata (recipe.ts).
const TEMPLATES_KEY = 'loki.transform.templates.v1';

export interface PromptTemplate {
  id: string;
  name: string;
  /** The custom prompt; '' for a preset intent's saved settings. */
  prompt: string;
  /**
   * The rest of the configuration the prompt goes with (output size, quality, variations, seed…), captured when it
   * is saved and refreshed each time it runs. Absent on prompts saved before this existed.
   */
  settings?: Partial<TransformSettings>;
}

// Everything but the prompt text itself (the template's own field). A custom intent's hand-written review prompt is
// a one-off rewrite and is dropped; a preset's is what the saved task runs instead of the built-in text, so it stays.
function templateConfig(id: IntentId, s: TransformSettings): Partial<TransformSettings> {
  const rest: Partial<TransformSettings> = { ...s };
  delete rest.prompt;
  if (intentKind(id) === 'custom' || rest.manualPrompt === null) delete rest.manualPrompt;
  return rest;
}

/** The settings to start from for a saved prompt: its saved configuration, else this intent's remembered one. */
export function templateSettings(id: IntentId, t: PromptTemplate): TransformSettings {
  const manual = intentKind(id) === 'custom' ? null : t.settings?.manualPrompt ?? null;
  return { ...settingsFor(id, loadRemembered(id)), ...(t.settings || {}), prompt: t.prompt, manualPrompt: manual };
}

/** How a saved task reads in a list: its prompt, else (a preset's) what its settings do. */
export function templateSummary(id: IntentId, t: PromptTemplate): string {
  return t.prompt || runSummary(id, templateSettings(id, t));
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

/**
 * Saves `prompt` under `name`, with the configuration it goes with when `settings` is given; a template with the
 * same name (case-insensitive) is replaced. A custom intent needs a prompt; a preset's saved task may have none.
 */
export function saveTemplate(id: IntentId, name: string, prompt: string, settings?: TransformSettings): PromptTemplate | null {
  const n = name.trim();
  const p = prompt.trim();
  if (!n || (!p && intentKind(id) === 'custom')) return null;
  const tpl: PromptTemplate = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, name: n, prompt: p };
  if (settings) tpl.settings = templateConfig(id, settings);
  const list = loadTemplates(id).filter((t) => t.name.toLowerCase() !== n.toLowerCase());
  writeTemplates({ ...readTemplates(), [id]: [...list, tpl] });
  return tpl;
}

/** After a run: saved prompts with exactly this text take on the configuration it ran with (size, quality…). */
export function rememberTemplateRun(id: IntentId, settings: TransformSettings): void {
  const p = settings.prompt.trim();
  const list = loadTemplates(id);
  if (!p || !list.some((t) => t.prompt === p)) return;
  const config = templateConfig(id, settings);
  writeTemplates({ ...readTemplates(), [id]: list.map((t) => (t.prompt === p ? { ...t, settings: config } : t)) });
}

/** Deletes a saved task; returns it and where it was, so it can be put back (restoreTemplate). */
export function deleteTemplate(id: IntentId, templateId: string): { template: PromptTemplate; index: number } | null {
  const list = loadTemplates(id);
  const index = list.findIndex((t) => t.id === templateId);
  if (index < 0) return null;
  writeTemplates({ ...readTemplates(), [id]: list.filter((t) => t.id !== templateId) });
  return { template: list[index], index };
}

/** Undo a delete: the saved task goes back where it was (unless one with its id or name has appeared since). */
export function restoreTemplate(id: IntentId, template: PromptTemplate, index: number): void {
  const list = loadTemplates(id);
  if (list.some((t) => t.id === template.id || t.name.toLowerCase() === template.name.toLowerCase())) return;
  const next = [...list];
  next.splice(Math.min(Math.max(0, index), next.length), 0, template);
  writeTemplates({ ...readTemplates(), [id]: next });
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

/** Every saved task, in intent order then saving order. */
export function useAllTemplates(): Array<{ intent: IntentId; t: PromptTemplate }> {
  const all = useSyncExternalStore(
    (cb) => {
      templateListeners.add(cb);
      return () => templateListeners.delete(cb);
    },
    readTemplates,
    () => NO_STORE
  );
  return INTENTS.flatMap((i) => (all[i.id] || NO_TEMPLATES).map((t) => ({ intent: i.id, t })));
}

const NO_STORE: TemplateStore = {};

/** Tests: forget the in-memory copy so the next read goes back to localStorage. */
export function resetTemplateCache(): void {
  templates = null;
}

// --- last run -------------------------------------------------------------------

// The most recent run's intent and every setting (prompt included), so it can be re-run on a new selection or
// loaded and adjusted. Only used when the user picks it: it never pre-fills an ordinary flow.
const LAST_RUN_KEY = 'loki.transform.lastRun.v1';

export interface LastRun {
  intent: IntentId;
  settings: TransformSettings;
  at: number;
}

let lastRun: LastRun | null | undefined;
const lastRunListeners = new Set<() => void>();

export function loadLastRun(): LastRun | null {
  if (lastRun !== undefined) return lastRun;
  try {
    const raw = window.localStorage.getItem(LAST_RUN_KEY);
    const parsed = raw ? (JSON.parse(raw) as LastRun) : null;
    lastRun = parsed && parsed.intent && parsed.settings ? parsed : null;
  } catch {
    lastRun = null;
  }
  return lastRun;
}

export function saveLastRun(intent: IntentId, settings: TransformSettings): void {
  lastRun = { intent, settings: { ...settings }, at: Date.now() };
  try {
    window.localStorage.setItem(LAST_RUN_KEY, JSON.stringify(lastRun));
  } catch {
    // storage unavailable: remembered for this session only
  }
  lastRunListeners.forEach((l) => l());
}

export function useLastRun(): LastRun | null {
  return useSyncExternalStore(
    (cb) => {
      lastRunListeners.add(cb);
      return () => lastRunListeners.delete(cb);
    },
    loadLastRun,
    () => null
  );
}

/** Tests: forget the in-memory copy so the next read goes back to localStorage. */
export function resetLastRunCache(): void {
  lastRun = undefined;
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
