// The LoRA files the server can apply (GET /api/loras: its training runs folder and a `loras` folder next to the
// engines' models), for the Generate / Reimagine LoRA picker. Fetched once and shared by every picker, refetched
// when older than a minute (a training run may have finished) and on demand.
import { useCallback, useEffect, useState } from 'react';
import { isElectron, mediaServerBase } from '../../platform';
import { serverAuthHeaders } from '../../server-auth';
import { sameModelFile, type DreamModel } from './dream-models';

export interface LoraFile {
  path: string;
  file: string;
  /** The run name (loki-tune), else the file name without extension. */
  name: string;
  /** The architecture it applies to: 'moss' (MOSS-Image), '' = none the engine can apply. */
  arch: string;
  trigger?: string;
  /** The checkpoint it was trained against (file name). */
  baseModel?: string;
  rank?: number;
  step?: number;
  steps?: number;
  /** An intermediate checkpoint of a training run. */
  checkpoint: boolean;
  run?: string;
  sizeMb: number;
  modified: string;
}

export interface LoraList {
  loras: LoraFile[];
  dirs: string[];
  loaded: boolean;
  loading: boolean;
  error?: string;
}

/** A LoRA chosen for a run: the file and how strongly it applies (loki-dream's FILE:SCALE). */
export interface ChosenLora {
  path: string;
  scale: number;
  /** For display and the trigger hint when the list is not loaded. */
  name?: string;
  trigger?: string;
}

const EMPTY: LoraList = { loras: [], dirs: [], loaded: false, loading: false };

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** The answer's LoRAs, each checked (it is data from the server). */
export function parseLoraList(body: unknown): Pick<LoraList, 'loras' | 'dirs'> {
  const o = body && typeof body === 'object' ? (body as Record<string, unknown>) : {};
  const loras: LoraFile[] = [];
  (Array.isArray(o.loras) ? o.loras : []).forEach((v) => {
    if (!v || typeof v !== 'object') return;
    const r = v as Record<string, unknown>;
    const path = str(r.path);
    if (!path) return;
    const file = str(r.file) || path.split(/[\\/]/).pop() || path;
    loras.push({
      path,
      file,
      name: str(r.name) || file.replace(/\.safetensors$/i, ''),
      arch: str(r.arch),
      trigger: str(r.trigger) || undefined,
      baseModel: str(r.base_model) || undefined,
      rank: num(r.rank),
      step: num(r.step),
      steps: num(r.steps),
      checkpoint: r.checkpoint === true,
      run: str(r.run) || undefined,
      sizeMb: num(r.size_mb) ?? 0,
      modified: str(r.modified),
    });
  });
  const dirs = (Array.isArray(o.dirs) ? o.dirs : []).filter((d): d is string => typeof d === 'string');
  return { loras, dirs };
}

const ARCH_NAMES: Record<string, string> = { moss: 'MOSS-Image', krea2: 'Krea 2', flux2: 'FLUX.2 klein', anima: 'Anima', sdxl: 'SDXL' };

/**
 * Whether a LoRA can be applied with a model (undefined: the server's default model, assumed MOSS-Image), and why
 * not. loki-dream applies LoRAs to MOSS-Image models only; `note` warns when it was trained on another checkpoint.
 */
export function loraFits(l: Pick<LoraFile, 'arch' | 'baseModel'>, m: Pick<DreamModel, 'arch' | 'file' | 'label'> | undefined): { ok: boolean; why?: string; note?: string } {
  if (!l.arch) return { ok: false, why: 'not a LoRA type the engine can apply' };
  const arch = m?.arch || 'moss';
  if (arch !== 'moss') return { ok: false, why: `${ARCH_NAMES[arch] || arch} models take no LoRAs here` };
  if (l.arch !== arch) return { ok: false, why: `made for ${ARCH_NAMES[l.arch] || l.arch}` };
  if (l.baseModel && m?.file && !sameModelFile(l.baseModel, m.file)) {
    return { ok: true, note: `trained on ${l.baseModel.replace(/\.safetensors$/i, '')}` };
  }
  return { ok: true };
}

/** The job's `loras` field: one path:scale per line (scale 1 written as the bare path). */
export function lorasField(chosen: ChosenLora[]): string {
  return chosen
    .filter((c) => c.path && c.scale !== 0)
    .map((c) => (Math.abs(c.scale - 1) < 1e-6 ? c.path : `${c.path}:${Number(c.scale.toFixed(2))}`))
    .join('\n');
}

/** "zkx · 1000 steps", "step 500 of 1000", for a LoRA's line. */
export function loraDetail(l: LoraFile): string {
  const parts: string[] = [];
  if (l.checkpoint && l.step) parts.push(`step ${l.step}${l.steps ? ` of ${l.steps}` : ''}`);
  else if (l.step) parts.push(`${l.step} steps`);
  if (l.rank) parts.push(`rank ${l.rank}`);
  if (l.baseModel) parts.push(`on ${l.baseModel.replace(/\.safetensors$/i, '')}`);
  return parts.join(' · ');
}

let shared: LoraList = EMPTY;
let fetchedAt = 0;
let inflight: Promise<void> | null = null;
const listeners = new Set<(l: LoraList) => void>();

function publish(l: LoraList) {
  shared = l;
  listeners.forEach((fn) => fn(l));
}

async function load(authToken: string | undefined): Promise<void> {
  if (inflight) return inflight;
  publish({ ...shared, loading: true, error: undefined });
  const ctl = new AbortController();
  const giveUp = window.setTimeout(() => ctl.abort(), 30000);
  const run = (async () => {
    try {
      const res = await fetch(`${mediaServerBase}/api/loras`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : serverAuthHeaders(),
        credentials: isElectron ? 'omit' : 'include',
        signal: ctl.signal,
      });
      if (res.status === 404) {
        publish({ ...EMPTY, loaded: true, error: 'This server cannot list LoRAs yet.' });
      } else if (!res.ok) {
        publish({ ...shared, loading: false, error: `The LoRA list failed (${res.status})` });
      } else {
        publish({ ...parseLoraList(await res.json()), loaded: true, loading: false });
        fetchedAt = Date.now();
      }
    } catch (e) {
      publish({ ...shared, loading: false, error: ctl.signal.aborted ? 'The LoRA list took too long' : e instanceof Error ? e.message : String(e) });
    } finally {
      window.clearTimeout(giveUp);
      inflight = null;
    }
  })();
  inflight = run;
  return run;
}

/** The shared LoRA list and a refresh. Nothing loads unless `enabled`. */
export function useLoras(enabled: boolean, authToken?: string | null): { list: LoraList; refresh: () => void } {
  const [list, setList] = useState<LoraList>(shared);
  useEffect(() => {
    listeners.add(setList);
    setList(shared);
    if (enabled && (!shared.loaded || Date.now() - fetchedAt > 60000)) load(authToken || undefined);
    return () => {
      listeners.delete(setList);
    };
  }, [enabled, authToken]);
  const refresh = useCallback(() => {
    if (enabled) load(authToken || undefined);
  }, [enabled, authToken]);
  return { list, refresh };
}

/** Tests: forget the shared list. */
export function resetLorasForTests() {
  shared = EMPTY;
  fetchedAt = 0;
  inflight = null;
}
