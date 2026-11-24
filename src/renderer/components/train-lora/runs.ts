// "Continue a run": the server's LoRA training runs (GET /api/lora-runs) and the train-lora job that continues one
// from a checkpoint (the task's `resume` option; loki-tune keeps the run's images, captions, model and rank, and
// takes new `steps` / `lr`). An earlier checkpoint than the run's last branches into <run>-from<step>.
import { useCallback, useEffect, useState } from 'react';
import { isElectron, mediaServerBase } from '../../platform';
import { serverAuthHeaders } from '../../server-auth';
import type { TrainLoraJob } from './plan';

export interface RunCheckpoint {
  step: number;
  dir: string;
  lora?: string;
  /** preview images at this checkpoint: _0 = a training caption, _1 = a new scene */
  samples: string[];
}

export interface LoraRun {
  name: string;
  dir: string;
  kind?: string;
  trigger?: string;
  model?: string;
  rank?: number;
  resolution?: number;
  lr?: number;
  /** the planned total */
  steps: number;
  images: number;
  /** the last step logged */
  step: number;
  loss?: number;
  finished: boolean;
  checkpoints: RunCheckpoint[];
  modified: string;
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

export function parseLoraRuns(body: unknown): LoraRun[] {
  const o = body && typeof body === 'object' ? (body as Record<string, unknown>) : {};
  const runs: LoraRun[] = [];
  (Array.isArray(o.runs) ? o.runs : []).forEach((v) => {
    if (!v || typeof v !== 'object') return;
    const r = v as Record<string, unknown>;
    const dir = str(r.dir);
    if (!dir) return;
    const checkpoints: RunCheckpoint[] = (Array.isArray(r.checkpoints) ? r.checkpoints : [])
      .filter((c): c is Record<string, unknown> => !!c && typeof c === 'object' && typeof (c as any).dir === 'string')
      .map((c) => ({
        step: num(c.step) ?? 0,
        dir: str(c.dir),
        lora: str(c.lora) || undefined,
        samples: (Array.isArray(c.samples) ? c.samples : []).filter((x): x is string => typeof x === 'string'),
      }))
      .sort((a, b) => a.step - b.step);
    runs.push({
      name: str(r.name) || dir.split(/[\\/]/).pop() || dir,
      dir,
      kind: str(r.kind) || undefined,
      trigger: str(r.trigger) || undefined,
      model: str(r.model) || undefined,
      rank: num(r.rank),
      resolution: num(r.resolution),
      lr: num(r.lr),
      steps: num(r.steps) ?? 0,
      images: num(r.images) ?? 0,
      step: num(r.step) ?? 0,
      loss: num(r.loss),
      finished: r.finished === true,
      checkpoints,
      modified: str(r.modified),
    });
  });
  return runs;
}

/** Only runs with a checkpoint can be continued. */
export function continuable(runs: LoraRun[]): LoraRun[] {
  return runs.filter((r) => r.checkpoints.length > 0);
}

export interface ContinueSettings {
  /** the checkpoint folder to continue from */
  checkpoint: string;
  /** the new total (more than the checkpoint's step) */
  steps: number;
  /** 0 = the run's own */
  lr: number;
  /** 0 = 250 */
  saveEvery: number;
}

/** A sensible continuation: 1000 more steps (a style at least to 2500 in all), the run's learning rate, except that
 * a style trained at the old 5e-5 default goes to 1e-4. */
export function continueDefaults(run: LoraRun, cp: RunCheckpoint = run.checkpoints[run.checkpoints.length - 1]): ContinueSettings {
  const style = run.kind === 'style';
  const steps = Math.max(cp.step + 1000, style ? 2500 : 0);
  const lr = style && run.lr !== undefined && run.lr < 1e-4 ? 1e-4 : 0;
  return { checkpoint: cp.dir, steps: Math.ceil(steps / 250) * 250, lr, saveEvery: 0 };
}

/** The checkpoint is not the run's last: continuing branches into a new run folder. */
export function branchesFrom(run: LoraRun, cp: RunCheckpoint): string | null {
  const last = run.checkpoints[run.checkpoints.length - 1];
  return last && cp.step < last.step ? `${run.name}-from${cp.step}` : null;
}

/** POST /create body: the train-lora task with `resume` (no images: they are the run's). */
export function buildContinueJob(c: ContinueSettings): TrainLoraJob {
  const fields: Record<string, string> = { resume: c.checkpoint, steps: String(Math.round(c.steps)) };
  if (c.lr > 0) fields.lr = String(c.lr);
  if (c.saveEvery > 0) fields.saveevery = String(Math.round(c.saveEvery));
  return { input: 'train-lora', fields };
}

export interface TrainLoraRunsState {
  runs: LoraRun[];
  loaded: boolean;
  error?: string;
}

export function useLoraRuns(enabled: boolean, authToken?: string | null): TrainLoraRunsState & { refresh: () => void } {
  const [state, setState] = useState<TrainLoraRunsState>({ runs: [], loaded: false });
  const load = useCallback(async () => {
    try {
      const res = await fetch(`${mediaServerBase}/api/lora-runs`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : serverAuthHeaders(),
        credentials: isElectron ? 'omit' : 'include',
        signal: AbortSignal.timeout(20000),
      });
      if (!res.ok) throw new Error(res.status === 404 ? 'This server cannot list training runs yet.' : `The run list failed (${res.status})`);
      setState({ runs: parseLoraRuns(await res.json()), loaded: true });
    } catch (e) {
      setState((s) => ({ ...s, loaded: true, error: e instanceof Error ? e.message : String(e) }));
    }
  }, [authToken]);
  useEffect(() => {
    if (enabled) load();
  }, [enabled, load]);
  return { ...state, refresh: load };
}
