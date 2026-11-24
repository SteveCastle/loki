// Tiny external store for the Train LoRA dialog: the context palette (which unmounts its contents when it hides)
// opens it through here, and a root-level host (TrainLoraHost) renders it.
import { useSyncExternalStore } from 'react';

export interface TrainLoraRequest {
  /** The selection: right-clicked file plus any multi-selection, in order. With `query64`: the images of the current
   * view (for the count and the preview), while the job trains on everything the query matches. */
  paths: string[];
  /** A library query (base64, as the palette's other jobs send it): train on all of its images. */
  query64?: string;
  /** How many images the query matches, when known (null: unknown, e.g. a tag target). */
  count?: number | null;
  /** The query in words ("Entire library", "tag: portrait"). */
  label?: string;
  authToken?: string | null;
  /** open on "Continue a run" (default: "New LoRA" when there are images) */
  mode?: 'new' | 'continue';
  /** with mode continue: the run to pre-select (its folder) */
  run?: string;
}

let current: TrainLoraRequest | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

export function openTrainLora(req: TrainLoraRequest): void {
  current = req;
  emit();
}

export function closeTrainLora(): void {
  if (current === null) return;
  current = null;
  emit();
}

export function useTrainLoraRequest(): TrainLoraRequest | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
    () => null
  );
}
