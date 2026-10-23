// Picking references from the library for a Scene editor or the Transform modal. The requester opens a PICKER
// WINDOW: the viewer's own bundle booted with ?pick=<JSON PickRequest>, i.e. the whole viewer (search, tags,
// browsing) over the last session, plus a picking banner and a palette action. Its answer goes back to the
// requester and the picker closes, so the window the user was working in never navigates away. Electron opens and
// relays through the main process (open-pick-window / scene-pick-result / scene-pick-request, see pick-window.ts);
// the web build opens a popup of this same SPA and a BroadcastChannel carries the messages (same origin).
import { useSyncExternalStore } from 'react';
import { isElectron, isPickerWindow, on, send } from '../../platform';
import type { RefKind } from './refs';

export interface PickRequest {
  id: string;
  /** Kinds the layer can take. */
  kinds: RefKind[];
  /** What is being picked for, e.g. "Motion 2 · Kitchen scene". */
  label: string;
}

type RequestMsg = { type: 'close'; id: string };
type ResultMsg = { id: string; paths: string[] | null };

const CHANNEL = 'loki-scene-pick';
let bc: BroadcastChannel | null = null;
function channel(): BroadcastChannel | null {
  if (isElectron || typeof BroadcastChannel === 'undefined') return null;
  if (!bc) bc = new BroadcastChannel(CHANNEL);
  return bc;
}

function postResult(msg: ResultMsg) {
  if (isElectron) send('scene-pick-result', [msg]);
  else channel()?.postMessage({ kind: 'result', msg });
}

function listenForResults(fn: (msg: ResultMsg) => void): () => void {
  if (isElectron) return on('scene-pick-result', (...args: unknown[]) => fn(args[0] as ResultMsg));
  const c = channel();
  if (!c) return () => undefined;
  const h = (e: MessageEvent) => e.data?.kind === 'result' && fn(e.data.msg);
  c.addEventListener('message', h);
  return () => c.removeEventListener('message', h);
}

/** The ?pick= value for a request (the picker window boots from it). */
export function pickParam(req: PickRequest): string {
  return JSON.stringify(req);
}

/** The request a ?pick= value carries, or null when it isn't one. */
export function parsePickParam(raw: string | null): PickRequest | null {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw);
    if (typeof v?.id !== 'string' || !Array.isArray(v.kinds)) return null;
    return { id: v.id, kinds: v.kinds, label: typeof v.label === 'string' ? v.label : '' };
  } catch {
    return null;
  }
}

// ---- requester side --------------------------------------------------------------

/**
 * Ask for references in a picker window. Resolves with the picked paths, or null when the pick is cancelled
 * (in the picker, by closing it, or by a newer request). `cancel` withdraws the request and closes the picker.
 */
export function requestLibraryPick(kinds: RefKind[], label: string): { result: Promise<string[] | null>; cancel: () => void } {
  const req: PickRequest = { id: `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, kinds, label };
  let stop: () => void = () => undefined;
  let settle: (v: string[] | null) => void = () => undefined;
  let popup: Window | null = null;
  const result = new Promise<string[] | null>((resolve) => {
    settle = (v) => {
      stop();
      resolve(v);
    };
    stop = listenForResults((msg) => {
      if (msg?.id === req.id) settle(msg.paths);
    });
  });
  if (isElectron) {
    send('open-pick-window', [req]);
  } else {
    const url = `${window.location.pathname}?pick=${encodeURIComponent(pickParam(req))}`;
    // One picker at a time: the shared name reuses an open one, whose pagehide then cancels its request.
    popup = window.open(url, 'loki-pick', 'popup,width=1280,height=860');
    popup?.focus();
  }
  return {
    result,
    cancel: () => {
      if (isElectron) send('scene-pick-request', [{ type: 'close', id: req.id } as RequestMsg]);
      else if (popup && !popup.closed) popup.close();
      settle(null);
    },
  };
}

// ---- picker window side ----------------------------------------------------------

// The open request: fixed for the picker window's life (null in every other window).
let current: PickRequest | null = isPickerWindow
  ? parsePickParam(new URLSearchParams(window.location.search).get('pick'))
  : null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

if (current && !isElectron) {
  const { id } = current;
  // Closing (or replacing) the popup without answering cancels; Electron's main process does this for its windows.
  window.addEventListener('pagehide', () => {
    if (current?.id === id) postResult({ id, paths: null });
  });
}

export function usePickRequest(): PickRequest | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
    () => null
  );
}

/** Which of `paths` the open request can take (by kind), in order, without duplicates. */
export function pickable(paths: string[], req: PickRequest | null, kindOf: (p: string) => RefKind | null): string[] {
  if (!req) return [];
  const seen = new Set<string>();
  return paths.filter((p) => {
    const k = kindOf(p);
    if (!p || !k || !req.kinds.includes(k) || seen.has(p)) return false;
    seen.add(p);
    return true;
  });
}

/** Answer the open request (paths, or null to cancel); the picker window then closes. */
export function answerPick(paths: string[] | null): void {
  if (!current) return;
  postResult({ id: current.id, paths });
  current = null;
  emit();
  // Electron: the main process closes the picker when the result passes through.
  if (!isElectron) window.close();
}
