// Picking references in the viewer for a Scene editor (or the Transform modal): the editor asks, the viewer
// (with its own search, tags and browsing) shows a picking banner and a palette
// action, and the chosen paths come back. Electron relays through the main
// process (scene-pick-request / scene-pick-result, see scene-window.ts); the
// web build's editor popup and viewer tab share an origin, so a
// BroadcastChannel carries the same messages.
import { useSyncExternalStore } from 'react';
import { isElectron, on, send } from '../../platform';
import type { RefKind } from './refs';

export interface PickRequest {
  id: string;
  /** Kinds the layer can take. */
  kinds: RefKind[];
  /** What is being picked for, e.g. "Motion 2 · Kitchen scene". */
  label: string;
}

type RequestMsg = { type: 'open'; req: PickRequest } | { type: 'close'; id: string };
type ResultMsg = { id: string; paths: string[] | null };

const CHANNEL = 'loki-scene-pick';
let bc: BroadcastChannel | null = null;
function channel(): BroadcastChannel | null {
  if (isElectron || typeof BroadcastChannel === 'undefined') return null;
  if (!bc) bc = new BroadcastChannel(CHANNEL);
  return bc;
}

function postRequest(msg: RequestMsg) {
  if (isElectron) send('scene-pick-request', [msg]);
  else channel()?.postMessage({ kind: 'request', msg });
}

function postResult(msg: ResultMsg) {
  if (isElectron) send('scene-pick-result', [msg]);
  else channel()?.postMessage({ kind: 'result', msg });
}

function listen(kind: 'request' | 'result', fn: (msg: any) => void): () => void {
  if (isElectron) return on(kind === 'request' ? 'scene-pick-request' : 'scene-pick-result', (...args: unknown[]) => fn(args[0]));
  const c = channel();
  if (!c) return () => undefined;
  const h = (e: MessageEvent) => e.data?.kind === kind && fn(e.data.msg);
  c.addEventListener('message', h);
  return () => c.removeEventListener('message', h);
}

// ---- editor side ---------------------------------------------------------------

/**
 * Ask the viewer for references. Resolves with the picked paths, or null when the pick is cancelled (from either
 * window). `cancel` withdraws the request.
 */
export function requestLibraryPick(kinds: RefKind[], label: string): { result: Promise<string[] | null>; cancel: () => void } {
  const id = `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  let stop: () => void = () => undefined;
  let settle: (v: string[] | null) => void = () => undefined;
  const result = new Promise<string[] | null>((resolve) => {
    settle = resolve;
    stop = listen('result', (msg: ResultMsg) => {
      if (msg?.id !== id) return;
      stop();
      resolve(msg.paths);
    });
  });
  postRequest({ type: 'open', req: { id, kinds, label } });
  return {
    result,
    cancel: () => {
      stop();
      postRequest({ type: 'close', id });
      settle(null);
    },
  };
}

// ---- viewer side ---------------------------------------------------------------

let current: PickRequest | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
let subscribed: (() => void) | null = null;
// A pick asked for by this window itself (the Transform modal): answered here, never relayed.
let local: { id: string; resolve: (paths: string[] | null) => void } | null = null;

function endLocal(paths: string[] | null) {
  const l = local;
  local = null;
  l?.resolve(paths);
}

/** Start receiving pick requests (the viewer calls this once; idempotent). */
export function listenForPickRequests(): void {
  if (subscribed) return;
  subscribed = listen('request', (msg: RequestMsg) => {
    if (msg?.type === 'open') {
      // An editor window's request takes over from one of our own.
      endLocal(null);
      current = msg.req;
    } else if (msg?.type === 'close' && current?.id === msg.id) current = null;
    else return;
    emit();
  });
}

/**
 * Pick references in this same window (the viewer stays usable around a hidden modal): the banner and the palette
 * action appear as for an editor window's request. Resolves with the paths, or null when cancelled.
 */
export function requestLocalPick(kinds: RefKind[], label: string): { result: Promise<string[] | null>; cancel: () => void } {
  endLocal(null);
  const id = `l${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const result = new Promise<string[] | null>((resolve) => {
    local = { id, resolve };
  });
  current = { id, kinds, label };
  emit();
  return {
    result,
    cancel: () => {
      if (local?.id !== id) return;
      endLocal(null);
      current = null;
      emit();
    },
  };
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

/** Answer the open request: paths, or null to cancel. */
export function answerPick(paths: string[] | null): void {
  if (!current) return;
  if (local?.id === current.id) endLocal(paths);
  else postResult({ id: current.id, paths });
  current = null;
  emit();
}
