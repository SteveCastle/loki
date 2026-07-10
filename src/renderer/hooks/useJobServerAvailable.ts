// Shared "is the job server reachable?" signal for components that gate UI on
// it (generate description/tags/transcript, etc.).
//
// History: each of those components used to fire its own one-shot /health
// probe on mount with a short timeout and latch the result in local state
// forever. Under socket-pool pressure (Chromium's 6-connections-per-origin
// cap — see stream-bus.ts) a probe could lose the race, latch `false`, and
// that one component showed "Job Service Required" until remount while the
// rest of the app worked fine.
//
// This hook fixes both halves:
// - A live SSE connection on the shared stream bus is proof the server is
//   reachable without spending a socket; /health is only probed while the bus
//   is down.
// - The result never latches: failed probes retry on an interval, and a bus
//   reconnect flips availability back to true immediately.
// - All mounted instances share ONE module-level probe loop and one answer,
//   instead of racing N concurrent /health fetches into the same pool.

import { useEffect, useState } from 'react';
import { mediaServerBase } from '../platform';
import { streamConnected, subscribeStreamStatus } from '../stream-bus';

type Listener = (available: boolean | null) => void;

const listeners = new Set<Listener>();
let available: boolean | null = null;
let retryTimer: number | null = null;
let probeInFlight = false;
let unsubscribeStream: (() => void) | null = null;
// Latest auth token from any mounted instance; they all read the same value
// from the state machine, so last-writer-wins is safe.
let authToken: string | null = null;

function setAvailable(next: boolean | null) {
  if (available === next) return;
  available = next;
  listeners.forEach((cb) => {
    try {
      cb(next);
    } catch {
      // one bad subscriber must not break the others
    }
  });
}

function clearRetry() {
  if (retryTimer != null) {
    window.clearTimeout(retryTimer);
    retryTimer = null;
  }
}

async function probe() {
  if (probeInFlight || listeners.size === 0) return;
  if (streamConnected()) {
    setAvailable(true);
    return;
  }
  probeInFlight = true;
  let ok = false;
  try {
    const headers: HeadersInit = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    const res = await fetch(`${mediaServerBase}/health`, {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000),
    });
    ok = res.ok;
  } catch {
    ok = false;
  }
  probeInFlight = false;
  if (listeners.size === 0) return;
  // The bus may have come up while the probe was in flight; trust it over a
  // probe that lost a socket-pool race.
  if (streamConnected()) {
    setAvailable(true);
    return;
  }
  setAvailable(ok);
  if (!ok && retryTimer == null) {
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      probe();
    }, 4000);
  }
}

function start() {
  if (unsubscribeStream) return;
  unsubscribeStream = subscribeStreamStatus((connected) => {
    if (connected) {
      clearRetry();
      setAvailable(true);
    } else {
      // The bus dropping isn't proof the server is down (it may just be
      // reconnecting) — probe to find out instead of flipping to false.
      probe();
    }
  });
}

function stopIfIdle() {
  if (listeners.size > 0) return;
  if (unsubscribeStream) {
    unsubscribeStream();
    unsubscribeStream = null;
  }
  clearRetry();
  // Next mount starts from "checking" instead of a stale snapshot.
  available = null;
}

/**
 * Returns `null` while checking, then whether the job server is reachable.
 * Never latches false — recovery is noticed without a remount.
 */
export default function useJobServerAvailable(
  token?: string | null
): boolean | null {
  const [state, setState] = useState<boolean | null>(available);

  useEffect(() => {
    authToken = token ?? null;
  }, [token]);

  useEffect(() => {
    listeners.add(setState);
    setState(available);
    start();
    probe();
    return () => {
      listeners.delete(setState);
      stopIfIdle();
    };
  }, []);

  return state;
}
