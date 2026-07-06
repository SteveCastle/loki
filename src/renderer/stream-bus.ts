// Shared SSE connection to the media server's /stream endpoint.
//
// Chromium allows only 6 concurrent HTTP/1.1 connections per origin, and every
// EventSource pins one for its lifetime. Three components independently held
// their own /stream connection (toast system, People grid, context palette),
// which — combined with untimed fetches — could exhaust the origin's socket
// pool: every later request to the server queued forever, health checks timed
// out, and the app looked "disconnected" until a restart reset the pool.
//
// This bus owns ONE EventSource and fans events out to any number of
// subscribers. It also centralizes the zombie watchdog: the server pings every
// 30s (stream.KeepAliveInterval), so >90s of silence or a CLOSED readyState
// means the connection is dead and gets rebuilt.

import { mediaServerBase } from './platform';

export type StreamListener = (type: string, event: MessageEvent) => void;
export type StatusListener = (connected: boolean) => void;

// Named SSE event types the server emits (jobqueue + media events). The
// default un-named 'message' events (keep-alive pings) only refresh the
// activity clock.
const EVENT_TYPES = [
  'create',
  'update',
  'delete',
  'media-updated',
  'media-created',
  'stats',
];

let es: EventSource | null = null;
let lastActivityAt = Date.now();
let connected = false;
let watchdog: number | null = null;
const listeners = new Set<StreamListener>();
const statusListeners = new Set<StatusListener>();

function notifyStatus(next: boolean) {
  if (connected === next) return;
  connected = next;
  statusListeners.forEach((cb) => {
    try {
      cb(next);
    } catch {
      // subscriber errors must not break the bus
    }
  });
}

function open() {
  if (es) return;
  try {
    es = new EventSource(`${mediaServerBase}/stream`);
  } catch {
    es = null;
    return;
  }
  const dispatch = (type: string) => (event: Event) => {
    lastActivityAt = Date.now();
    listeners.forEach((cb) => {
      try {
        cb(type, event as MessageEvent);
      } catch {
        // one bad subscriber must not starve the others
      }
    });
  };
  for (const t of EVENT_TYPES) es.addEventListener(t, dispatch(t));
  es.onmessage = () => {
    // Keep-alive ping — activity only.
    lastActivityAt = Date.now();
  };
  es.onopen = () => {
    lastActivityAt = Date.now();
    notifyStatus(true);
  };
  es.onerror = () => {
    // EventSource auto-reconnects on transient errors; only report down when
    // genuinely offline. The watchdog handles silent zombies.
    if (!navigator.onLine) notifyStatus(false);
  };
}

function close() {
  if (!es) return;
  try {
    es.close();
  } catch {
    // closing a dead EventSource can throw in some environments
  }
  es = null;
}

function maybeClose() {
  if (listeners.size === 0 && statusListeners.size === 0) {
    close();
    if (watchdog != null) {
      window.clearInterval(watchdog);
      watchdog = null;
    }
  }
}

function ensureWatchdog() {
  if (watchdog != null) return;
  watchdog = window.setInterval(() => {
    if (listeners.size === 0 && statusListeners.size === 0) return;
    if (!navigator.onLine) return; // pointless to rebuild while offline
    const idleSeconds = (Date.now() - lastActivityAt) / 1000;
    if (!es || es.readyState === 2 /* CLOSED */ || idleSeconds > 90) {
      notifyStatus(false);
      close();
      // Reset the clock so the next tick doesn't immediately re-thrash while
      // the fresh connection is still handshaking.
      lastActivityAt = Date.now();
      open();
    }
  }, 15000);
}

// Subscribe to named /stream events. Returns an unsubscribe function.
export function subscribeStream(cb: StreamListener): () => void {
  listeners.add(cb);
  ensureWatchdog();
  open();
  return () => {
    listeners.delete(cb);
    maybeClose();
  };
}

// Subscribe to connection status (true = live SSE connection). The current
// status is delivered immediately.
export function subscribeStreamStatus(cb: StatusListener): () => void {
  statusListeners.add(cb);
  ensureWatchdog();
  open();
  try {
    cb(connected);
  } catch {
    // ignore
  }
  return () => {
    statusListeners.delete(cb);
    maybeClose();
  };
}

// Snapshot of the current connection state — a cheap "is the server there?"
// signal that never costs a socket.
export function streamConnected(): boolean {
  return connected;
}
