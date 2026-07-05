import { useEffect, useRef, useState } from 'react';
import { fetchStatus, type DepStatus } from './api';

interface Options {
  /** Origin prefix for the deps API ('' = same origin). */
  base?: string;
  /**
   * Open the live progress EventSource. Leave off for lightweight consumers
   * (task chips, per-file buttons) — the 4s poll already carries progress in
   * `detail`, and each EventSource holds a per-host connection slot.
   */
  sse?: boolean;
}

export function useDepsStatus(opts: Options = {}): { status: DepStatus[]; refresh: () => void; error: string | null } {
  const { base = '', sse = true } = opts;
  const [status, setStatus] = useState<DepStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const sseRef = useRef<EventSource | null>(null);

  const refresh = async () => {
    try {
      const s = await fetchStatus(base);
      setStatus(s);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'unknown error');
    }
  };

  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, 4000);

    let es: EventSource | null = null;
    if (sse) {
      es = new EventSource(`${base}/api/deps/models/progress`);
      sseRef.current = es;
      es.onmessage = (ev) => {
        try {
          const inst = JSON.parse(ev.data);
          setStatus((cur) =>
            cur.map((s) =>
              (s.category === 'model' || s.category === 'tool') && s.id === inst.id
                ? { ...s, state: inst.state, detail: inst, error: inst.error }
                : s,
            ),
          );
        } catch {
          /* drop */
        }
      };
      es.onerror = () => { /* EventSource retries on its own */ };
    }

    return () => {
      window.clearInterval(t);
      es?.close();
    };
  }, [base, sse]);

  return { status, refresh, error };
}
