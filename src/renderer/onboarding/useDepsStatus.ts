import { useEffect, useRef, useState } from 'react';
import { fetchStatus, type DepStatus } from './api';

export function useDepsStatus(): { status: DepStatus[]; refresh: () => void; error: string | null } {
  const [status, setStatus] = useState<DepStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const sseRef = useRef<EventSource | null>(null);

  const refresh = async () => {
    try {
      const s = await fetchStatus();
      setStatus(s);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'unknown error');
    }
  };

  useEffect(() => {
    refresh();
    const t = window.setInterval(refresh, 4000);

    const es = new EventSource('/api/deps/models/progress');
    sseRef.current = es;
    es.onmessage = (ev) => {
      try {
        const inst = JSON.parse(ev.data);
        setStatus((cur) =>
          cur.map((s) =>
            s.category === 'model' && s.id === inst.id
              ? { ...s, state: inst.state, detail: inst, error: inst.error }
              : s,
          ),
        );
      } catch {
        /* drop */
      }
    };
    es.onerror = () => { /* EventSource retries on its own */ };

    return () => {
      window.clearInterval(t);
      es.close();
    };
  }, []);

  return { status, refresh, error };
}
