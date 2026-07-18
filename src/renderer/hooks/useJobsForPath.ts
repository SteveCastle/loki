// Tracks which jobs are actively operating on one media path.
//
// Initial answer comes from GET /api/jobs/for-path (the server's path→job
// index); afterwards the list stays live off the shared /stream SSE bus:
//  - 'create'  → refetch (a new job's membership is only known server-side)
//  - 'update'  → merge the job's new state (terminal states make components
//                drop it via isActiveJobState)
//  - 'progress' → merge done/total for the progress label
//
// All fetches are timed out and latest-wins (see renderer-socket-starvation
// notes in stream-bus.ts); SSE consumption goes through the shared bus.

import { useCallback, useEffect, useRef, useState } from 'react';
import { mediaServerBase } from '../platform';
import { subscribeStream } from '../stream-bus';
import { PathJob, mergeJobProgress, mergeJobUpdate } from '../job-status';

export default function useJobsForPath(
  path: string | undefined,
  authToken: string | null | undefined,
  enabled: boolean
): { jobs: PathJob[]; noteJob: (job: PathJob) => void } {
  const [jobs, setJobs] = useState<PathJob[]>([]);
  // Generation counter makes overlapping fetches latest-wins: a slow response
  // for a previous path (or an older refetch) can't clobber newer state.
  const generationRef = useRef(0);

  const refetch = useCallback(() => {
    if (!enabled || !path) return;
    const gen = ++generationRef.current;
    const headers: HeadersInit = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    fetch(
      `${mediaServerBase}/api/jobs/for-path?path=${encodeURIComponent(path)}`,
      { headers, signal: AbortSignal.timeout(5000) }
    )
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { jobs?: PathJob[] } | null) => {
        if (gen !== generationRef.current) return;
        if (data && Array.isArray(data.jobs)) setJobs(data.jobs);
      })
      .catch(() => {
        // Unreachable/slow server: keep whatever we have; the SSE bus (or the
        // next path change) refreshes when the server is back.
      });
  }, [path, authToken, enabled]);

  // Optimistic insert right after this client submits a job, so the UI flips
  // to "Queued…" without waiting for the SSE 'create' round-trip.
  const noteJob = useCallback((job: PathJob) => {
    setJobs((prev) =>
      prev.some((j) => j.id === job.id) ? prev : [...prev, job]
    );
  }, []);

  useEffect(() => {
    generationRef.current++;
    setJobs([]);
    if (!enabled || !path) return undefined;
    refetch();

    const unsubscribe = subscribeStream((type, event) => {
      if (type === 'create') {
        // Membership of the new job is only known server-side; ask again.
        refetch();
        return;
      }
      if (type === 'update') {
        try {
          const payload = JSON.parse(event.data) as { job?: PathJob };
          if (payload.job) setJobs((prev) => mergeJobUpdate(prev, payload.job as PathJob));
        } catch {
          // best-effort parse
        }
        return;
      }
      if (type === 'progress') {
        try {
          const p = JSON.parse(event.data) as {
            id?: string;
            done?: number;
            total?: number;
          };
          if (p.id && typeof p.total === 'number') {
            setJobs((prev) =>
              mergeJobProgress(prev, p.id as string, p.done ?? 0, p.total ?? 0)
            );
          }
        } catch {
          // best-effort parse
        }
      }
    });
    return unsubscribe;
  }, [path, enabled, refetch]);

  return { jobs, noteJob };
}
