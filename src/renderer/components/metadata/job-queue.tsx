import { useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { mediaServerBase, send } from '../../platform';
import { subscribeStream } from '../../stream-bus';
import { getJobTitle } from '../controls/toast-system';
import { useCanWrite } from '../../hooks/useCanWrite';
import './job-queue.css';

type QueueJob = {
  id: string;
  command: string;
  arguments?: string[];
  input: string;
  state: string;
  created_at: string;
  progress_done?: number;
  progress_total?: number;
};

const ACTIVE = new Set(['pending', 'in_progress', 'paused']);
const MAX_FINISHED_SHOWN = 100;

const STATE_LABEL: Record<string, string> = {
  pending: 'Queued',
  in_progress: 'Running',
  paused: 'Paused',
  completed: 'Done',
  cancelled: 'Cancelled',
  error: 'Failed',
};

function timeAgo(iso: string): string {
  const t = Date.parse(iso);
  if (!t || t < 0) return '';
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export default function JobQueue() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const canWrite = useCanWrite();
  const [jobs, setJobs] = useState<QueueJob[]>([]);
  const [loaded, setLoaded] = useState(false);

  const headers = useMemo(() => {
    const h: Record<string, string> = {};
    if (authToken) h.Authorization = `Bearer ${authToken}`;
    return h;
  }, [authToken]);

  const refetch = useCallback(() => {
    fetch(`${mediaServerBase}/jobs/list`, {
      headers,
      signal: AbortSignal.timeout(8000),
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: QueueJob[] | null) => {
        if (Array.isArray(data)) {
          setJobs(data);
          setLoaded(true);
        }
      })
      .catch(() => {
        // Keep what we have; the next stream event refreshes.
      });
  }, [headers]);

  useEffect(() => {
    refetch();
    return subscribeStream((type, event) => {
      try {
        if (type === 'create') {
          const { job } = JSON.parse(event.data) as { job?: QueueJob };
          if (job) {
            setJobs((prev) => [...prev.filter((j) => j.id !== job.id), job]);
          }
        } else if (type === 'update') {
          const { job } = JSON.parse(event.data) as { job?: QueueJob };
          if (!job) return;
          setJobs((prev) =>
            prev.some((j) => j.id === job.id)
              ? prev.map((j) => (j.id === job.id ? { ...j, ...job } : j))
              : [...prev, job]
          );
        } else if (type === 'delete') {
          const { job } = JSON.parse(event.data) as { job?: QueueJob };
          if (job) setJobs((prev) => prev.filter((j) => j.id !== job.id));
        } else if (type === 'progress') {
          const p = JSON.parse(event.data) as {
            id?: string;
            done?: number;
            total?: number;
          };
          if (!p.id) return;
          setJobs((prev) =>
            prev.map((j) =>
              j.id === p.id
                ? {
                    ...j,
                    progress_done: p.done ?? 0,
                    progress_total: p.total ?? 0,
                  }
                : j
            )
          );
        }
      } catch {
        // malformed event — ignore
      }
    });
  }, [refetch]);

  const post = useCallback(
    async (path: string) => {
      try {
        const res = await fetch(`${mediaServerBase}${path}`, {
          method: 'POST',
          headers,
          signal: AbortSignal.timeout(8000),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      } catch (e) {
        console.error('Job queue action failed:', path, e);
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'error',
            title: 'Job action failed',
            message: 'Could not communicate with job service',
          },
        });
      }
      refetch();
    },
    [headers, libraryService, refetch]
  );

  const { active, finished, finishedTotal } = useMemo(() => {
    const byNewest = (a: QueueJob, b: QueueJob) =>
      Date.parse(b.created_at) - Date.parse(a.created_at);
    const act = jobs.filter((j) => ACTIVE.has(j.state));
    const fin = jobs.filter((j) => !ACTIVE.has(j.state)).sort(byNewest);
    // Running first, then queued/paused in submission order.
    act.sort((a, b) => {
      if ((a.state === 'in_progress') !== (b.state === 'in_progress')) {
        return a.state === 'in_progress' ? -1 : 1;
      }
      return Date.parse(a.created_at) - Date.parse(b.created_at);
    });
    return {
      active: act,
      finished: fin.slice(0, MAX_FINISHED_SHOWN),
      finishedTotal: fin.length,
    };
  }, [jobs]);

  const renderJob = (job: QueueJob) => {
    const total = job.progress_total ?? 0;
    const done = job.progress_done ?? 0;
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    const isActive = ACTIVE.has(job.state);
    return (
      <div className={`jq-job ${job.state}`} key={job.id}>
        <div className="jq-job-main">
          <span className={`jq-dot ${job.state}`} />
          <div className="jq-job-text">
            <span
              className="jq-job-title"
              onClick={() =>
                send('open-external', [`${mediaServerBase}/job/${job.id}`])
              }
              title="Open job details"
            >
              {getJobTitle(job as any)}
            </span>
            <span className="jq-job-meta">
              {STATE_LABEL[job.state] ?? job.state}
              {total > 0 ? ` · ${done}/${total}` : ''}
              {job.created_at ? ` · ${timeAgo(job.created_at)}` : ''}
            </span>
            {total > 0 && isActive && (
              <div className="jq-progress">
                <div
                  className="jq-progress-fill"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
          </div>
        </div>
        {canWrite && (
          <div className="jq-actions">
            {(job.state === 'in_progress' || job.state === 'pending') && (
              <button
                type="button"
                title="Pause after the current item"
                onClick={() => post(`/job/${job.id}/pause`)}
              >
                ⏸
              </button>
            )}
            {job.state === 'paused' && (
              <button
                type="button"
                title="Resume"
                onClick={() => post(`/job/${job.id}/resume`)}
              >
                ▶
              </button>
            )}
            {isActive && (
              <button
                type="button"
                title="Cancel job"
                onClick={() => post(`/job/${job.id}/cancel`)}
              >
                ■
              </button>
            )}
            {!isActive && (
              <button
                type="button"
                title="Run again"
                onClick={() => post(`/job/${job.id}/copy`)}
              >
                ↻
              </button>
            )}
            {!isActive && (
              <button
                type="button"
                title="Remove from list"
                onClick={() => post(`/job/${job.id}/remove`)}
              >
                ×
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="JobQueue">
      <div className="jq-header">
        <span>
          {active.length} active · {finishedTotal} finished
        </span>
        {canWrite && finishedTotal > 0 && (
          <button
            type="button"
            className="jq-clear"
            onClick={() => post('/jobs/clear')}
          >
            Clear finished
          </button>
        )}
      </div>
      {!loaded && <div className="jq-empty">Loading…</div>}
      {loaded && jobs.length === 0 && (
        <div className="jq-empty">The job queue is empty.</div>
      )}
      {active.length > 0 && (
        <div className="jq-section">
          <div className="jq-section-title">Active</div>
          {active.map(renderJob)}
        </div>
      )}
      {finished.length > 0 && (
        <div className="jq-section">
          <div className="jq-section-title">Finished</div>
          {finished.map(renderJob)}
          {finishedTotal > finished.length && (
            <div className="jq-empty">
              + {finishedTotal - finished.length} older jobs
            </div>
          )}
        </div>
      )}
    </div>
  );
}
