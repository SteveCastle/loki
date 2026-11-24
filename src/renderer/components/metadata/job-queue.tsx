import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { mediaServerBase, send } from '../../platform';
import { subscribeStream } from '../../stream-bus';
import { getJobTitle } from '../controls/toast-system';
import { useCanWrite } from '../../hooks/useCanWrite';
import { baseName, jobInputPaths, navigateToMedia } from '../../job-paths';
import {
  formatClock,
  formatDuration,
  jobRunTiming,
  jobTimesTooltip,
} from '../../job-timing';
import './job-queue.css';

export type QueueJob = {
  id: string;
  command: string;
  arguments?: string[];
  input: string;
  state: string;
  created_at: string;
  /** First time the job began running (kept through pause/resume). */
  started_at?: string;
  claimed_at?: string;
  /** When it reached any terminal state (completed/error/cancelled). */
  completed_at?: string;
  /** started_at → completed_at; null until both exist. */
  duration_ms?: number | null;
  progress_done?: number;
  progress_total?: number;
  /** A pause was requested and the job hasn't parked yet. */
  pause_requested?: boolean;
  /** Files the job wrote (registered outputs). */
  output_files?: string[] | null;
};

/** Paths shown per kind before "+N more". */
const FILES_SHOWN = 3;

/**
 * A job's input and output files as clickable rows: clicking one shows it in
 * the viewer. Long lists show the first few and expand into a scroll box.
 */
function JobFiles({
  job,
  onOpen,
}: {
  job: QueueJob;
  onOpen: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const inputs = jobInputPaths(job.input);
  const outputs = (job.output_files || []).filter(Boolean);
  if (!inputs.length && !outputs.length) return null;
  const hidden =
    Math.max(0, inputs.length - FILES_SHOWN) +
    Math.max(0, outputs.length - FILES_SHOWN);
  const rows = (kind: 'in' | 'out', paths: string[]) =>
    (expanded ? paths : paths.slice(0, FILES_SHOWN)).map((p, i) => (
      <button
        // eslint-disable-next-line react/no-array-index-key
        key={`${kind}-${i}-${p}`}
        type="button"
        className={`jq-file ${kind}`}
        title={`Show ${p}`}
        onClick={() => onOpen(p)}
      >
        <span className="jq-file-kind">{kind === 'in' ? 'in' : 'out'}</span>
        <span className="jq-file-name">{baseName(p)}</span>
      </button>
    ));
  return (
    <div className={`jq-files${expanded ? ' expanded' : ''}`}>
      {rows('in', inputs)}
      {rows('out', outputs)}
      {(hidden > 0 || expanded) && (
        <button
          type="button"
          className="jq-files-more"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? 'show less' : `+ ${hidden} more`}
        </button>
      )}
    </div>
  );
}

/** Live "time since start" for a running job, re-rendered once a second. */
function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  return <>{formatDuration(now - since)}</>;
}

/**
 * Start → end and duration on one line: "14:02:11 → 14:05:13 · 3m 02s",
 * or "14:02:11 → now · 1m 05s" (ticking) while running. Nothing for a job
 * that never started.
 */
function JobTimes({ job }: { job: QueueJob }) {
  const t = jobRunTiming(job);
  if (t.startedMs === null) return null;
  let end = '';
  if (t.endedMs !== null) end = formatClock(t.endedMs);
  else if (t.live) end = 'now';
  return (
    <span className="jq-job-times" title={jobTimesTooltip(job)}>
      {formatClock(t.startedMs)}
      {end ? ` → ${end}` : ''}
      {t.durationMs !== null && ` · ${formatDuration(t.durationMs)}`}
      {t.live && (
        <>
          {' · '}
          <Elapsed since={t.startedMs} />
        </>
      )}
    </span>
  );
}

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

export interface JobFilter {
  /** Words that must all appear in the job's command, arguments, input, id or title. */
  q: string;
  /** Commands to show (typeMode 'show') or hide ('hide'); none = every type. */
  types: string[];
  typeMode: 'show' | 'hide';
  /** A state, 'active' (pending, running, paused) or '' = every state. */
  state: string;
}

export const NO_FILTER: JobFilter = { q: '', types: [], typeMode: 'show', state: '' };

export const STATE_FILTERS: Array<{ value: string; label: string }> = [
  { value: '', label: 'All states' },
  { value: 'active', label: 'Active (queued, running, paused)' },
  { value: 'pending', label: 'Queued' },
  { value: 'in_progress', label: 'Running' },
  { value: 'paused', label: 'Paused' },
  { value: 'completed', label: 'Done' },
  { value: 'error', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
];

/** Whether a job passes the filter (the server Jobs page's filters: search, type, state). */
export function jobMatches(job: QueueJob, f: JobFilter, title = ''): boolean {
  if (f.types.length && f.types.includes(job.command) !== (f.typeMode === 'show')) return false;
  if (f.state === 'active') {
    if (!ACTIVE.has(job.state)) return false;
  } else if (f.state && job.state !== f.state) {
    return false;
  }
  const words = f.q.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = [job.command, ...(job.arguments || []), job.input, job.id, title]
    .join(' ')
    .toLowerCase();
  return words.every((w) => hay.includes(w));
}

// The filter is a per-viewer convenience: kept in this browser only.
const FILTER_KEY = 'lowkey.jobQueue.filter';

function loadFilter(): JobFilter {
  try {
    const raw = JSON.parse(window.localStorage.getItem(FILTER_KEY) || 'null');
    if (raw && typeof raw === 'object') {
      let types: string[] = [];
      if (Array.isArray(raw.types)) {
        types = raw.types.filter((t: unknown) => typeof t === 'string' && t);
      } else if (typeof raw.type === 'string' && raw.type) {
        types = [raw.type]; // saved before types could be combined
      }
      return {
        q: typeof raw.q === 'string' ? raw.q : '',
        types,
        typeMode: raw.typeMode === 'hide' ? 'hide' : 'show',
        state: STATE_FILTERS.some((o) => o.value === raw.state) ? raw.state : '',
      };
    }
  } catch {
    // no storage: start unfiltered
  }
  return NO_FILTER;
}

function saveFilter(f: JobFilter) {
  try {
    window.localStorage.setItem(FILTER_KEY, JSON.stringify(f));
  } catch {
    // not kept
  }
}

/** The type filter's button text: every type, the checked ones, or every type but them. */
export function typesLabel(f: Pick<JobFilter, 'types' | 'typeMode'>): string {
  if (!f.types.length) return 'All types';
  const names = f.types.length <= 2 ? f.types.join(', ') : `${f.types.length} types`;
  return f.typeMode === 'hide' ? `All but ${names}` : names;
}

/** A dropdown of checkboxes, one per job type, with Show / Hide for the checked ones. */
function TypeFilter({
  options,
  filter,
  onChange,
}: {
  options: Array<[string, number]>;
  filter: JobFilter;
  onChange: (patch: Partial<JobFilter>) => void;
}) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);
  const toggle = (c: string, on: boolean) =>
    onChange({
      types: on
        ? [...filter.types.filter((t) => t !== c), c]
        : filter.types.filter((t) => t !== c),
    });
  return (
    <div
      className="jq-types"
      ref={box}
      onKeyDown={(e) => {
        if (e.key === 'Escape' && open) {
          e.preventDefault();
          e.stopPropagation();
          setOpen(false);
          button.current?.focus();
        }
      }}
    >
      <button
        type="button"
        ref={button}
        className="jq-types-btn"
        aria-haspopup="true"
        aria-expanded={open}
        title={
          filter.types.length
            ? `${filter.typeMode === 'hide' ? 'Hiding' : 'Showing'}: ${filter.types.join(', ')}`
            : 'Filter by job type'
        }
        onClick={() => setOpen((v) => !v)}
      >
        {typesLabel(filter)} ▾
      </button>
      {open && (
        <div className="jq-types-pop" role="group" aria-label="Job types">
          <div className="jq-types-mode">
            {(['show', 'hide'] as const).map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={filter.typeMode === m}
                title={m === 'show' ? 'Show only the checked types' : 'Show every type but the checked ones'}
                onClick={() => onChange({ typeMode: m })}
              >
                {m === 'show' ? 'Show' : 'Hide'}
              </button>
            ))}
          </div>
          {options.map(([c, n]) => (
            <label key={c} className="jq-types-item">
              <input
                type="checkbox"
                checked={filter.types.includes(c)}
                onChange={(e) => toggle(c, e.target.checked)}
              />
              <span className="jq-types-name">{c}</span>
              <span className="jq-types-n">{n}</span>
            </label>
          ))}
          <button
            type="button"
            className="jq-types-none"
            onClick={() => onChange({ types: [] })}
          >
            All types
          </button>
        </div>
      )}
    </div>
  );
}

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
  const [filter, setFilterState] = useState<JobFilter>(loadFilter);
  // the search box updates at once; the list follows a moment later (fast typing over thousands of jobs)
  const [search, setSearch] = useState(filter.q);
  const setFilter = useCallback((patch: Partial<JobFilter>) => {
    setFilterState((prev) => {
      const next = { ...prev, ...patch };
      saveFilter(next);
      return next;
    });
  }, []);
  useEffect(() => {
    if (search === filter.q) return undefined;
    const t = setTimeout(() => setFilter({ q: search }), 120);
    return () => clearTimeout(t);
  }, [search, filter.q, setFilter]);

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

  // every type in the list, with its count, plus checked ones no job has any more
  const typeOptions = useMemo(() => {
    const counts = new Map<string, number>();
    jobs.forEach((j) => {
      if (j.command) counts.set(j.command, (counts.get(j.command) || 0) + 1);
    });
    filter.types.forEach((t) => {
      if (!counts.has(t)) counts.set(t, 0);
    });
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [jobs, filter.types]);

  const filtering = !!(filter.q || filter.types.length || filter.state);
  const shownJobs = useMemo(
    () =>
      filtering
        ? jobs.filter((j) => jobMatches(j, filter, getJobTitle(j as any)))
        : jobs,
    [jobs, filter, filtering]
  );

  const { active, finished, finishedTotal } = useMemo(() => {
    const byNewest = (a: QueueJob, b: QueueJob) =>
      Date.parse(b.created_at) - Date.parse(a.created_at);
    const act = shownJobs.filter((j) => ACTIVE.has(j.state));
    const fin = shownJobs.filter((j) => !ACTIVE.has(j.state)).sort(byNewest);
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
  }, [shownJobs]);

  const openFile = useCallback(
    (path: string) => navigateToMedia(libraryService as any, path),
    [libraryService]
  );

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
              {job.pause_requested && job.state === 'in_progress'
                ? 'Pausing…'
                : STATE_LABEL[job.state] ?? job.state}
              {total > 0 ? ` · ${done}/${total}` : ''}
              {job.created_at ? ` · queued ${timeAgo(job.created_at)}` : ''}
            </span>
            <JobTimes job={job} />
            {total > 0 && isActive && (
              <div className="jq-progress">
                <div
                  className="jq-progress-fill"
                  style={{ width: `${pct}%` }}
                />
              </div>
            )}
            <JobFiles job={job} onOpen={openFile} />
          </div>
        </div>
        {canWrite && (
          <div className="jq-actions">
            {(job.state === 'in_progress' || job.state === 'pending') && (
              <button
                type="button"
                disabled={!!job.pause_requested}
                title={
                  job.pause_requested
                    ? 'Pausing…'
                    : 'Pause — finished work is kept; a running generation stops and restarts on resume'
                }
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
      {jobs.length > 0 && (
        <div className="jq-filters" role="search">
          <input
            type="search"
            className="jq-search"
            placeholder="Search command, arguments, input or id…"
            aria-label="Search jobs"
            autoComplete="off"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => {
              // Esc in the box clears the search first
              if (e.key === 'Escape' && search) {
                e.preventDefault();
                e.stopPropagation();
                setSearch('');
                setFilter({ q: '' });
              }
            }}
          />
          <TypeFilter options={typeOptions} filter={filter} onChange={setFilter} />
          <select
            aria-label="Job state"
            value={filter.state}
            onChange={(e) => setFilter({ state: e.target.value })}
          >
            {STATE_FILTERS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
          {filtering && (
            <>
              <span className="jq-count" aria-live="polite">
                Showing {shownJobs.length} of {jobs.length}
              </span>
              <button
                type="button"
                className="jq-filter-clear"
                onClick={() => {
                  setSearch('');
                  setFilter(NO_FILTER);
                }}
              >
                Clear filters
              </button>
            </>
          )}
        </div>
      )}
      {!loaded && <div className="jq-empty">Loading…</div>}
      {loaded && jobs.length === 0 && (
        <div className="jq-empty">The job queue is empty.</div>
      )}
      {filtering && jobs.length > 0 && shownJobs.length === 0 && (
        <div className="jq-empty">No jobs match these filters.</div>
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
