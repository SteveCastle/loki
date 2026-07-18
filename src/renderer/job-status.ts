// Types and pure helpers for the path→job lookup
// (GET /api/jobs/for-path on the media server, jobs_for_path_api.go).
//
// The server indexes which ACTIVE jobs (pending / in-progress / paused) are
// operating on each media path: path-list jobs from the moment they're
// created, query jobs once they resolve their input at claim time. Components
// use this to swap a per-item action button (e.g. Generate Transcript) for a
// live status indicator, then follow the job over the shared /stream SSE bus.

export type PathJobState =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'cancelled'
  | 'error'
  | 'paused';

export interface PathJob {
  id: string;
  command: string;
  arguments?: string[];
  state: PathJobState;
  progress_done?: number;
  progress_total?: number;
  created_at?: string;
}

export function isActiveJobState(state: string): boolean {
  return state === 'pending' || state === 'in_progress' || state === 'paused';
}

// Returns the newest ACTIVE job for one of the given commands (the endpoint
// returns jobs in queue order, so the last match is the latest submission).
export function pickActiveJob(
  jobs: PathJob[],
  commands: string[]
): PathJob | null {
  for (let i = jobs.length - 1; i >= 0; i--) {
    const j = jobs[i];
    if (isActiveJobState(j.state) && commands.includes(j.command)) return j;
  }
  return null;
}

// Short status line shown in place of the action button, e.g.
// "Queued…", "Transcribing… 2/5", "Paused".
export function jobStatusLabel(job: PathJob, verb = 'Working'): string {
  if (job.state === 'pending') return 'Queued…';
  if (job.state === 'paused') return 'Paused';
  const total = job.progress_total ?? 0;
  if (total > 1) return `${verb}… ${job.progress_done ?? 0}/${total}`;
  return `${verb}…`;
}

// Merges an SSE job "update" event into the tracked list. Unknown ids are
// ignored (the 'create' refetch handles membership changes).
export function mergeJobUpdate(jobs: PathJob[], update: PathJob): PathJob[] {
  if (!jobs.some((j) => j.id === update.id)) return jobs;
  return jobs.map((j) => (j.id === update.id ? { ...j, ...update } : j));
}

// Merges an SSE "progress" event ({id, done, total}) into the tracked list.
export function mergeJobProgress(
  jobs: PathJob[],
  id: string,
  done: number,
  total: number
): PathJob[] {
  if (!jobs.some((j) => j.id === id)) return jobs;
  return jobs.map((j) =>
    j.id === id ? { ...j, progress_done: done, progress_total: total } : j
  );
}
