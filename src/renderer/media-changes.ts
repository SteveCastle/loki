// What the viewer does when a job (or Studio) writes media files.
//
// The server announces finished output with two stream events:
//   media-created  {jobId, paths}  new files beside the originals
//   media-updated  {jobId, paths}  files overwritten in place
// and Electron's Studio window reports its saves the same two ways.
//
// The contract, decided here and nowhere else:
//   - Nothing ever moves the cursor or edits the current view. New files are
//     ANNOUNCED: the job's completion toast lists them, each opening in a new
//     window so the user's flow is left alone.
//   - An overwritten file that is on screen right now is RELOADED in place,
//     so the edit shows immediately. It is not announced (the user is
//     already looking at it); other overwritten files are.
//   - Every overwritten file's cached previews are dropped.
//
// This module is pure (planning only); toast-system.tsx applies the plan.

export type MediaChangeKind = 'created' | 'updated';

export interface MediaChange {
  kind: MediaChangeKind;
  paths: string[];
  /** The job that wrote the files; absent for Studio saves and old servers. */
  jobId?: string;
}

export interface MediaChangePlan {
  /** Paths whose cached previews and media URLs must be refreshed. */
  refresh: string[];
  /** The file on screen was overwritten: reload it now. */
  reloadCurrent: boolean;
  /** Paths to offer on a toast (opened in a new window). */
  announce: string[];
}

/**
 * Comparable form of a path: forward slashes, no trailing slash, and
 * case-folded for Windows drive and UNC paths (the server and the viewer can
 * spell the same file with different separators or drive-letter case).
 */
export function normalizePath(p: string): string {
  let s = p.trim().replace(/\\/g, '/');
  if (s.length > 1) s = s.replace(/\/+$/, '');
  return /^([A-Za-z]:\/|\/\/)/.test(s) ? s.toLowerCase() : s;
}

export function samePath(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false;
  return normalizePath(a) === normalizePath(b);
}

/**
 * Parse a media-created / media-updated stream payload. The stream wraps the
 * task's JSON in `{msg: "<json>"}`; a bare object is accepted too. Returns
 * null for anything without paths.
 */
export function parseMediaChange(kind: MediaChangeKind, data: string): MediaChange | null {
  try {
    const outer = JSON.parse(data);
    const inner = typeof outer?.msg === 'string' ? JSON.parse(outer.msg) : outer;
    const paths = Array.isArray(inner?.paths)
      ? inner.paths.filter((p: unknown): p is string => typeof p === 'string' && p.trim() !== '')
      : [];
    if (paths.length === 0) return null;
    const jobId = typeof inner.jobId === 'string' && inner.jobId ? inner.jobId : undefined;
    return { kind, paths: dedupe(paths), jobId };
  } catch {
    return null;
  }
}

export function planMediaChange(change: MediaChange, currentPath: string | null | undefined): MediaChangePlan {
  const paths = dedupe(change.paths);
  if (change.kind === 'created') {
    return { refresh: [], reloadCurrent: false, announce: paths };
  }
  const isCurrent = (p: string) => samePath(p, currentPath);
  return {
    refresh: paths,
    reloadCurrent: paths.some(isCurrent),
    announce: paths.filter((p) => !isCurrent(p)),
  };
}

/** Append paths not already in the list (by normalized path), keeping order. */
export function mergePaths(existing: string[], added: string[]): string[] {
  return dedupe([...existing, ...added]);
}

function dedupe(paths: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const p of paths) {
    const k = normalizePath(p);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(p);
  }
  return out;
}
