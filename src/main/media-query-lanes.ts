// Interruptible connections for the library query (load-media-by-query).
//
// A substring path search (`path LIKE '%x%'`) can't use an index and scans
// every media row: 1–20s on a multi-million-item library. Run on the shared
// connection, that scan held it for its whole duration — tag counts,
// previews and every other renderer query queued behind it — and it could
// not be stopped: a superseded or cancelled search still ran to completion
// and the NEXT search queued behind it too.
//
// So each window (webContents) gets its own read lane: a dedicated connection
// that only ever runs that window's library queries, one at a time, latest
// wins. A new request (or cancel) interrupts the lane's in-flight statement,
// which is safe precisely because nothing else shares the connection. The
// shared connection is the fallback if a lane can't be opened.
import type { WebContents } from 'electron';
import { Database } from './database';

// Returned in place of rows when a request was superseded or cancelled. The
// renderer that sent it has already moved on (its machine state exited), so
// this only needs to be distinguishable from a real (possibly empty) result.
export const QUERY_CANCELLED = { cancelled: true } as const;

interface Lane {
  db: Database;
  // Bumped by every new request and every cancel; a request whose generation
  // is no longer current was superseded.
  generation: number;
  inflight: Promise<unknown> | null;
}

let lanesDbPath = '';
const lanes = new Map<number, Promise<Lane | null>>();

const isInterrupt = (err: unknown) =>
  /SQLITE_INTERRUPT|interrupted/i.test(
    String((err as { message?: string })?.message ?? err)
  );

const closeLane = (lane: Promise<Lane | null>) =>
  lane
    .then((l) => l?.db.close())
    .catch(() => undefined);

// Called whenever load-db (re)opens the library: lanes on the old file are
// closed and reopen lazily against the new one.
export function setMediaQueryDbPath(dbPath: string): void {
  if (dbPath === lanesDbPath) return;
  lanesDbPath = dbPath;
  lanes.forEach(closeLane);
  lanes.clear();
}

export function closeAllMediaQueryLanes(): Promise<unknown> {
  const all = [...lanes.values()].map(closeLane);
  lanes.clear();
  return Promise.all(all);
}

function laneFor(sender: WebContents): Promise<Lane | null> {
  const existing = lanes.get(sender.id);
  if (existing) return existing;
  const dbPath = lanesDbPath;
  const created = (async () => {
    if (!dbPath) return null;
    const db = new Database(dbPath);
    try {
      await db.ready;
    } catch (err) {
      console.warn('[media-query] lane open failed, using shared db:', err);
      db.close().catch(() => undefined);
      return null;
    }
    return { db, generation: 0, inflight: null } as Lane;
  })();
  lanes.set(sender.id, created);
  sender.once('destroyed', () => {
    if (lanes.get(sender.id) === created) lanes.delete(sender.id);
    closeLane(created);
  });
  return created;
}

/**
 * Run a library query on the sender's lane, latest-wins. Resolves to the rows,
 * or QUERY_CANCELLED when a newer request / cancel overtook this one.
 */
export async function runMediaQuery(
  sender: WebContents,
  shared: Database,
  sql: string,
  params: unknown[]
): Promise<any[] | typeof QUERY_CANCELLED> {
  const lane = await laneFor(sender);
  if (!lane) return shared.all(sql, params as any[], 'loadMediaByQuery');

  const gen = ++lane.generation;
  if (lane.inflight) {
    // Stop the superseded scan, and let it fully unwind before starting:
    // SQLite also interrupts statements begun while an interrupt is pending.
    lane.db.interrupt();
    await lane.inflight.catch(() => undefined);
  }
  // Another request (or a cancel) arrived while we waited.
  if (gen !== lane.generation) return QUERY_CANCELLED;

  for (let attempt = 0; ; attempt++) {
    const run = lane.db.all(sql, params as any[], 'loadMediaByQuery');
    lane.inflight = run;
    try {
      return await run;
    } catch (err) {
      if (!isInterrupt(err)) throw err;
      if (gen !== lane.generation) return QUERY_CANCELLED;
      // Interrupted although still current: caught the tail of an earlier
      // interrupt. Run once more rather than fail a live search.
      if (attempt >= 1) throw err;
    } finally {
      if (lane.inflight === run) lane.inflight = null;
    }
  }
}

/** Stop the sender's in-flight library query, if any. */
export async function cancelMediaQuery(sender: WebContents): Promise<void> {
  const pending = lanes.get(sender.id);
  if (!pending) return;
  const lane = await pending;
  if (!lane) return;
  lane.generation++;
  if (lane.inflight) lane.db.interrupt();
}
