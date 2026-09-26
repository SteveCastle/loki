// Duplicates: duplicate-candidate review, rendered inside the taxonomy panel
// as a special case of a tag category — the same shape as People. Groups are
// NOT tags: the server's find-duplicates task records clusters of visually
// identical (or near-identical) media in its own tables, and this panel is
// where the user decides what to do with each one. Clicking a group filters
// the library to its members (a `dupe:<id>` predicate, so it composes with
// every other chip like a tag would); the controls merge the copies the user
// agrees are copies (server-side MergeInto — the same merge behind the
// viewer's Merge action), dismiss a group of look-alikes, or pull one member
// out. Nothing here deletes anything until the user confirms a merge.
//
// The Duplicates entry is always listed (it is the only way to reach this
// panel); the Hide Suggested Tags setting would only apply to per-item
// duplicate chips, which are not rendered on list/detail overlays today.
import { useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import {
  isElectron,
  mediaServerBase,
  mediaServerConfigured,
} from '../../platform';
import { subscribeStream } from '../../stream-bus';
import filter from '../../filter';
import { getFileType, FileTypes } from '../../../file-types';
import { Image } from '../media-viewers/image';
import { Video } from '../media-viewers/video';
import './people-grid.css';
import './duplicates-grid.css';

export const DUPLICATES_CATEGORY = 'Duplicates';

export interface DuplicateMemberItem {
  path: string;
  score: number;
  excluded: boolean;
  anchor: boolean;
  // keeper: the server's default merge target for this group — highest
  // resolution, then largest file (media.PreferredDuplicateKeeper). The
  // same rule the bulk merge task and the merge endpoint apply.
  keeper: boolean;
  width?: number;
  height?: number;
  size?: number;
  elo?: number;
}

export interface DuplicateGroup {
  id: number;
  model: string;
  threshold: number;
  anchorPath: string;
  keeperPath?: string;
  status: 'pending' | 'dismissed';
  createdAt: number;
  updatedAt: number;
  memberCount: number;
  activeCount: number;
  minScore: number;
  members: DuplicateMemberItem[];
}

export interface DuplicateStats {
  pending: number;
  dismissed: number;
  members: number;
  pendingItems: number;
}

type ListStatus = 'pending' | 'dismissed';
// Group order: newest found first, or the biggest clusters first.
type GroupSort = 'newest' | 'members';

interface GroupPage {
  groups: DuplicateGroup[];
  total: number;
  limit: number;
  offset: number;
}

const PAGE_SIZE = 25;
// A merge deletes every other member from disk; big groups take a while.
const MERGE_TIMEOUT_MS = 30 * 60 * 1000;
// Members shown per card before the "+N more" expander.
const MEMBER_PREVIEW_CAP = 8;

function authHeaders(authToken: string | null): HeadersInit {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

async function apiGet<T>(authToken: string | null, path: string): Promise<T> {
  const res = await fetch(`${mediaServerBase}${path}`, {
    headers: authHeaders(authToken),
    credentials: 'include',
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as T;
}

// timeoutMs: merges of big groups delete thousands of files and must not be
// abandoned by the client (the server finishes anyway, but the panel would
// report an error for a merge that succeeded).
async function apiSend<T>(
  authToken: string | null,
  method: 'POST' | 'DELETE',
  path: string,
  body?: unknown,
  timeoutMs = 120000
): Promise<T> {
  const res = await fetch(`${mediaServerBase}${path}`, {
    method,
    headers: {
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...authHeaders(authToken),
    },
    credentials: 'include',
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const text = await res.text();
      if (text) msg = text.trim();
    } catch {
      /* status alone */
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

// useDuplicateStats is the shared pending/dismissed count. Keyed under the
// 'taxonomy' prefix like the people queries so the broad invalidations that
// already refresh the sidebar (DB swaps, tag mutations) refresh this too.
export function useDuplicateStats(enabled = true) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const initSessionId = useSelector(
    libraryService,
    (s) => s.context.initSessionId
  );
  return useQuery<DuplicateStats, Error>(
    ['taxonomy', 'duplicates', 'stats', initSessionId],
    () => apiGet<DuplicateStats>(authToken, '/api/duplicates/stats'),
    { enabled: enabled && !!initSessionId, staleTime: 60_000, retry: 1 }
  );
}

function pct(score: number): string {
  return `${(Math.max(0, Math.min(1, score)) * 100).toFixed(1)}%`;
}

function baseName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

// Library paths and member paths are both the stored spelling, but compare
// separator- and case-insensitively (Windows) to be safe.
function samePath(a: string, b: string): boolean {
  const norm = (p: string) => p.replace(/\\/g, '/').toLowerCase();
  return norm(a) === norm(b);
}

function fmtBytes(n: number): string {
  if (n >= 1024 * 1024 * 1024)
    return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} B`;
}

function MemberPreview({ path }: { path: string }) {
  // Same rule as the list view: with a thumbnail cache in play a GIF is
  // previewed by the video viewer (its cached thumbnail is a video-style
  // still), so pass gifIsVideo = true.
  const type = getFileType(path, true);
  if (type === FileTypes.Video) {
    return (
      <Video
        path={path}
        initialTimestamp={0.5}
        scaleMode="cover"
        orientation="landscape"
        cache="thumbnail_path_600"
        startTime={0}
      />
    );
  }
  return (
    <Image
      path={path}
      scaleMode="cover"
      orientation="landscape"
      cache="thumbnail_path_600"
    />
  );
}

// A two-click "armed" confirmation that disarms itself after a few seconds —
// the same pattern every destructive People/Cleanup button uses.
function useArmed(ms = 4000): [boolean, () => boolean, () => void] {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return undefined;
    const t = window.setTimeout(() => setArmed(false), ms);
    return () => window.clearTimeout(t);
  }, [armed, ms]);
  // Returns true when the click should proceed (second click).
  const fire = () => {
    if (armed) {
      setArmed(false);
      return true;
    }
    setArmed(true);
    return false;
  };
  return [armed, fire, () => setArmed(false)];
}

interface GroupCardProps {
  group: DuplicateGroup;
  canWrite: boolean;
  isDisabled: boolean;
  isActive: boolean;
  busy: boolean;
  // The library cursor's current item. While this group is the active
  // filter, the keep highlight follows it (scrolling the detail view moves
  // the highlight) and clicking a member moves it (see onView).
  cursorPath: string | null;
  // pinPath (member clicks) lands the library cursor on that item once the
  // cluster filter has loaded, so the detail panel shows it.
  onView: (g: DuplicateGroup, pinPath?: string) => void;
  onMerge: (g: DuplicateGroup, keep: string) => Promise<void>;
  onStatus: (g: DuplicateGroup, status: ListStatus) => Promise<void>;
  onExclude: (
    g: DuplicateGroup,
    path: string,
    excluded: boolean
  ) => Promise<void>;
  onForget: (g: DuplicateGroup) => Promise<void>;
}

function GroupCard({
  group,
  canWrite,
  isDisabled,
  isActive,
  busy,
  cursorPath,
  onView,
  onMerge,
  onStatus,
  onExclude,
  onForget,
}: GroupCardProps) {
  const [expanded, setExpanded] = useState(false);
  // Which copy survives a merge. Defaults to the server's keeper (highest
  // resolution, then largest file); a member that is no longer in the group
  // (merged away, excluded) falls back to it too.
  const defaultKeep = group.keeperPath || group.anchorPath;
  const [keep, setKeep] = useState<string>(defaultKeep);
  const [mergeArmed, fireMerge] = useArmed();
  const [forgetArmed, fireForget] = useArmed();
  const active = group.members.filter((m) => !m.excluded);
  // Cursor → highlight: while this group is what the library shows, the
  // highlighted (keep) member is always the one in the detail view.
  const activeKey = active.map((m) => m.path).join('\n');
  useEffect(() => {
    if (!isActive || !cursorPath) return;
    const hit = active.find((m) => samePath(m.path, cursorPath));
    if (hit && hit.path !== keep) setKeep(hit.path);
    // `active` is derived from group.members; activeKey stands in for it.
  }, [isActive, cursorPath, activeKey]);
  const keepPath = active.some((m) => m.path === keep) ? keep : defaultKeep;
  const toMerge = active.filter((m) => m.path !== keepPath).length;
  const shown = expanded
    ? group.members
    : group.members.slice(0, MEMBER_PREVIEW_CAP);
  const hidden = group.members.length - shown.length;
  const dismissed = group.status === 'dismissed';

  return (
    <div
      className={`dupes-group${isActive ? ' active' : ''}${
        dismissed ? ' dismissed' : ''
      }${isDisabled || busy ? ' disabled' : ''}`}
    >
      <div className="dupes-group-head">
        <button
          type="button"
          className="dupes-group-title"
          onClick={() => onView(group)}
          title="Show this group's items in the library (adds a dupe: filter chip)"
        >
          <span className="dupes-group-id">#{group.id}</span>
          <span className="dupes-group-count">
            {group.activeCount} item{group.activeCount === 1 ? '' : 's'}
          </span>
          <span
            className="dupes-group-score"
            title="Lowest similarity to the group's anchor"
          >
            ≥ {pct(group.minScore)}
          </span>
          {group.memberCount > group.activeCount && (
            <span className="dupes-group-excluded">
              {group.memberCount - group.activeCount} excluded
            </span>
          )}
        </button>
        {canWrite && (
          <div className="dupes-group-actions">
            {!dismissed && (
              <button
                type="button"
                className={`dupes-btn dupes-btn-merge${
                  mergeArmed ? ' danger' : ''
                }`}
                disabled={toMerge === 0 || busy}
                onClick={() => {
                  if (fireMerge()) void onMerge(group, keepPath);
                }}
                title={
                  mergeArmed
                    ? `Click again: ${toMerge} file${
                        toMerge === 1 ? '' : 's'
                      } will be deleted from disk after their tags, embeddings, and transcript are merged into ${baseName(
                        keepPath
                      )}`
                    : `Keep ${baseName(
                        keepPath
                      )} and merge the other ${toMerge} into it (deletes them). Asks to confirm.`
                }
              >
                {busy
                  ? `Merging ${toMerge}…`
                  : mergeArmed
                  ? `Confirm — delete ${toMerge}`
                  : `Merge ${toMerge} into kept`}
              </button>
            )}
            <button
              type="button"
              className="dupes-btn"
              disabled={busy}
              onClick={() =>
                void onStatus(group, dismissed ? 'pending' : 'dismissed')
              }
              title={
                dismissed
                  ? 'Put this group back in the review queue'
                  : 'Not duplicates: hide this group. The decision sticks — future scans will not regroup these items.'
              }
            >
              {dismissed ? 'Restore' : 'Not duplicates'}
            </button>
            <button
              type="button"
              className={`dupes-btn dupes-btn-forget${
                forgetArmed ? ' danger' : ''
              }`}
              disabled={busy}
              onClick={() => {
                if (fireForget()) void onForget(group);
              }}
              title={
                forgetArmed
                  ? 'Click again to forget this group (files are untouched; a future scan may regroup them)'
                  : 'Forget this group without deciding. Files are untouched.'
              }
            >
              {forgetArmed ? 'Confirm forget' : 'Forget'}
            </button>
          </div>
        )}
      </div>
      <div className="dupes-members">
        {shown.map((m) => {
          const isKeep = !dismissed && m.path === keepPath && !m.excluded;
          return (
            <div
              key={m.path}
              className={`dupes-member${m.excluded ? ' excluded' : ''}${
                isKeep ? ' keep' : ''
              }`}
              title={`${m.path}\n${pct(m.score)} similar to the anchor${
                m.width && m.height ? `\n${m.width}×${m.height}` : ''
              }\nClick: show this cluster in the library with this item selected${
                canWrite && !dismissed && !m.excluded
                  ? ' (and keep it in a merge)'
                  : ''
              }`}
              onClick={() => {
                // Same cluster filter as the group header, plus the cursor
                // lands on this item so the detail panel shows it.
                if (canWrite && !dismissed && !m.excluded) setKeep(m.path);
                onView(group, m.path);
              }}
            >
              <div className="dupes-member-thumb">
                <MemberPreview path={m.path} />
                {m.anchor && (
                  <span
                    className="dupes-member-badge anchor"
                    title="Anchor: every score is measured against this item"
                  >
                    anchor
                  </span>
                )}
                {isKeep && (
                  <span
                    className="dupes-member-badge keep"
                    title={
                      m.keeper
                        ? 'Survives the merge — the best copy by default (highest resolution, then largest file)'
                        : 'Survives the merge (your pick; the default would be the highest-resolution copy)'
                    }
                  >
                    {m.keeper ? 'keep · best' : 'keep'}
                  </span>
                )}
                {m.excluded && (
                  <span className="dupes-member-badge excluded">excluded</span>
                )}
              </div>
              <div className="dupes-member-info">
                <span className="dupes-member-name">{baseName(m.path)}</span>
                <span className="dupes-member-meta">
                  {pct(m.score)}
                  {m.width && m.height ? ` · ${m.width}×${m.height}` : ''}
                  {m.size ? ` · ${fmtBytes(m.size)}` : ''}
                </span>
              </div>
              {canWrite && (
                <button
                  type="button"
                  className="dupes-member-toggle"
                  disabled={busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    void onExclude(group, m.path, !m.excluded);
                  }}
                  title={
                    m.excluded
                      ? 'Put this item back into the group'
                      : 'This one is not a duplicate: drop it from the group (it will not be merged, and will not be regrouped here)'
                  }
                  aria-label={
                    m.excluded ? 'Include in group' : 'Exclude from group'
                  }
                >
                  {m.excluded ? '↩' : '×'}
                </button>
              )}
            </div>
          );
        })}
        {hidden > 0 && (
          <button
            type="button"
            className="dupes-member dupes-member-more"
            onClick={() => setExpanded(true)}
          >
            +{hidden} more
          </button>
        )}
      </div>
    </div>
  );
}

export default function DuplicatesGrid({
  isDisabled,
}: {
  isDisabled: boolean;
}) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const canWrite = useSelector(libraryService, (s) => s.context.canWrite);
  const initSessionId = useSelector(
    libraryService,
    (s) => s.context.initSessionId
  );
  const filteringMode = useSelector(
    libraryService,
    (s) => s.context.settings.filteringMode
  );
  const predicates = useSelector(
    libraryService,
    (s) => s.context.query?.predicates ?? []
  );
  // The item under the library cursor (same ordered view the list and the
  // detail panel use), so the active group can highlight what is on screen.
  const cursorPath = useSelector(libraryService, (s) => {
    const view = filter(
      s.context.libraryLoadId,
      s.context.textFilter,
      s.context.library,
      s.context.settings.filters,
      s.context.settings.sortBy
    );
    return (view?.[s.context.cursor]?.path as string | undefined) ?? null;
  });
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<ListStatus>('pending');
  const [sortBy, setSortBy] = useState<GroupSort>(() => {
    try {
      return window.localStorage.getItem('lowkey:dupes-sort') === 'members'
        ? 'members'
        : 'newest';
    } catch {
      return 'newest';
    }
  });
  const changeSort = (s: GroupSort) => {
    setSortBy(s);
    setPages(1);
    try {
      window.localStorage.setItem('lowkey:dupes-sort', s);
    } catch {
      /* session only */
    }
  };
  const [pages, setPages] = useState(1);
  const [busyGroup, setBusyGroup] = useState<number | null>(null);
  const [threshold, setThreshold] = useState<number>(() => {
    try {
      const v = Number(window.localStorage.getItem('lowkey:dupes-threshold'));
      return v >= 50 && v <= 100 ? v : 100;
    } catch {
      return 100;
    }
  });
  const [scanning, setScanning] = useState(false);
  const [resetArmed, fireReset] = useArmed();
  const [clearArmed, fireClear] = useArmed();
  const [mergeAllArmed, fireMergeAll] = useArmed();
  const [mergingAll, setMergingAll] = useState(false);

  const { data: stats } = useDuplicateStats();

  // Groups, one page per query so "Load more" appends without refetching
  // the pages already shown. Every mutation and every duplicates-updated
  // broadcast invalidates the whole prefix.
  const pageQueries = useMemo(
    () => Array.from({ length: pages }, (_, i) => i * PAGE_SIZE),
    [pages]
  );
  const [pageData, setPageData] = useState<Record<number, GroupPage>>({});
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloadTick, setReloadTick] = useState(0);
  const reload = useCallback(() => setReloadTick((t) => t + 1), []);

  useEffect(() => {
    if (!initSessionId) return undefined;
    let cancelled = false;
    setLoading(true);
    Promise.all(
      pageQueries.map((offset) =>
        apiGet<GroupPage>(
          authToken,
          `/api/duplicates?status=${status}&sort=${sortBy}&limit=${PAGE_SIZE}&offset=${offset}`
        ).then((p) => [offset, p] as const)
      )
    )
      .then((entries) => {
        if (cancelled) return;
        const next: Record<number, GroupPage> = {};
        for (const [offset, p] of entries) next[offset] = p;
        setPageData(next);
        setError(null);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [authToken, initSessionId, status, sortBy, pageQueries, reloadTick]);

  const groups = useMemo(() => {
    const seen = new Set<number>();
    const out: DuplicateGroup[] = [];
    for (const offset of pageQueries) {
      for (const g of pageData[offset]?.groups ?? []) {
        if (!seen.has(g.id)) {
          seen.add(g.id);
          out.push(g);
        }
      }
    }
    return out;
  }, [pageData, pageQueries]);
  const total = pageData[0]?.total ?? 0;

  // Live updates: the scan broadcasts as it writes groups (throttled
  // server-side), and every manual mutation from any window broadcasts too.
  // Job events tell us whether a scan is running so the toolbar can say so.
  useEffect(() => {
    let timer: number | null = null;
    return subscribeStream((type, event) => {
      if (type === 'duplicates-updated') {
        if (timer !== null) window.clearTimeout(timer);
        timer = window.setTimeout(() => {
          timer = null;
          queryClient.invalidateQueries({
            queryKey: ['taxonomy', 'duplicates'],
          });
          reload();
        }, 600);
        return;
      }
      if (type === 'create' || type === 'update' || type === 'delete') {
        try {
          const job = JSON.parse(event.data)?.job as
            | { command?: string; state?: string }
            | undefined;
          if (
            job?.command !== 'find-duplicates' &&
            job?.command !== 'merge-duplicates'
          ) {
            return;
          }
          const running =
            job.state === 'pending' || job.state === 'in_progress';
          if (job.command === 'merge-duplicates') setMergingAll(running);
          else setScanning(running);
          if (!running) {
            queryClient.invalidateQueries({
              queryKey: ['taxonomy', 'duplicates'],
            });
            reload();
          }
        } catch {
          /* malformed event */
        }
      }
    });
  }, [queryClient, reload]);

  // Which group (if any) the library is currently filtered to, so its card
  // reads as active like a selected tag.
  const activeGroupId = useMemo(() => {
    const p = predicates.find(
      (x) => x.type === 'dupe' && !x.exclude && /^\d+$/.test(x.value)
    );
    return p ? Number(p.value) : null;
  }, [predicates]);

  const toast = (
    type: 'success' | 'error' | 'info',
    title: string,
    message?: string
  ) =>
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type, title, message: message ?? '' },
    });

  const invalidateAfterMutation = () => {
    queryClient.invalidateQueries({ queryKey: ['taxonomy', 'duplicates'] });
    reload();
  };

  const runScan = async (input: string) => {
    try {
      await apiSend(authToken, 'POST', '/create', { input });
      // The job's own toast (ToastSystem, via the stream) announces the run.
    } catch (err) {
      toast('error', 'Failed to start duplicate scan', String(err));
    }
  };

  const handleScan = () =>
    void runScan(`find-duplicates --threshold=${threshold}`);
  const handleRebuild = () => {
    if (!fireReset()) return;
    void runScan(`find-duplicates --reset --threshold=${threshold}`);
  };
  // Bulk accept: the merge-duplicates task merges every pending group into
  // its anchor (a long-running job with progress; the panel follows its
  // duplicates-updated broadcasts and the job toast reports the totals).
  const handleMergeAll = () => {
    if (!fireMergeAll()) return;
    void runScan('merge-duplicates');
  };
  const handleClearAll = async () => {
    if (!fireClear()) return;
    try {
      await apiSend(
        authToken,
        'DELETE',
        `/api/duplicates/all?confirm=true&status=${status}`
      );
      invalidateAfterMutation();
    } catch (err) {
      toast('error', 'Could not clear groups', String(err));
    }
  };

  const handleView = (g: DuplicateGroup, pinPath?: string) => {
    // Already looking at this cluster: just move the cursor (no re-query).
    if (pinPath && activeGroupId === g.id) {
      const s = libraryService.getSnapshot();
      const view = filter(
        s.context.libraryLoadId,
        s.context.textFilter,
        s.context.library,
        s.context.settings.filters,
        s.context.settings.sortBy
      );
      const idx = (view ?? []).findIndex((it: { path: string }) =>
        samePath(it.path, pinPath)
      );
      if (idx >= 0) {
        libraryService.send('SET_CURSOR', { idx });
        return;
      }
    }
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'dupe',
          value: String(g.id),
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
        // Member click: land the cursor on that item (runningQuery prefers
        // pinnedPath when it places the cursor in the new result set).
        pinnedPath: pinPath,
      },
    });
  };

  const withBusy = async (g: DuplicateGroup, fn: () => Promise<void>) => {
    setBusyGroup(g.id);
    try {
      await fn();
    } finally {
      setBusyGroup(null);
    }
  };

  const handleMerge = (g: DuplicateGroup, keep: string) =>
    withBusy(g, async () => {
      try {
        const res = await apiSend<{
          merge: {
            target: string;
            tags: number;
            embeddings: number;
            transcript: boolean;
            deleted: string[];
            failed: string[];
          };
        }>(
          authToken,
          'POST',
          `/api/duplicates/${g.id}/merge`,
          { keep },
          MERGE_TIMEOUT_MS
        );
        const m = res.merge;
        if (m.deleted?.length) {
          // Deleted copies leave the in-memory library immediately (their DB
          // rows are already gone) — same as the palette's Merge.
          libraryService.send('REMOVE_MERGED_FILES', {
            data: { paths: m.deleted },
          });
        }
        queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
        queryClient.invalidateQueries({ queryKey: ['metadata'] });
        queryClient.invalidateQueries({ queryKey: ['embeddings'] });
        queryClient.invalidateQueries({ queryKey: ['transcript'] });
        invalidateAfterMutation();
        const failNote = m.failed?.length
          ? ` — ${m.failed.length} could not be deleted`
          : '';
        toast(
          m.failed?.length ? 'info' : 'success',
          `Merged ${m.deleted?.length ?? 0} into ${baseName(m.target)}`,
          `Kept ${m.tags} tag${m.tags === 1 ? '' : 's'}, ${
            m.embeddings
          } embedding${m.embeddings === 1 ? '' : 's'}${
            m.transcript ? ', transcript' : ''
          }${failNote}`
        );
      } catch (err) {
        toast('error', 'Merge failed', String(err));
      }
    });

  const handleStatus = (g: DuplicateGroup, next: ListStatus) =>
    withBusy(g, async () => {
      try {
        await apiSend(
          authToken,
          'POST',
          `/api/duplicates/${g.id}/${
            next === 'dismissed' ? 'dismiss' : 'restore'
          }`
        );
        invalidateAfterMutation();
      } catch (err) {
        toast('error', 'Could not update group', String(err));
      }
    });

  const handleExclude = (g: DuplicateGroup, path: string, excluded: boolean) =>
    withBusy(g, async () => {
      try {
        await apiSend(authToken, 'POST', `/api/duplicates/${g.id}/exclude`, {
          path,
          excluded,
        });
        invalidateAfterMutation();
      } catch (err) {
        toast('error', 'Could not update member', String(err));
      }
    });

  const handleForget = (g: DuplicateGroup) =>
    withBusy(g, async () => {
      try {
        await apiSend(authToken, 'DELETE', `/api/duplicates/${g.id}`);
        invalidateAfterMutation();
      } catch (err) {
        toast('error', 'Could not forget group', String(err));
      }
    });

  const emptyIcon = (
    <svg
      className="people-empty-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <rect x="3" y="3" width="12" height="12" rx="2" />
      <rect x="9" y="9" width="12" height="12" rx="2" />
    </svg>
  );

  if (error) {
    if (/HTTP 401/.test(error.message)) {
      return (
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title">Sign in to see Duplicates</div>
          <p className="people-empty-body">
            The media server is running but this session isn’t signed in.
            Duplicate groups are managed by the server, so log in and this panel
            will fill in.
          </p>
        </div>
      );
    }
    if (isElectron && !mediaServerConfigured) {
      return (
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title">Media server not installed</div>
          <p className="people-empty-body">
            Duplicate detection is provided by the Lowkey Media Server, a
            companion app that runs alongside the viewer. It doesn’t look like
            it has been installed on this machine yet.
          </p>
          <p className="people-empty-hint">
            Install and start the media server, then reopen this panel.
          </p>
        </div>
      );
    }
    return (
      <div className="people-empty">
        {emptyIcon}
        <div className="people-empty-title">Media server isn’t responding</div>
        <p className="people-empty-body">
          The media server is installed but unreachable right now. Start it (or
          check the connection), then try again.
        </p>
        <button type="button" className="people-empty-cta" onClick={reload}>
          Try again
        </button>
        <p className="people-empty-hint">{error.message}</p>
      </div>
    );
  }

  const toolbar = (
    <div className="people-grid-toolbar dupes-toolbar">
      <div className="dupes-tabs" role="tablist">
        {(['pending', 'dismissed'] as ListStatus[]).map((s) => (
          <button
            key={s}
            type="button"
            role="tab"
            aria-selected={status === s}
            className={`dupes-tab${status === s ? ' active' : ''}`}
            onClick={() => {
              setStatus(s);
              setPages(1);
            }}
          >
            {s === 'pending' ? 'To review' : 'Dismissed'}
            {stats && (
              <span className="people-btn-count">
                {(s === 'pending'
                  ? stats.pending
                  : stats.dismissed
                ).toLocaleString()}
              </span>
            )}
          </button>
        ))}
      </div>
      <div
        className="dupes-sort"
        role="group"
        aria-label="Sort groups"
        title="Order the groups: newest found first, or the clusters with the most items first"
      >
        {(['newest', 'members'] as GroupSort[]).map((s) => (
          <button
            key={s}
            type="button"
            className={`dupes-tab${sortBy === s ? ' active' : ''}`}
            aria-pressed={sortBy === s}
            onClick={() => changeSort(s)}
          >
            {s === 'newest' ? 'Newest' : 'Most items'}
          </button>
        ))}
      </div>
      {canWrite && (
        <>
          <label
            className="dupes-threshold"
            title="Similarity floor for the next scan. 100% finds visually identical items (exact copies, re-encodes, resizes); lower it to catch crops and edits."
          >
            ≥
            <input
              type="number"
              min={50}
              max={100}
              step={0.5}
              value={threshold}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isFinite(v)) return;
                const clamped = Math.max(50, Math.min(100, v));
                setThreshold(clamped);
                try {
                  window.localStorage.setItem(
                    'lowkey:dupes-threshold',
                    String(clamped)
                  );
                } catch {
                  /* session only */
                }
              }}
            />
            %
          </label>
          <button
            type="button"
            className="people-cluster-btn"
            onClick={handleScan}
            disabled={scanning}
            title="Scan every visual embedding for duplicate candidates. Incremental: items already in a group are left alone, new copies join their group, and groups appear here as they are found. Nothing is merged or deleted."
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="6" />
              <path d="M20 20l-4.5-4.5" />
            </svg>
            <span>{scanning ? 'Scanning…' : 'Find duplicates'}</span>
          </button>
          <button
            type="button"
            className={`people-cluster-btn${resetArmed ? ' danger' : ''}`}
            onClick={handleRebuild}
            disabled={scanning}
            title={
              resetArmed
                ? 'Click again to forget every pending group and rescan at the current threshold. Dismissed groups are kept.'
                : 'Rebuild: forget the pending groups and rescan from scratch (e.g. after changing the threshold). Dismissed groups are kept.'
            }
          >
            <span>{resetArmed ? 'Confirm rebuild' : 'Rebuild'}</span>
          </button>
          {status === 'pending' && total > 0 && (
            <button
              type="button"
              className={`people-cluster-btn${mergeAllArmed ? ' danger' : ''}`}
              onClick={handleMergeAll}
              disabled={scanning || mergingAll}
              title={
                mergeAllArmed
                  ? `Click again to start: every pending group is merged into its anchor and the other members are DELETED from disk (${
                      stats?.pendingItems && stats?.pending
                        ? `${(
                            stats.pendingItems - stats.pending
                          ).toLocaleString()} files`
                        : 'many files'
                    }). Runs as a job with progress; excluded members and dismissed groups are left alone.`
                  : 'Accept every pending group: merge each into its anchor and delete the other copies. Runs as a background job. Asks to confirm.'
              }
            >
              <span>
                {mergingAll
                  ? 'Merging all…'
                  : mergeAllArmed
                  ? 'Confirm — merge all'
                  : 'Merge all'}
              </span>
            </button>
          )}
          {total > 0 && (
            <button
              type="button"
              className={`people-cluster-btn${clearArmed ? ' danger' : ''}`}
              onClick={() => void handleClearAll()}
              title={
                clearArmed
                  ? `Click again to forget all ${status} groups (files are untouched)`
                  : `Forget every ${status} group. Files are untouched; a future scan may regroup them.`
              }
            >
              <span>{clearArmed ? 'Confirm clear' : 'Clear all'}</span>
            </button>
          )}
        </>
      )}
    </div>
  );

  if (loading && groups.length === 0) {
    return (
      <div className="people-grid-wrap">
        {toolbar}
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title people-empty-loading">
            Loading duplicate groups…
          </div>
        </div>
      </div>
    );
  }

  if (groups.length === 0) {
    return (
      <div className="people-grid-wrap">
        {toolbar}
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title">
            {status === 'pending'
              ? 'No duplicate candidates to review'
              : 'No dismissed groups'}
          </div>
          <p className="people-empty-body">
            {status === 'pending'
              ? scanning
                ? 'A scan is running — groups will appear here as they are found.'
                : 'Run a scan to group visually identical items from your visual embeddings. Groups show up here as they are found; nothing is merged or deleted until you say so.'
              : 'Groups you mark “Not duplicates” land here, and their items stay out of future scans.'}
          </p>
          {canWrite && status === 'pending' && !scanning && (
            <button
              type="button"
              className="people-empty-cta"
              onClick={handleScan}
              disabled={isDisabled}
            >
              Find duplicates
            </button>
          )}
          {status === 'pending' && (
            <p className="people-empty-hint">
              Items need a visual embedding first (the Embeddings task).
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="people-grid-wrap dupes-wrap">
      {toolbar}
      <div className="dupes-list">
        {groups.map((g) => (
          <GroupCard
            key={g.id}
            group={g}
            canWrite={canWrite}
            isDisabled={isDisabled}
            isActive={activeGroupId === g.id}
            busy={busyGroup === g.id}
            cursorPath={cursorPath}
            onView={handleView}
            onMerge={handleMerge}
            onStatus={handleStatus}
            onExclude={handleExclude}
            onForget={handleForget}
          />
        ))}
        {groups.length < total && (
          <button
            type="button"
            className="dupes-load-more"
            onClick={() => setPages((p) => p + 1)}
            disabled={loading}
          >
            {loading
              ? 'Loading…'
              : `Load more (${(
                  total - groups.length
                ).toLocaleString()} more group${
                  total - groups.length === 1 ? '' : 's'
                })`}
          </button>
        )}
      </div>
    </div>
  );
}
