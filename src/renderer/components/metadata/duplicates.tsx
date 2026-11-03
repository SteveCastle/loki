// Duplicates tab: does the CURRENT item have copies? It reads the same
// vector-similarity duplicate groups as the taxonomy panel's Duplicates
// category (find-duplicates, media-server/tasks/find_duplicates.go) and
// renders the item's group with the panel's own GroupCard, so merge,
// "not duplicates", and per-member exclusion work exactly as they do there.
//
// An item no scan has grouped yet still gets an answer: GET
// /api/duplicates/check runs one nearest-neighbour query for it and returns
// the group a scan would record (a preview, id 0). Nothing is written until
// the user acts on it; the first action records the group (POST
// /api/duplicates/check/record) and then applies to the recorded group.
//
// Cost is one small request per item (debounced while the cursor moves) and
// at most MEMBER_PREVIEW_CAP thumbnails, however big the group is.
import { useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { isElectron, mediaServerConfigured } from '../../platform';
import { subscribeStream } from '../../stream-bus';
import {
  apiGet,
  apiSend,
  DuplicateGroup,
  GroupCard,
  ListStatus,
  MergeResult,
  MERGE_TIMEOUT_MS,
  reportMerge,
} from '../taxonomy/duplicates-grid';
import '../taxonomy/people-grid.css';
import '../taxonomy/duplicates-grid.css';
import './duplicates.css';

type Props = {
  basePath: string;
};

interface CheckResult {
  group: DuplicateGroup | null;
  recorded: boolean;
  noEmbedding: boolean;
}

// Cursor moves faster than anyone reads this tab; only check where it rests.
const CHECK_DEBOUNCE_MS = 250;
// Shared with the taxonomy panel's scan threshold, so both agree on what
// counts as a duplicate.
const THRESHOLD_KEY = 'lowkey:dupes-threshold';

function readThreshold(): number {
  try {
    const v = Number(window.localStorage.getItem(THRESHOLD_KEY));
    return v >= 50 && v <= 100 ? v : 100;
  } catch {
    return 100;
  }
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

export default function Duplicates({ basePath }: Props) {
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
  const queryClient = useQueryClient();
  const [threshold, setThreshold] = useState<number>(readThreshold);
  const [busy, setBusy] = useState(false);
  const [embedQueued, setEmbedQueued] = useState<string | null>(null);

  const path = useDebounced(basePath, CHECK_DEBOUNCE_MS);
  const settled = path === basePath;

  const checkKey = useMemo(
    () => ['taxonomy', 'duplicates', 'check', initSessionId, path, threshold],
    [initSessionId, path, threshold]
  );
  const { data, error, isLoading, refetch } = useQuery<CheckResult, Error>(
    checkKey,
    () =>
      apiGet<CheckResult>(
        authToken,
        `/api/duplicates/check?path=${encodeURIComponent(
          path
        )}&threshold=${threshold}`
      ),
    { enabled: !!initSessionId && !!path, staleTime: 30_000, retry: 1 }
  );

  // Groups change under us (a scan, the panel, another window), and an embed
  // job finishing makes an unembedded item checkable.
  useEffect(() => {
    let timer: number | null = null;
    const refresh = () => {
      if (timer !== null) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        timer = null;
        queryClient.invalidateQueries({
          queryKey: ['taxonomy', 'duplicates', 'check'],
        });
      }, 600);
    };
    const unsubscribe = subscribeStream((type, event) => {
      if (type === 'duplicates-updated') {
        refresh();
        return;
      }
      if (type === 'update') {
        try {
          const job = JSON.parse(event.data)?.job as
            | { command?: string; state?: string }
            | undefined;
          if (job?.command === 'embed' && job.state === 'completed') {
            setEmbedQueued(null);
            refresh();
          }
        } catch {
          /* malformed event */
        }
      }
    });
    return () => {
      if (timer !== null) window.clearTimeout(timer);
      unsubscribe();
    };
  }, [queryClient]);

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

  const changed = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['taxonomy', 'duplicates'] });
  }, [queryClient]);

  // Every action needs a recorded group; a preview is recorded first.
  const recordedId = async (g: DuplicateGroup): Promise<number> => {
    if (g.id !== 0) return g.id;
    const res = await apiSend<{ id: number }>(
      authToken,
      'POST',
      '/api/duplicates/check/record',
      { path, threshold }
    );
    return res.id;
  };

  const withBusy = async (label: string, fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (err) {
      toast('error', label, String(err));
      changed();
    } finally {
      setBusy(false);
    }
  };

  const handleView = (g: DuplicateGroup, pinPath?: string) => {
    // A preview has no group to filter to yet; the click only picks the
    // keeper (GroupCard does that itself).
    if (g.id === 0) return;
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'dupe',
          value: String(g.id),
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
        pinnedPath: pinPath ?? basePath,
      },
    });
  };

  const handleMerge = (g: DuplicateGroup, keep: string) =>
    withBusy('Merge failed', async () => {
      const id = await recordedId(g);
      // Merge exactly what the card showed: recording a join can add
      // members the user has not looked at.
      const sources = g.members
        .filter((m) => !m.excluded && m.path !== keep)
        .map((m) => m.path);
      const res = await apiSend<{ merge: MergeResult }>(
        authToken,
        'POST',
        `/api/duplicates/${id}/merge`,
        { keep, sources },
        MERGE_TIMEOUT_MS
      );
      reportMerge(libraryService, queryClient, res.merge);
    });

  const handleStatus = (g: DuplicateGroup, next: ListStatus) =>
    withBusy('Could not update group', async () => {
      const id = await recordedId(g);
      await apiSend(
        authToken,
        'POST',
        `/api/duplicates/${id}/${next === 'dismissed' ? 'dismiss' : 'restore'}`
      );
      changed();
    });

  const handleExclude = (g: DuplicateGroup, p: string, excluded: boolean) =>
    withBusy('Could not update member', async () => {
      const id = await recordedId(g);
      await apiSend(authToken, 'POST', `/api/duplicates/${id}/exclude`, {
        path: p,
        excluded,
      });
      changed();
    });

  const handleForget = (g: DuplicateGroup) =>
    withBusy('Could not forget group', async () => {
      await apiSend(authToken, 'DELETE', `/api/duplicates/${g.id}`);
      changed();
    });

  const handleEmbed = async () => {
    try {
      await apiSend(authToken, 'POST', '/create', {
        input: `embed "${basePath}"`,
      });
      setEmbedQueued(basePath);
    } catch (err) {
      toast('error', 'Could not start the embedding job', String(err));
    }
  };

  const thresholdControl = (
    <label
      className="dupes-threshold"
      title="Similarity floor. 100% finds visually identical items (exact copies, re-encodes, resizes); lower it to catch crops and edits. Shared with the Duplicates panel's scans."
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
            window.localStorage.setItem(THRESHOLD_KEY, String(clamped));
          } catch {
            /* session only */
          }
        }}
      />
      %
    </label>
  );

  const group = settled ? data?.group ?? null : null;
  let status: string;
  if (!settled || isLoading) status = 'Checking for duplicates…';
  else if (error) status = 'Duplicate check unavailable';
  else if (data?.noEmbedding) status = 'No visual embedding yet';
  else if (!group) status = 'No duplicates found';
  else if (data?.recorded)
    status =
      group.status === 'dismissed'
        ? 'Marked not duplicates'
        : `In duplicate group #${group.id}`;
  else
    status = `${group.activeCount - 1} possible duplicate${
      group.activeCount === 2 ? '' : 's'
    }`;

  let body: JSX.Element | null = null;
  if (error && settled) {
    const unauthorized = /HTTP 401/.test(error.message);
    body = (
      <div className="people-empty">
        <div className="people-empty-title">
          {unauthorized
            ? 'Sign in to check for duplicates'
            : isElectron && !mediaServerConfigured
            ? 'Media server not installed'
            : 'Media server isn’t responding'}
        </div>
        <p className="people-empty-body">
          Duplicate detection compares visual embeddings on the Lowkey Media
          Server.
        </p>
        {!unauthorized && (
          <button
            type="button"
            className="people-empty-cta"
            onClick={() => void refetch()}
          >
            Try again
          </button>
        )}
      </div>
    );
  } else if (settled && data?.noEmbedding) {
    body = (
      <div className="people-empty">
        <p className="people-empty-body">
          Duplicates are found by comparing visual embeddings, and this item
          doesn’t have one yet.
        </p>
        {canWrite && (
          <button
            type="button"
            className="people-empty-cta"
            onClick={() => void handleEmbed()}
            disabled={embedQueued === basePath}
          >
            {embedQueued === basePath ? 'Embedding…' : 'Embed this item'}
          </button>
        )}
      </div>
    );
  } else if (group) {
    body = (
      <GroupCard
        key={`${group.id}:${group.anchorPath}`}
        group={group}
        canWrite={canWrite}
        isDisabled={false}
        isActive={group.id !== 0 && activeGroupId === group.id}
        busy={busy}
        cursorPath={basePath}
        currentPath={basePath}
        onView={handleView}
        onMerge={handleMerge}
        onStatus={handleStatus}
        onExclude={handleExclude}
        onForget={data?.recorded ? handleForget : undefined}
      />
    );
  } else if (settled && !isLoading) {
    body = (
      <p className="dupes-tab-hint">
        Nothing in the library is at least {threshold}% similar to this item.
        {threshold === 100 && ' Lower the threshold to catch crops and edits.'}
      </p>
    );
  }

  return (
    <div className="dupes-tab">
      <div className="dupes-tab-toolbar">
        <span className="dupes-tab-status">{status}</span>
        {thresholdControl}
      </div>
      {body}
      {group && !data?.recorded && (
        <p className="dupes-tab-hint">
          Found just now — merging, dismissing or excluding records this group
          in the Duplicates panel.
        </p>
      )}
    </div>
  );
}
