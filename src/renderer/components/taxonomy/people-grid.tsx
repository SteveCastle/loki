// People: the face-identity category, rendered inside the taxonomy panel as a
// special case of a tag category. Person names ARE tags (in the "People"
// category — the server keeps them in sync), so clicking a person filters the
// library exactly like clicking a tag. What differs is management: renames,
// merges, and deletes must go through /api/people so the person table and its
// taxonomy tag stay in sync — renaming through the normal tag editor would
// desync them. Cards show a face crop instead of a media preview.
import { useContext, useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import { useDrag, useDrop } from 'react-dnd';
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import useHideNativeDragPreview from '../../hooks/useHideNativeDragPreview';
import { GlobalStateContext } from '../../state';
import {
  isElectron,
  mediaServerBase,
  mediaServerConfigured,
} from '../../platform';
import { subscribeStream } from '../../stream-bus';
import { displayTagLabel } from '../../tag-display';
import './new-modal.css';
import './people-grid.css';

export const PEOPLE_CATEGORY = 'People';

export interface Person {
  id: number;
  name: string;
  coverFaceId?: number;
  faceCount: number;
  mediaCount: number;
  // How many faces are human-confirmed (user-assigned). Locked faces survive
  // every regroup and carry extra weight when clustering.
  lockedCount: number;
  // Recognizer(s) the cluster's faces were embedded with (comma-separated;
  // normally one) — distinguishes photo-face clusters from anime-character
  // clusters.
  models?: string;
}

// One face of a person in the review drawer. Typicality is the face's cosine
// against the group's (user-weighted) center — the server returns the list
// least-typical first, so likely mistakes surface at the top.
interface ReviewFace {
  id: number;
  path: string;
  frameTs: number;
  detScore: number;
  assignedBy: string;
  typicality: number;
}

// describeModels renders the recognizer id(s) as a human label so the admin
// can tell a photographic face cluster from a drawn-character one.
function describeModels(models?: string): string | null {
  if (!models) return null;
  const label = (id: string): string => {
    if (id === 'anime-ccip') return 'Anime character (CCIP)';
    if (id === 'sface') return 'Photo face (SFace)';
    if (id.includes('anime') || id.includes('ccip')) {
      return `Anime character (${id})`;
    }
    return `Photo face (${id})`;
  };
  return models.split(',').map(label).join(', ');
}

function authHeaders(authToken: string | null): HeadersInit {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

async function fetchPeople(authToken: string | null): Promise<Person[]> {
  const res = await fetch(`${mediaServerBase}/api/people`, {
    headers: authHeaders(authToken),
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as Person[];
}

// usePeople is the shared people list. Every consumer (the People grid, the
// taxonomy search results, media tag chips) reads the SAME cache entry, keyed
// under the 'taxonomy' prefix so existing broad invalidations (tag mutations,
// people-updated broadcasts, DB swaps) refresh them all together. retry: 1 so
// a dead server resolves to an error quickly instead of sitting in
// default-backoff limbo.
export function usePeople(enabled = true) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const initSessionId = useSelector(
    libraryService,
    (s) => s.context.initSessionId
  );
  return useQuery<Person[], Error>(
    ['taxonomy', 'people', initSessionId],
    () => fetchPeople(authToken),
    { enabled: enabled && !!initSessionId, staleTime: 60_000, retry: 1 }
  );
}

// useMergePeople returns the drag-merge handler (drop one person card onto
// another): every face and taxonomy row of `from` moves to `into`, and `from`
// is removed. Shared by the People grid and the taxonomy search results.
function useMergePeople(): (from: Person, into: Person) => Promise<void> {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const queryClient = useQueryClient();
  return async (from: Person, into: Person) => {
    try {
      const res = await fetch(
        `${mediaServerBase}/api/people/${from.id}/merge`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...authHeaders(authToken),
          },
          credentials: 'include',
          body: JSON.stringify({ intoId: into.id }),
        }
      );
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'People merged',
          message: `“${displayTagLabel(from.name)}” merged into “${displayTagLabel(into.name)}”`,
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
    } catch (err) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to merge people',
          message: String(err),
        },
      });
    }
  };
}

// How many detected faces have no person yet — the work "Group new faces"
// would pick up. Shown as a badge on that button.
async function fetchUngroupedCount(
  authToken: string | null
): Promise<number> {
  const res = await fetch(`${mediaServerBase}/api/faces/ungrouped`, {
    headers: authHeaders(authToken),
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { count: number };
  return data.count ?? 0;
}

// Grouping tuner (Tune sliders), persisted in the server config so it applies
// to every clustering pass — the toolbar buttons, tuned regroups, and the
// automatic in-scan passes — on every client.
type ClusterTuning = { offset: number; minCluster: number; minQuality: number };

async function fetchTuning(authToken: string | null): Promise<ClusterTuning> {
  const res = await fetch(`${mediaServerBase}/api/faces/tuning`, {
    headers: authHeaders(authToken),
    credentials: 'include',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as {
    thresholdOffset: number;
    minCluster: number;
    minQuality: number;
  };
  return {
    offset: data.thresholdOffset ?? 0,
    minCluster: data.minCluster ?? 3,
    minQuality: data.minQuality ?? 0.75,
  };
}

async function saveTuning(
  authToken: string | null,
  t: ClusterTuning
): Promise<void> {
  const res = await fetch(`${mediaServerBase}/api/faces/tuning`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(authToken),
    },
    credentials: 'include',
    body: JSON.stringify({
      thresholdOffset: t.offset,
      minCluster: t.minCluster,
      minQuality: t.minQuality,
    }),
  });
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
}

// One not-yet-grouped face from GET /api/faces/ungrouped?faces=1.
type UngroupedFace = {
  id: number;
  path: string;
  frameTs: number;
  detScore: number;
  model: string;
};

async function fetchUngroupedFaces(
  authToken: string | null,
  offset: number
): Promise<{ count: number; faces: UngroupedFace[] }> {
  const res = await fetch(
    `${mediaServerBase}/api/faces/ungrouped?faces=1&limit=120&offset=${offset}`,
    { headers: authHeaders(authToken), credentials: 'include' }
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as { count: number; faces?: UngroupedFace[] };
  return { count: data.count ?? 0, faces: data.faces ?? [] };
}

// A face crop loaded with auth (an <img src> can't carry a bearer token in
// Electron), delivered as a blob URL. Falls back to a silhouette.
// Exported for the custom drag layer's person chip (controls/drag-layer.tsx).
export function FaceCrop({
  faceId,
  authToken,
  size = 160,
}: {
  faceId?: number;
  authToken: string | null;
  size?: number;
}) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!faceId) return undefined;
    let objectUrl: string | null = null;
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(
          `${mediaServerBase}/media/facecrop?id=${faceId}&size=${size}`,
          {
            headers: authHeaders(authToken),
            credentials: 'include',
            signal: controller.signal,
          }
        );
        if (!res.ok) return;
        objectUrl = URL.createObjectURL(await res.blob());
        setUrl(objectUrl);
      } catch {
        // silhouette fallback
      }
    })();
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [faceId, authToken, size]);

  if (url) return <img className="person-card-face" src={url} alt="" />;
  return (
    <div className="person-card-face person-card-face--empty" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />
      </svg>
    </div>
  );
}

// Create/edit modal for a person. person == null → create. Uses the shared
// input-modal styles so it matches the tag/category modals.
export function PersonEditModal({
  person,
  people,
  handleClose,
}: {
  person: Person | null;
  people: Person[];
  handleClose: () => void;
}) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const queryClient = useQueryClient();
  // Edit the DISPLAY name; the server re-applies the _cluster suffix if the
  // plain name is owned by a curated tag.
  const [name, setName] = useState(person ? displayTagLabel(person.name) : '');
  const [mergeInto, setMergeInto] = useState<number>(0);
  const ref = useRef(null);
  useOnClickOutside(ref, handleClose);

  const toastError = (title: string, err: unknown) =>
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type: 'error', title, message: String(err) },
    });
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
    queryClient.invalidateQueries({ queryKey: ['metadata'] });
    // Person names double as People-category tags on media items, so any
    // person mutation must also refresh the per-media tag lists.
    queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
  };

  const call = async (path: string, init: RequestInit) => {
    const res = await fetch(`${mediaServerBase}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(authToken),
        ...(init.headers || {}),
      },
      credentials: 'include',
    });
    if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
    return res;
  };

  const handleSave = async () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      if (!person) {
        await call('/api/people', {
          method: 'POST',
          body: JSON.stringify({ name: trimmed }),
        });
      } else if (trimmed !== displayTagLabel(person.name)) {
        const res = await call(`/api/people/${person.id}/rename`, {
          method: 'POST',
          body: JSON.stringify({ name: trimmed }),
        });
        // The person's name doubles as a library-filter tag. If the current
        // view filters on the OLD name, swap in the resolved new one (the
        // server may suffix it) so the view keeps showing the same set
        // instead of going blank on a now-nonexistent tag.
        const out = (await res.json()) as { name?: string };
        if (out.name) {
          libraryService.send({
            type: 'RENAME_TAG_PREDICATE',
            data: { from: person.name, to: out.name },
          });
        }
      }
      refresh();
      handleClose();
    } catch (err) {
      toastError(person ? 'Failed to rename person' : 'Failed to create person', err);
    }
  };

  const handleMerge = async () => {
    if (!person || !mergeInto) return;
    try {
      await call(`/api/people/${person.id}/merge`, {
        method: 'POST',
        body: JSON.stringify({ intoId: mergeInto }),
      });
      refresh();
      handleClose();
    } catch (err) {
      toastError('Failed to merge people', err);
    }
  };

  const handleDelete = async () => {
    if (!person) return;
    try {
      await call(`/api/people/${person.id}`, { method: 'DELETE' });
      refresh();
      handleClose();
    } catch (err) {
      toastError('Failed to delete person', err);
    }
  };

  const handleDeleteWithFaces = async () => {
    if (!person) return;
    try {
      await call(`/api/people/${person.id}?deleteFaces=true`, {
        method: 'DELETE',
      });
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'Group purged',
          message: `“${displayTagLabel(person.name)}” and its ${person.faceCount} face${
            person.faceCount === 1 ? '' : 's'
          } were deleted`,
        },
      });
      refresh();
      handleClose();
    } catch (err) {
      toastError('Failed to purge group', err);
    }
  };

  const handleRegenerateCover = async () => {
    if (!person) return;
    try {
      await call(`/api/people/${person.id}/cover`, { method: 'POST' });
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'Preview regenerated',
          message: `Picked a new face for “${displayTagLabel(person.name)}”`,
        },
      });
      refresh();
      handleClose();
    } catch (err) {
      toastError('Failed to regenerate preview', err);
    }
  };

  const mergeTargets = person
    ? people.filter((p) => p.id !== person.id)
    : [];

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {person ? 'Edit Person' : 'New Person'}
          </div>
          <div className="input-modal-close" onClick={handleClose}>
            <img src={cancel} />
          </div>
        </div>

        <div className="input-modal-properties">
          <label>Name</label>
          <input
            autoFocus
            type="text"
            value={name}
            onChange={(e) => {
              e.stopPropagation();
              setName(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') handleSave();
            }}
          />
          {person && describeModels(person.models) && (
            <div className="person-model-info">
              Embeddings: {describeModels(person.models)} ·{' '}
              {person.faceCount} face{person.faceCount === 1 ? '' : 's'}
            </div>
          )}
        </div>

        {person && (
          <>
            <div className="input-modal-divider" />
            <div className="input-modal-actions">
              <div className="input-modal-actions-label">Actions</div>
              {mergeTargets.length > 0 && (
                <div className="action-row">
                  <div className="action-row-text">
                    <div className="action-row-title">Merge into</div>
                    <div className="action-row-description">
                      Move every face and tag of “{displayTagLabel(person.name)}” onto another
                      person, then remove this one
                    </div>
                    <select
                      className="person-merge-select"
                      value={mergeInto}
                      onChange={(e) => setMergeInto(Number(e.target.value))}
                    >
                      <option value={0}>Choose a person…</option>
                      {mergeTargets.map((p) => (
                        <option key={p.id} value={p.id}>
                          {displayTagLabel(p.name)} ({p.faceCount})
                        </option>
                      ))}
                    </select>
                  </div>
                  <button onClick={handleMerge} disabled={!mergeInto}>
                    Merge
                  </button>
                </div>
              )}
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">Regenerate preview</div>
                  <div className="action-row-description">
                    Picks the clearest face in this group that actually
                    renders and uses it as the card image
                  </div>
                </div>
                <button onClick={handleRegenerateCover}>Regenerate</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">Delete person</div>
                  <div className="action-row-description">
                    Unassigns their faces and removes the tag. The faces are
                    kept and can join other groups, but this exact group is
                    remembered and won’t re-form on its own
                  </div>
                </div>
                <button onClick={handleDelete}>Delete</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Delete person and faces
                  </div>
                  <div className="action-row-description">
                    Also deletes every face embedding in this group, so a
                    messy cluster won&apos;t re-form on the next grouping.
                    The media itself is untouched; faces only come back if
                    you rescan it.
                  </div>
                </div>
                <button onClick={handleDeleteWithFaces}>Purge</button>
              </div>
            </div>
          </>
        )}

        <div className="input-modal-divider" />
        <div className="input-modal-footer">
          <button className="btn-cancel" onClick={handleClose}>
            Cancel
          </button>
          <button className="btn-save" onClick={handleSave}>
            {person ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Face review drawer: every face in the group, least-typical first, so the
// human can slice the cluster by hand. ✓ locks a face in (a user assignment —
// survives every regroup and weighs extra when clustering); ✗ rejects it (the
// face is removed AND permanently barred from this group, even if the group is
// dissolved and re-forms). "Lock all" endorses the whole group at once.
export function FaceReviewModal({
  person,
  handleClose,
}: {
  person: Person;
  handleClose: () => void;
}) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const queryClient = useQueryClient();
  const [faces, setFaces] = useState<ReviewFace[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // 'suspects' = server order (least typical first); 'best' = reversed.
  const [sortMode, setSortMode] = useState<'suspects' | 'best'>('suspects');
  // Checked faces are the KEEPERS for the bulk curate action ("keep checked,
  // discard the rest"). Seeded with the already-confirmed faces on load.
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const [curateArmed, setCurateArmed] = useState(false);
  const ref = useRef(null);
  useOnClickOutside(ref, handleClose);

  useEffect(() => {
    if (!curateArmed) return undefined;
    const t = window.setTimeout(() => setCurateArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [curateArmed]);

  const toast = (
    type: 'success' | 'error',
    title: string,
    message: string
  ) =>
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type, title, message },
    });

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
    queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
  };

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(
          `${mediaServerBase}/api/people/${person.id}/faces`,
          {
            headers: authHeaders(authToken),
            credentials: 'include',
            signal: controller.signal,
          }
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as { faces: ReviewFace[] };
        setFaces(data.faces ?? []);
        setChecked(
          new Set(
            (data.faces ?? [])
              .filter((f) => f.assignedBy === 'user')
              .map((f) => f.id)
          )
        );
      } catch (err) {
        if (!controller.signal.aborted) setLoadError(String(err));
      }
    })();
    return () => controller.abort();
  }, [person.id, authToken]);

  const post = async (path: string, body: unknown) => {
    const res = await fetch(`${mediaServerBase}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(authToken),
      },
      credentials: 'include',
      body: JSON.stringify(body ?? {}),
    });
    if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
    return res;
  };

  const toggleChecked = (id: number) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleConfirm = async (face: ReviewFace) => {
    try {
      await post(`/api/faces/${face.id}/assign`, { personId: person.id });
      setFaces(
        (prev) =>
          prev &&
          prev.map((f) =>
            f.id === face.id ? { ...f, assignedBy: 'user' } : f
          )
      );
      setChecked((prev) => new Set(prev).add(face.id));
      refresh();
    } catch (err) {
      toast('error', 'Failed to confirm face', String(err));
    }
  };

  const handleReject = async (face: ReviewFace) => {
    try {
      await post(`/api/faces/${face.id}/reject`, { personId: person.id });
      setFaces((prev) => prev && prev.filter((f) => f.id !== face.id));
      setChecked((prev) => {
        const next = new Set(prev);
        next.delete(face.id);
        return next;
      });
      refresh();
    } catch (err) {
      toast('error', 'Failed to remove face', String(err));
    }
  };

  // Keep checked, discard the rest: locks every checked face and permanently
  // rejects everything else in one server call — the fast false-positive
  // cleanup. Two-click confirm, since the discards can't rejoin this group.
  const handleCurate = async () => {
    if (!faces) return;
    if (!curateArmed) {
      setCurateArmed(true);
      return;
    }
    setCurateArmed(false);
    const keepIds = faces.filter((f) => checked.has(f.id)).map((f) => f.id);
    const discards = faces.length - keepIds.length;
    try {
      await post(`/api/people/${person.id}/curate`, { keepFaceIds: keepIds });
      setFaces(
        (prev) =>
          prev &&
          prev
            .filter((f) => checked.has(f.id))
            .map((f) => ({ ...f, assignedBy: 'user' }))
      );
      toast(
        'success',
        'Group cleaned up',
        `Kept ${keepIds.length} confirmed face${
          keepIds.length === 1 ? '' : 's'
        }, discarded ${discards} — they can never return to “${displayTagLabel(
          person.name
        )}”`
      );
      refresh();
    } catch (err) {
      toast('error', 'Failed to clean up group', String(err));
    }
  };

  const handleLockAll = async () => {
    try {
      await post(`/api/people/${person.id}/lock`, {});
      setFaces(
        (prev) => prev && prev.map((f) => ({ ...f, assignedBy: 'user' }))
      );
      toast(
        'success',
        'Group locked',
        `Every face of “${displayTagLabel(person.name)}” is now confirmed`
      );
      refresh();
    } catch (err) {
      toast('error', 'Failed to lock group', String(err));
    }
  };

  const shown =
    faces && sortMode === 'best' ? [...faces].reverse() : faces ?? [];
  const lockedCount = faces
    ? faces.filter((f) => f.assignedBy === 'user').length
    : 0;
  const allLocked = !!faces && faces.length > 0 && lockedCount === faces.length;

  return (
    <div className="input-modal">
      <div className="input-modal-content face-review-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            Review “{displayTagLabel(person.name)}”
            {faces && (
              <span className="face-review-counts">
                {faces.length} face{faces.length === 1 ? '' : 's'} ·{' '}
                {lockedCount} confirmed
              </span>
            )}
          </div>
          <div className="input-modal-close" onClick={handleClose}>
            <img src={cancel} />
          </div>
        </div>

        <div className="face-review-toolbar">
          <div className="face-review-sort">
            <button
              type="button"
              className={sortMode === 'suspects' ? 'active' : ''}
              onClick={() => setSortMode('suspects')}
              title="Faces least similar to the group first — likely mistakes on top"
            >
              Suspects first
            </button>
            <button
              type="button"
              className={sortMode === 'best' ? 'active' : ''}
              onClick={() => setSortMode('best')}
              title="Most typical faces first"
            >
              Best first
            </button>
          </div>
          <div className="face-review-check-tools">
            <button
              type="button"
              onClick={() =>
                setChecked(new Set((faces ?? []).map((f) => f.id)))
              }
              disabled={!faces || faces.length === 0}
            >
              Check all
            </button>
            <button
              type="button"
              onClick={() => setChecked(new Set())}
              disabled={checked.size === 0}
            >
              None
            </button>
          </div>
          <button
            type="button"
            className="face-review-lock-all"
            onClick={handleLockAll}
            disabled={!faces || faces.length === 0 || allLocked}
            title="Confirm every face in this group. Confirmed faces survive every regroup and carry extra weight when grouping."
          >
            {allLocked ? 'All confirmed' : 'Lock all'}
          </button>
        </div>

        <div className="face-review-grid">
          {!faces && !loadError && (
            <div className="face-review-status">Loading faces…</div>
          )}
          {loadError && (
            <div className="face-review-status">
              Couldn’t load faces: {loadError}
            </div>
          )}
          {faces && faces.length === 0 && (
            <div className="face-review-status">
              This group has no faces right now.
            </div>
          )}
          {shown.map((face) => {
            const locked = face.assignedBy === 'user';
            const isChecked = checked.has(face.id);
            // Cosine typicality mapped onto a 0–100% bar (clamped; scale is
            // model-dependent so this is a relative cue, not a probability).
            const barPct = Math.max(
              4,
              Math.min(100, Math.round(face.typicality * 100))
            );
            return (
              <div
                key={face.id}
                className={`face-review-tile${locked ? ' locked' : ''}${
                  isChecked ? ' checked' : ''
                }`}
                onClick={() => toggleChecked(face.id)}
                title={`Similarity to group: ${face.typicality.toFixed(2)}${
                  locked ? ' · confirmed by you' : ''
                } · click to ${isChecked ? 'uncheck' : 'check as a keeper'}`}
              >
                <FaceCrop faceId={face.id} authToken={authToken} size={120} />
                <span
                  className="face-review-check"
                  role="checkbox"
                  aria-checked={isChecked}
                  aria-label={isChecked ? 'Keeper' : 'Not checked'}
                >
                  {isChecked ? '✓' : ''}
                </span>
                {locked && (
                  <span className="face-review-lock-badge" aria-hidden="true">
                    ✓
                  </span>
                )}
                <span
                  className="face-review-typicality"
                  style={{ width: `${barPct}%` }}
                  aria-hidden="true"
                />
                <div className="face-review-tile-actions">
                  {!locked && (
                    <button
                      type="button"
                      className="confirm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleConfirm(face);
                      }}
                      title="It's them — lock this face in. Survives every regroup and weighs extra when grouping."
                      aria-label="Confirm face"
                    >
                      ✓
                    </button>
                  )}
                  <button
                    type="button"
                    className="reject"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReject(face);
                    }}
                    title="Not them — remove this face and never let grouping put it back here."
                    aria-label="Remove face"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {faces && faces.length > 0 && (
          <div className="face-review-curate">
            <span className="face-review-curate-counts">
              {checked.size} keeper{checked.size === 1 ? '' : 's'} checked ·{' '}
              {faces.length - checked.size} to discard
            </span>
            <button
              type="button"
              className={`face-review-curate-btn${
                curateArmed ? ' danger' : ''
              }`}
              onClick={handleCurate}
              disabled={checked.size === faces.length}
              title={
                curateArmed
                  ? 'Click again to confirm. Discarded faces are permanently barred from this group.'
                  : 'Lock every checked face and discard the rest. Discards can never be regrouped back into this person.'
              }
            >
              {curateArmed
                ? `Confirm: discard ${faces.length - checked.size}`
                : 'Keep checked, discard rest'}
            </button>
          </div>
        )}

        <p className="face-review-hint">
          Click faces to check the ones that belong, then “Keep checked,
          discard rest” to clean the group in one go. ✓ confirms a single
          face; ✕ removes one. Confirmed faces survive every regroup and pull
          similar faces here; discarded faces can never return to this group —
          even after regrouping with new settings.
        </p>
      </div>
    </div>
  );
}

// Manual triage for the faces clustering never placed — the "Ungrouped"
// pseudo-group. Best detections sort first (those plausibly SHOULD have
// grouped; the blurry tail comes last). Pick a person once in the toolbar and
// each clicked face is assigned to them (as a confirmed, locked face), or use
// a tile's ＋ to mint a brand-new group seeded by that face.
function UngroupedFacesModal({
  people,
  handleClose,
}: {
  people: Person[];
  handleClose: () => void;
}) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const queryClient = useQueryClient();
  const [faces, setFaces] = useState<UngroupedFace[] | null>(null);
  const [count, setCount] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [targetId, setTargetId] = useState(0);
  // Faces with an assign in flight — blocks the double-click that would mint
  // two groups (or re-assign) before the tile disappears.
  const busy = useRef<Set<number>>(new Set());
  const ref = useRef(null);
  useOnClickOutside(ref, handleClose);

  const toast = (
    type: 'success' | 'error',
    title: string,
    message: string
  ) =>
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type, title, message },
    });

  const loadPage = async (offset: number) => {
    const data = await fetchUngroupedFaces(authToken, offset);
    setCount(data.count);
    setFaces((prev) =>
      offset === 0 ? data.faces : [...(prev ?? []), ...data.faces]
    );
  };

  useEffect(() => {
    loadPage(0).catch((err) => setLoadError(String(err)));
  }, []);

  // Mirrors loadingMore for the stream handler below — the subscription
  // closure mounts once and would otherwise read the first render's value.
  const loadingMoreRef = useRef(false);
  loadingMoreRef.current = loadingMore;

  // Live refresh: background clustering (in-scan passes, queued rebuilds) and
  // manual mutations broadcast "people-updated" while this modal is open. The
  // People grid re-queries on it, but this list is plain local state — without
  // this, tiles that just got grouped linger and read as a broken filter.
  // Debounced (broadcasts come in bursts). The count always updates; the
  // tiles are replaced only while just the first page is loaded — a user who
  // paged deeper keeps their scroll position and staleness resolves on close/
  // reopen (clicking a stale tile still does the right thing: it assigns).
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const unsubscribe = subscribeStream((type) => {
      if (type !== 'people-updated') return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        if (busy.current.size > 0 || loadingMoreRef.current) return;
        fetchUngroupedFaces(authToken, 0)
          .then((data) => {
            setCount(data.count);
            setFaces((prev) =>
              prev && prev.length > data.faces.length ? prev : data.faces
            );
          })
          .catch(() => {
            /* keep the current list; the next broadcast retries */
          });
      }, 1500);
    });
    return () => {
      if (timer) clearTimeout(timer);
      unsubscribe();
    };
  }, [authToken]);

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
    queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
  };

  // Assign one face (to an existing person or a fresh group). The tile
  // disappearing is the success feedback; errors toast.
  const assign = async (
    face: UngroupedFace,
    body: { personId?: number; newPerson?: boolean }
  ): Promise<{ created: boolean; name: string } | null> => {
    if (busy.current.has(face.id)) return null;
    busy.current.add(face.id);
    try {
      const res = await fetch(
        `${mediaServerBase}/api/faces/${face.id}/assign`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...authHeaders(authToken),
          },
          credentials: 'include',
          body: JSON.stringify(body),
        }
      );
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const out = (await res.json()) as { created: boolean; name: string };
      setFaces((prev) => prev && prev.filter((f) => f.id !== face.id));
      setCount((c) => Math.max(0, c - 1));
      refresh();
      return out;
    } catch (err) {
      toast('error', 'Failed to assign face', String(err));
      return null;
    } finally {
      busy.current.delete(face.id);
    }
  };

  const target = people.find((p) => p.id === targetId);
  const handleTileClick = (face: UngroupedFace) => {
    if (!target) return; // the toolbar hint explains the flow
    void assign(face, { personId: target.id });
  };
  const handleNewGroup = (face: UngroupedFace) => {
    void assign(face, { newPerson: true }).then(
      (out) =>
        out &&
        toast(
          'success',
          'New group created',
          `“${displayTagLabel(out.name)}” started from this face`
        )
    );
  };

  // Named people first (alphabetical), then the anonymous clusters.
  const sortedPeople = [...people].sort((a, b) => {
    const aU = a.name.startsWith('Unknown #');
    const bU = b.name.startsWith('Unknown #');
    if (aU !== bU) return aU ? 1 : -1;
    return displayTagLabel(a.name).localeCompare(displayTagLabel(b.name));
  });

  return (
    <div className="input-modal">
      <div className="input-modal-content face-review-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            Ungrouped faces
            {faces && (
              <span className="face-review-counts">
                {count.toLocaleString()} not in any group
              </span>
            )}
          </div>
          <div className="input-modal-close" onClick={handleClose}>
            <img src={cancel} />
          </div>
        </div>

        <div className="face-review-toolbar">
          <select
            className="ungrouped-assign-select"
            value={targetId}
            onChange={(e) => setTargetId(Number(e.target.value))}
            title="Every face you click is added to this person (confirmed — survives every regroup)"
          >
            <option value={0}>Assign clicked faces to…</option>
            {sortedPeople.map((p) => (
              <option key={p.id} value={p.id}>
                {displayTagLabel(p.name)}
              </option>
            ))}
          </select>
          <span className="ungrouped-assign-hint">
            {target
              ? `Click faces to add them to “${displayTagLabel(target.name)}”`
              : 'Pick a person, then click faces — or ＋ starts a new group'}
          </span>
        </div>

        <div className="face-review-grid">
          {!faces && !loadError && (
            <div className="face-review-status">Loading faces…</div>
          )}
          {loadError && (
            <div className="face-review-status">
              Couldn’t load faces: {loadError}
            </div>
          )}
          {faces && faces.length === 0 && (
            <div className="face-review-status">
              Every face is in a group. 🎉
            </div>
          )}
          {(faces ?? []).map((face) => (
            <div
              key={face.id}
              className={`face-review-tile${target ? ' assignable' : ''}`}
              onClick={() => handleTileClick(face)}
              title={`${face.path}${
                target
                  ? ` · click to add to “${displayTagLabel(target.name)}”`
                  : ''
              }`}
            >
              <FaceCrop faceId={face.id} authToken={authToken} size={120} />
              <div className="face-review-tile-actions">
                <button
                  type="button"
                  className="add"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleNewGroup(face);
                  }}
                  title="Start a brand-new group from this face"
                  aria-label="New group from this face"
                >
                  ＋
                </button>
              </div>
            </div>
          ))}
          {faces && faces.length < count && (
            <button
              type="button"
              className="face-review-more"
              disabled={loadingMore}
              onClick={() => {
                setLoadingMore(true);
                loadPage(faces.length)
                  .catch((err) =>
                    toast('error', 'Failed to load more faces', String(err))
                  )
                  .finally(() => setLoadingMore(false));
              }}
            >
              {loadingMore
                ? 'Loading…'
                : `Show more (${(count - faces.length).toLocaleString()} left)`}
            </button>
          )}
        </div>

        <p className="face-review-hint">
          These faces aren’t similar enough to anything to group on their own —
          small groups, hard angles, or odd lighting. Adding one by hand
          confirms it: it anchors its group in every future regroup and pulls
          lookalikes in with it.
        </p>
      </div>
    </div>
  );
}

// Drag-only chip: drop it onto any media item (list or detail view) to mint a
// brand-new person from that image's face. The face is pulled out of whatever
// cluster it sits in (with a veto so it can't drift back) and becomes the new
// group's confirmed seed + cover; the group then collects more faces by drag
// or clustering like any other person. Shares the 'PERSON' drag type with
// person cards — id 0 tells the drop handler to create instead of assign.
function NewGroupChip({ isDisabled }: { isDisabled: boolean }) {
  const [{ isDragging }, drag, dragPreview] = useDrag(
    () => ({
      type: 'PERSON',
      item: { id: 0, name: 'New group' },
      canDrag: !isDisabled,
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
    }),
    [isDisabled]
  );
  useHideNativeDragPreview(dragPreview);
  return (
    <div
      ref={drag}
      className={`people-new-group-chip${isDragging ? ' dragging' : ''}${
        isDisabled ? ' disabled' : ''
      }`}
      title="Drag onto an image to start a new group from that face. The face leaves its current group (and can't drift back) and seeds the new one — then drag the new card onto more images of that person."
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <circle cx="10" cy="8" r="3.2" />
        <path d="M4 19c0-3 2.7-4.8 6-4.8s6 1.8 6 4.8" />
        <path d="M18 5v6M15 8h6" />
      </svg>
      <span className="people-btn-label">New group</span>
    </div>
  );
}

// One person card: click filters, ✎ edits, and drag-onto-another merges.
// Drag/drop goes through react-dnd (type 'PERSON') — the app runs react-dnd's
// HTML5 backend globally (tag reorder, file drops), whose window-level capture
// handlers break raw HTML5 draggable elements.
function PersonCard({
  person,
  isDisabled,
  active,
  authToken,
  onSelect,
  onEdit,
  onReview,
  onMerge,
}: {
  person: Person;
  isDisabled: boolean;
  active: boolean;
  authToken: string | null;
  onSelect: (p: Person) => void;
  onEdit: (p: Person) => void;
  onReview: (p: Person) => void;
  onMerge: (from: Person, into: Person) => void;
}) {
  const [{ isDragging }, drag, dragPreview] = useDrag(
    () => ({
      type: 'PERSON',
      item: person,
      canDrag: !isDisabled,
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
    }),
    [person, isDisabled]
  );
  useHideNativeDragPreview(dragPreview);
  const [{ isOver, canDrop }, drop] = useDrop<
    Person,
    unknown,
    { isOver: boolean; canDrop: boolean }
  >(
    () => ({
      accept: 'PERSON',
      canDrop: (item) => item.id !== person.id,
      drop: (item) => onMerge(item, person),
      collect: (monitor) => ({
        isOver: monitor.isOver(),
        canDrop: monitor.canDrop(),
      }),
    }),
    [person, onMerge]
  );

  return (
    <div
      ref={(node) => drag(drop(node))}
      className={`person-card${active ? ' active' : ''}${
        isDisabled ? ' disabled' : ''
      }${isDragging ? ' dragging' : ''}${
        isOver && canDrop ? ' drop-target' : ''
      }`}
      onClick={() => onSelect(person)}
      title={`${displayTagLabel(person.name)} — ${person.faceCount} face${
        person.faceCount === 1 ? '' : 's'
      } in ${person.mediaCount} item${person.mediaCount === 1 ? '' : 's'}${
        describeModels(person.models)
          ? ` · ${describeModels(person.models)}`
          : ''
      } · drag onto another person to merge`}
    >
      <FaceCrop faceId={person.coverFaceId} authToken={authToken} />
      {person.lockedCount > 0 && (
        <span
          className="person-card-locked"
          title={`${person.lockedCount} of ${person.faceCount} faces confirmed by you — they survive every regroup`}
        >
          ✓{person.lockedCount}
        </span>
      )}
      <div className="person-card-info">
        <span className="person-card-name">{displayTagLabel(person.name)}</span>
        <span className="person-card-count">{person.mediaCount}</span>
      </div>
      <button
        type="button"
        className="person-card-review"
        onClick={(e) => {
          e.stopPropagation();
          onReview(person);
        }}
        title="Review faces: confirm the right ones, remove the wrong ones"
        aria-label={`Review faces of ${displayTagLabel(person.name)}`}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <rect x="3" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="3" width="7" height="7" rx="1" />
          <rect x="3" y="14" width="7" height="7" rx="1" />
          <path d="M15 17.5l2 2 4-4.5" />
        </svg>
      </button>
      <button
        type="button"
        className="person-card-edit"
        onClick={(e) => {
          e.stopPropagation();
          onEdit(person);
        }}
        title="Rename, merge, or delete"
        aria-label={`Edit ${displayTagLabel(person.name)}`}
      >
        ✎
      </button>
    </div>
  );
}

export default function PeopleGrid({ isDisabled }: { isDisabled: boolean }) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const initSessionId = useSelector(
    libraryService,
    (s) => s.context.initSessionId
  );
  const selectedTags = useSelector(
    libraryService,
    (s) => s.context.dbQuery.tags
  );
  const filteringMode = useSelector(
    libraryService,
    (s) => s.context.settings.filteringMode
  );
  const [editing, setEditing] = useState<Person | null>(null);
  const [editOpen, setEditOpen] = useState(false);
  const [reviewing, setReviewing] = useState<Person | null>(null);
  const [ungroupedOpen, setUngroupedOpen] = useState(false);
  const queryClient = useQueryClient();

  const {
    data: people,
    error,
    isLoading,
    isFetching,
    refetch,
  } = usePeople();

  // Ungrouped-face count for the "Group new faces" badge. Shares the
  // 'taxonomy' key prefix, so every invalidation that refreshes the people
  // list (clustering finished, faces assigned/rejected, people-updated
  // broadcasts) refreshes this too. Best-effort: on error just no badge.
  const { data: ungroupedCount } = useQuery<number, Error>(
    ['taxonomy', 'ungrouped-faces', initSessionId],
    () => fetchUngroupedCount(authToken),
    { enabled: !!initSessionId, staleTime: 60_000, retry: 1 }
  );

  // Person names, kept fresh for the stream handler below (the subscription
  // closure would otherwise pin the first render's list).
  const peopleNamesRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    peopleNamesRef.current = new Set((people ?? []).map((p) => p.name));
  }, [people]);

  // If the CURRENT library view filters on a person (their tag or the People
  // category), re-run the query in place so media move in/out the moment
  // faces are discarded, reassigned, or regrouped — no manual reload.
  // DELETED_ASSIGNMENT is the machine's "a tag was removed from an item"
  // refresh: it re-queries the current view and preserves scroll.
  const rerunLibraryQueryIfPersonFiltered = () => {
    const snapshot = libraryService.getSnapshot();
    if (!snapshot.matches({ library: 'loadedFromDB' })) return;
    const preds: Array<{ type: string; value: string }> =
      snapshot.context.query?.predicates ?? [];
    const legacyTags: string[] = snapshot.context.dbQuery?.tags ?? [];
    const referencesPeople =
      preds.some(
        (p) =>
          (p.type === 'tag' && peopleNamesRef.current.has(p.value)) ||
          (p.type === 'category' && p.value === PEOPLE_CATEGORY)
      ) || legacyTags.some((t) => peopleNamesRef.current.has(t));
    if (referencesPeople) {
      libraryService.send({ type: 'DELETED_ASSIGNMENT' });
    }
  };

  // Live refresh: while the People panel is mounted, watch the job stream and
  // refetch when a face job finishes — otherwise "Group new faces" (or a scan
  // started elsewhere) completes silently and the grid looks stale until the
  // next manual reload.
  useEffect(() => {
    // Shared /stream bus — never opens a second EventSource (Chromium caps
    // connections per origin at 6; see stream-bus.ts).
    return subscribeStream((type, event) => {
      // Live incremental updates: "people-updated" is broadcast by clustering
      // passes AND by every manual mutation (assign, reject, curate, merge,
      // …), so the grid and any person-filtered view stay current in every
      // window as changes happen.
      if (type === 'people-updated') {
        queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
        rerunLibraryQueryIfPersonFiltered();
        return;
      }
      if (type !== 'update' && type !== 'delete') return;
      try {
        const parsed = JSON.parse(event.data);
        const job = parsed?.job;
        if (!job) return;
        if (
          (job.command === 'faces' || job.command === 'faces-cluster') &&
          job.state === 'completed'
        ) {
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
          // Clustering assigns faces to people, which writes People tags
          // onto media — refresh the per-media tag lists too.
          queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
          rerunLibraryQueryIfPersonFiltered();
        }
      } catch {
        // malformed event — ignore
      }
    });
  }, [queryClient]);

  // Merge `from` into `into` (drag a card onto another card). Mirrors the
  // edit-modal Merge action; shared with the taxonomy search results.
  const handleMergeDrop = useMergePeople();

  const handleSelect = (person: Person) => {
    if (isDisabled) return;
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'tag',
          value: person.name,
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
      },
    });
  };

  // The Ungrouped pseudo-card filters like a person card: clicking adds a
  // faces:ungrouped predicate (media with detected faces, NONE of them in a
  // group yet — an item whose main face is grouped already carries its person
  // tag and would read as a broken filter here). Highlighted while that
  // predicate is in the query, mirroring person cards' active state.
  const queryPredicates = useSelector(
    libraryService,
    (s) =>
      (s.context.query?.predicates ?? []) as Array<{
        type: string;
        value: string;
        exclude: boolean;
      }>
  );
  const ungroupedFilterActive = queryPredicates.some(
    (p) => p.type === 'faces' && p.value === 'ungrouped' && !p.exclude
  );
  const handleUngroupedFilter = () => {
    if (isDisabled) return;
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'faces',
          value: 'ungrouped',
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
      },
    });
  };

  // Two-click confirm for the destructive rebuild (unnamed clusters are
  // dissolved and regrouped). Resets after a few seconds if not confirmed.
  const [rebuildArmed, setRebuildArmed] = useState(false);
  useEffect(() => {
    if (!rebuildArmed) return undefined;
    const t = window.setTimeout(() => setRebuildArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [rebuildArmed]);

  // Grouping tuner. Strictness is an OFFSET from each recognizer's default
  // threshold so one slider works across routed models (SFace/CCIP/BYO have
  // different cosine scales). Saved to the SERVER config (debounced below),
  // where it applies to every clustering pass — Group new faces, Rebuild,
  // in-scan passes — not just the "Regroup with these settings" button.
  const TUNING_KEY = 'lowkey:people-cluster-tuning';
  const defaultTuning = { offset: 0, minCluster: 3, minQuality: 0.75 };
  const [tuneOpen, setTuneOpen] = useState(false);
  const [tuning, setTuning] = useState(defaultTuning);
  // Set once the user drags a slider — a later server fetch must not clobber
  // what they're editing.
  const tuningTouched = useRef(false);
  // Tuning lives in the SERVER config (applies to every clustering pass on
  // every client: the buttons here, in-scan passes, lokictl jobs). Seed the
  // sliders from it; a leftover localStorage value from the old client-only
  // persistence migrates to the server once, so previously chosen settings
  // start applying globally.
  useQuery<typeof defaultTuning, Error>(
    ['taxonomy', 'face-tuning', initSessionId],
    () => fetchTuning(authToken),
    {
      enabled: !!initSessionId,
      staleTime: 60_000,
      retry: 1,
      onSuccess: (server) => {
        if (tuningTouched.current) return;
        try {
          const raw = window.localStorage.getItem(TUNING_KEY);
          if (raw) {
            const legacy = { ...defaultTuning, ...JSON.parse(raw) };
            window.localStorage.removeItem(TUNING_KEY);
            setTuning(legacy);
            tuningTouched.current = true; // triggers the save effect below
            return;
          }
        } catch {
          /* unreadable legacy value — just use the server's */
        }
        setTuning(server);
      },
    }
  );
  // Debounced persistence: slider drags settle for 600ms, then save.
  useEffect(() => {
    if (!tuningTouched.current) return undefined;
    const t = window.setTimeout(() => {
      saveTuning(authToken, tuning).catch((err) => {
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'error',
            title: 'Failed to save grouping settings',
            message: String(err),
          },
        });
      });
    }, 600);
    return () => window.clearTimeout(t);
  }, [tuning]);
  const updateTuning = (patch: Partial<typeof defaultTuning>) => {
    tuningTouched.current = true;
    setTuning((prev: typeof defaultTuning) => ({ ...prev, ...patch }));
  };
  const [regroupArmed, setRegroupArmed] = useState(false);
  useEffect(() => {
    if (!regroupArmed) return undefined;
    const t = window.setTimeout(() => setRegroupArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [regroupArmed]);

  // The job's own toast (ToastSystem, via the SSE stream) announces the run
  // with a formatted title/subtitle — no manual toast here or it doubles up.
  const runClusterJob = async (input: string) => {
    try {
      const res = await fetch(`${mediaServerBase}/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(authToken),
        },
        credentials: 'include',
        body: JSON.stringify({ input }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to start clustering',
          message: String(err),
        },
      });
    }
  };

  // Additive: assigns only currently-unassigned faces; never moves anything.
  const handleCluster = () => runClusterJob('faces-cluster');

  // Rebuild: dissolves the anonymous "Unknown #N" clusters and regroups them
  // from scratch. Named people and user-assigned faces are never touched.
  const handleRebuild = () => {
    if (!rebuildArmed) {
      setRebuildArmed(true);
      return;
    }
    setRebuildArmed(false);
    runClusterJob('faces-cluster --reset');
  };

  // Regroup everything with the tuned parameters: clears EVERY automatic
  // assignment (even inside named people — only manual labels survive) and
  // reclusters from scratch, so parameter changes actually take effect
  // instead of being locked in by the previous run's groups.
  const handleTunedRegroup = () => {
    if (!regroupArmed) {
      setRegroupArmed(true);
      return;
    }
    setRegroupArmed(false);
    const flags = [
      'faces-cluster',
      '--reset-all',
      `--threshold-offset=${tuning.offset.toFixed(2)}`,
      `--min-cluster=${tuning.minCluster}`,
      `--min-quality=${tuning.minQuality.toFixed(2)}`,
    ];
    runClusterJob(flags.join(' '));
  };

  // Empty-state CTA: kick off a whole-library face scan right here. People
  // then appear in this grid live (the scan clusters incrementally and
  // broadcasts people-updated), so the empty state resolves itself.
  const [scanStarted, setScanStarted] = useState(false);
  const handleScanLibrary = async () => {
    const query64 = btoa(unescape(encodeURIComponent('path:*')));
    setScanStarted(true);
    await runClusterJob(`faces --query64=${query64}`);
  };

  // The status views below share the silhouette icon.
  const emptyIcon = (
    <div className="people-empty-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />
      </svg>
    </div>
  );

  // First load (or waiting on session init): we don't KNOW yet whether there
  // are people — showing "No people yet" here misreads a slow fetch as an
  // empty library.
  if (!initSessionId || (isLoading && isFetching)) {
    return (
      <div className="people-empty">
        {emptyIcon}
        <div className="people-empty-title people-empty-loading">
          Loading people…
        </div>
      </div>
    );
  }

  if (error) {
    // Server is up but rejected us — a connection problem message would
    // send the user restarting a perfectly healthy server.
    if (/HTTP 401/.test(error.message)) {
      return (
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title">Sign in to see People</div>
          <p className="people-empty-body">
            The media server is running but this session isn’t signed in.
            People are managed by the server, so log in and this panel will
            fill in.
          </p>
        </div>
      );
    }
    // Electron only: no server config.json on this machine and no LOWKEY_PORT
    // override → the media server has never run here. "Start it" would be a
    // dead end; say what's actually missing. (In web mode the SPA is served
    // BY the server, so this state can't occur.)
    if (isElectron && !mediaServerConfigured) {
      return (
        <div className="people-empty">
          {emptyIcon}
          <div className="people-empty-title">Media server not installed</div>
          <p className="people-empty-body">
            People (face detection and grouping) are provided by the Lowkey
            Media Server, a companion app that runs alongside the viewer. It
            doesn’t look like it has been installed on this machine yet.
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
          The media server is installed but unreachable right now. Start it
          (or check the connection), then try again.
        </p>
        <button
          type="button"
          className="people-empty-cta"
          onClick={() => refetch()}
        >
          Try again
        </button>
        <p className="people-empty-hint">{error.message}</p>
      </div>
    );
  }

  const list = people ?? [];
  const named = list.filter((p) => !p.name.startsWith('Unknown #'));
  const unknown = list.filter((p) => p.name.startsWith('Unknown #'));

  const renderCard = (person: Person) => (
    <PersonCard
      key={person.id}
      person={person}
      isDisabled={isDisabled}
      active={selectedTags.includes(person.name)}
      authToken={authToken}
      onSelect={handleSelect}
      onEdit={(p) => {
        setEditing(p);
        setEditOpen(true);
      }}
      onReview={(p) => setReviewing(p)}
      onMerge={handleMergeDrop}
    />
  );

  return (
    <div className="people-grid-wrap">
      {list.length > 0 && (
        <div className="people-grid-toolbar">
          <NewGroupChip isDisabled={isDisabled} />
          <button
            type="button"
            className="people-cluster-btn"
            onClick={handleCluster}
            title={`Group new faces: assign unassigned faces to people. Additive — never moves existing assignments.${
              ungroupedCount != null
                ? ` ${ungroupedCount.toLocaleString()} face${
                    ungroupedCount === 1 ? '' : 's'
                  } waiting to be grouped.`
                : ''
            }`}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <circle cx="9" cy="8" r="3.2" />
              <path d="M3 19c0-3 2.7-4.8 6-4.8s6 1.8 6 4.8" />
              <path d="M18 6v6M15 9h6" />
            </svg>
            <span className="people-btn-label">Group new faces</span>
            {!!ungroupedCount && (
              <span
                className="people-btn-count"
                aria-label={`${ungroupedCount.toLocaleString()} ungrouped faces`}
              >
                {ungroupedCount > 9999
                  ? `${Math.floor(ungroupedCount / 1000)}k`
                  : ungroupedCount.toLocaleString()}
              </span>
            )}
          </button>
          <button
            type="button"
            className={`people-cluster-btn${rebuildArmed ? ' danger' : ''}`}
            onClick={handleRebuild}
            title={
              rebuildArmed
                ? 'Click again to confirm the rebuild.'
                : 'Rebuild: dissolve the Unknown clusters and regroup them from scratch. Named people and manually assigned faces are never touched.'
            }
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d="M20 12a8 8 0 1 1-2.34-5.66" />
              <path d="M20 3v4h-4" />
            </svg>
            <span className="people-btn-label">
              {rebuildArmed ? 'Confirm rebuild' : 'Rebuild groups'}
            </span>
          </button>
          <button
            type="button"
            className={`people-cluster-btn${tuneOpen ? ' active' : ''}`}
            onClick={() => setTuneOpen((v) => !v)}
            title="Tune grouping: adjust strictness and floors. Saved on the server and used by every grouping run, including scans."
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <path d="M4 8h10M18 8h2M4 16h2M10 16h10" />
              <circle cx="15.5" cy="8" r="2" />
              <circle cx="7.5" cy="16" r="2" />
            </svg>
            <span className="people-btn-label">Tune</span>
          </button>
        </div>
      )}
      {list.length > 0 && tuneOpen && (
        <div className="people-tune-panel">
          <div className="people-tune-row">
            <label htmlFor="people-tune-strictness">
              Match strictness
              <span className="people-tune-value">
                {tuning.offset === 0
                  ? 'Default'
                  : `${tuning.offset > 0 ? '+' : ''}${tuning.offset.toFixed(2)}`}
              </span>
            </label>
            <input
              id="people-tune-strictness"
              type="range"
              min={-0.1}
              max={0.2}
              step={0.01}
              value={tuning.offset}
              onChange={(e) => updateTuning({ offset: Number(e.target.value) })}
            />
            <div className="people-tune-hints">
              <span>Looser · bigger groups</span>
              <span>Stricter · purer groups</span>
            </div>
          </div>
          <div className="people-tune-row">
            <label htmlFor="people-tune-minsize">
              Minimum group size
              <span className="people-tune-value">{tuning.minCluster}</span>
            </label>
            <input
              id="people-tune-minsize"
              type="range"
              min={2}
              max={10}
              step={1}
              value={tuning.minCluster}
              onChange={(e) => updateTuning({ minCluster: Number(e.target.value) })}
            />
            <div className="people-tune-hints">
              <span>More groups</span>
              <span>Only well-supported groups</span>
            </div>
          </div>
          <div className="people-tune-row">
            <label htmlFor="people-tune-quality">
              Face quality floor
              <span className="people-tune-value">{tuning.minQuality.toFixed(2)}</span>
            </label>
            <input
              id="people-tune-quality"
              type="range"
              min={0.5}
              max={0.95}
              step={0.05}
              value={tuning.minQuality}
              onChange={(e) => updateTuning({ minQuality: Number(e.target.value) })}
            />
            <div className="people-tune-hints">
              <span>Include blurry faces</span>
              <span>Clear faces only</span>
            </div>
          </div>
          <p className="people-tune-note">
            Saved automatically and used by <strong>all</strong> grouping —
            “Group new faces”, “Rebuild groups”, and scans. “Regroup” below
            additionally rebuilds everything from scratch right now.
          </p>
          <div className="people-tune-actions">
            <button
              type="button"
              className="people-tune-reset"
              onClick={() => updateTuning(defaultTuning)}
            >
              Reset to defaults
            </button>
            <button
              type="button"
              className={`people-cluster-btn${regroupArmed ? ' danger' : ''}`}
              onClick={handleTunedRegroup}
              title={
                regroupArmed
                  ? 'Click again to confirm. All automatic groups are dissolved and rebuilt with these settings.'
                  : 'Dissolve ALL automatic groups (including inside named people) and regroup with these settings. Faces you confirmed stay put, and faces you removed from a group can never return to it.'
              }
            >
              {regroupArmed ? 'Confirm full regroup' : 'Regroup with these settings'}
            </button>
          </div>
        </div>
      )}
      {list.length === 0 && (
        <div className="people-empty">
          <div className="people-empty-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="8" r="4" />
              <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />
            </svg>
          </div>
          <div className="people-empty-title">No people yet</div>
          <p className="people-empty-body">
            Scan your library to find faces. People appear here on their own
            while the scan runs — name them, merge duplicates, and filter your
            library by who’s in the picture.
          </p>
          <button
            type="button"
            className="people-empty-cta"
            onClick={handleScanLibrary}
            disabled={scanStarted}
          >
            {scanStarted ? 'Scanning — people will appear here' : 'Scan library for faces'}
          </button>
          <p className="people-empty-hint">
            Photos and drawn characters are each matched with the right
            recognizer automatically. You can also start a scan anytime from
            the context palette (Generate → Faces).
          </p>
        </div>
      )}
      {named.length > 0 && (
        <div className="people-grid">{named.map(renderCard)}</div>
      )}
      {unknown.length > 0 && (
        <>
          <div className="people-grid-heading">Unnamed clusters</div>
          <div className="people-grid">{unknown.map(renderCard)}</div>
        </>
      )}
      {!!ungroupedCount && (
        <>
          <div className="people-grid-heading">Not grouped yet</div>
          <div className="people-grid">
            <div
              className={`person-card people-ungrouped-card${
                ungroupedFilterActive ? ' active' : ''
              }${isDisabled ? ' disabled' : ''}`}
              onClick={handleUngroupedFilter}
              title={`${ungroupedCount.toLocaleString()} face${
                ungroupedCount === 1 ? ' isn’t' : 's aren’t'
              } in any group — click to filter the library to media whose faces aren’t grouped yet`}
            >
              <div
                className="person-card-face person-card-face--empty"
                aria-hidden="true"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <circle cx="12" cy="8" r="4" strokeDasharray="3 2.5" />
                  <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" strokeDasharray="3 2.5" />
                </svg>
              </div>
              <div className="person-card-info">
                <span className="person-card-name">Ungrouped faces</span>
                <span className="person-card-count">
                  {ungroupedCount.toLocaleString()}
                </span>
              </div>
              <button
                type="button"
                className="person-card-review"
                onClick={(e) => {
                  e.stopPropagation();
                  setUngroupedOpen(true);
                }}
                title="Review ungrouped faces: assign them to people by hand"
                aria-label="Review ungrouped faces"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <rect x="3" y="3" width="7" height="7" rx="1" />
                  <rect x="14" y="3" width="7" height="7" rx="1" />
                  <rect x="3" y="14" width="7" height="7" rx="1" />
                  <path d="M15 17.5l2 2 4-4.5" />
                </svg>
              </button>
            </div>
          </div>
        </>
      )}
      {editOpen && (
        <PersonEditModal
          person={editing}
          people={list}
          handleClose={() => {
            setEditing(null);
            setEditOpen(false);
          }}
        />
      )}
      {reviewing && (
        <FaceReviewModal
          person={reviewing}
          handleClose={() => setReviewing(null)}
        />
      )}
      {ungroupedOpen && (
        <UngroupedFacesModal
          people={list}
          handleClose={() => setUngroupedOpen(false)}
        />
      )}
    </div>
  );
}

// PeopleSearchResults renders the People-category matches of the taxonomy
// type-ahead as real person cards — face-crop preview plus the full person
// controls (click filters, ✎ renames/merges, review drawer, drag onto media
// to assign, drag onto another card to merge) — instead of the blank generic
// tag cards person tags used to fall into. Names that don't resolve to a
// person (list still loading, or the server is away) render nothing here;
// the caller keeps them out of the tag grid either way.
export function PeopleSearchResults({
  names,
  isDisabled,
}: {
  names: string[];
  isDisabled: boolean;
}) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);
  const selectedTags = useSelector(
    libraryService,
    (s) => s.context.dbQuery.tags
  );
  const filteringMode = useSelector(
    libraryService,
    (s) => s.context.settings.filteringMode
  );
  const { data: people } = usePeople();
  const handleMerge = useMergePeople();
  const [editing, setEditing] = useState<Person | null>(null);
  const [reviewing, setReviewing] = useState<Person | null>(null);

  const byName = new Map((people ?? []).map((p) => [p.name, p]));
  const matched = names
    .map((n) => byName.get(n))
    .filter((p): p is Person => !!p);
  if (matched.length === 0) return null;

  const handleSelect = (person: Person) => {
    if (isDisabled) return;
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'tag',
          value: person.name,
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
      },
    });
  };

  return (
    <div className="search-people-results">
      <div className="people-grid-heading">People</div>
      <div className="people-grid">
        {matched.map((person) => (
          <PersonCard
            key={person.id}
            person={person}
            isDisabled={isDisabled}
            active={selectedTags.includes(person.name)}
            authToken={authToken}
            onSelect={handleSelect}
            onEdit={(p) => setEditing(p)}
            onReview={(p) => setReviewing(p)}
            onMerge={handleMerge}
          />
        ))}
      </div>
      {editing && (
        <PersonEditModal
          person={editing}
          people={people ?? []}
          handleClose={() => setEditing(null)}
        />
      )}
      {reviewing && (
        <FaceReviewModal
          person={reviewing}
          handleClose={() => setReviewing(null)}
        />
      )}
    </div>
  );
}
