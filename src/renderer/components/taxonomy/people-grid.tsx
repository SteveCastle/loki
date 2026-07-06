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
import { GlobalStateContext } from '../../state';
import { mediaServerBase } from '../../platform';
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
  // Recognizer(s) the cluster's faces were embedded with (comma-separated;
  // normally one) — distinguishes photo-face clusters from anime-character
  // clusters.
  models?: string;
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

// A face crop loaded with auth (an <img src> can't carry a bearer token in
// Electron), delivered as a blob URL. Falls back to a silhouette.
function FaceCrop({
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
        await call(`/api/people/${person.id}/rename`, {
          method: 'POST',
          body: JSON.stringify({ name: trimmed }),
        });
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
                    Unassigns their faces and removes the tag; scanned faces
                    are kept and can be regrouped later
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
  onMerge,
}: {
  person: Person;
  isDisabled: boolean;
  active: boolean;
  authToken: string | null;
  onSelect: (p: Person) => void;
  onEdit: (p: Person) => void;
  onMerge: (from: Person, into: Person) => void;
}) {
  const [{ isDragging }, drag] = useDrag(
    () => ({
      type: 'PERSON',
      item: person,
      canDrag: !isDisabled,
      collect: (monitor) => ({ isDragging: monitor.isDragging() }),
    }),
    [person, isDisabled]
  );
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
      <div className="person-card-info">
        <span className="person-card-name">{displayTagLabel(person.name)}</span>
        <span className="person-card-count">{person.mediaCount}</span>
      </div>
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
  const queryClient = useQueryClient();

  // Under the 'taxonomy' key prefix so existing broad invalidations (tag
  // mutations, DB swaps) refresh the people list too.
  const { data: people, error } = useQuery<Person[], Error>(
    ['taxonomy', 'people', initSessionId],
    () => fetchPeople(authToken),
    { enabled: !!initSessionId, staleTime: 60_000 }
  );

  // Live refresh: while the People panel is mounted, watch the job stream and
  // refetch when a face job finishes — otherwise "Group new faces" (or a scan
  // started elsewhere) completes silently and the grid looks stale until the
  // next manual reload.
  useEffect(() => {
    // Shared /stream bus — never opens a second EventSource (Chromium caps
    // connections per origin at 6; see stream-bus.ts).
    return subscribeStream((type, event) => {
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
        }
      } catch {
        // malformed event — ignore
      }
    });
  }, [queryClient]);

  // Merge `from` into `into` (drag a card onto another card). Every face and
  // taxonomy row of the dragged person moves to the target; the dragged
  // person is removed. Mirrors the edit-modal Merge action.
  const handleMergeDrop = async (from: Person, into: Person) => {
    try {
      const res = await fetch(`${mediaServerBase}/api/people/${from.id}/merge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(authToken),
        },
        credentials: 'include',
        body: JSON.stringify({ intoId: into.id }),
      });
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

  // Two-click confirm for the destructive rebuild (unnamed clusters are
  // dissolved and regrouped). Resets after a few seconds if not confirmed.
  const [rebuildArmed, setRebuildArmed] = useState(false);
  useEffect(() => {
    if (!rebuildArmed) return undefined;
    const t = window.setTimeout(() => setRebuildArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [rebuildArmed]);

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

  if (error) {
    return (
      <div className="people-grid-empty">
        People need the media server ({error.message}).
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
      onMerge={handleMergeDrop}
    />
  );

  return (
    <div className="people-grid-wrap">
      <div className="people-grid-toolbar">
        <button
          type="button"
          className="people-cluster-btn"
          onClick={handleCluster}
          title="Assign unassigned faces to people. Additive — never moves existing assignments."
        >
          Group new faces
        </button>
        <button
          type="button"
          className={`people-cluster-btn${rebuildArmed ? ' danger' : ''}`}
          onClick={handleRebuild}
          title="Dissolve the Unknown clusters and regroup them from scratch. Named people and manually assigned faces are never touched."
        >
          {rebuildArmed ? 'Click again to confirm' : 'Rebuild unnamed groups'}
        </button>
      </div>
      {list.length === 0 && (
        <div className="people-grid-empty">
          No people yet. Scan faces from the context palette (Generate →
          Faces), then group them here.
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
    </div>
  );
}
