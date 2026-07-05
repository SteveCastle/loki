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
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import { GlobalStateContext } from '../../state';
import { mediaServerBase } from '../../platform';
import './new-modal.css';
import './people-grid.css';

export const PEOPLE_CATEGORY = 'People';

export interface Person {
  id: number;
  name: string;
  coverFaceId?: number;
  faceCount: number;
  mediaCount: number;
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
  const [name, setName] = useState(person?.name ?? '');
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
      } else if (trimmed !== person.name) {
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
                      Move every face and tag of “{person.name}” onto another
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
                          {p.name} ({p.faceCount})
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
                  <div className="action-row-title">Delete person</div>
                  <div className="action-row-description">
                    Unassigns their faces and removes the tag; scanned faces
                    are kept and can be regrouped later
                  </div>
                </div>
                <button onClick={handleDelete}>Delete</button>
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

  // Under the 'taxonomy' key prefix so existing broad invalidations (tag
  // mutations, DB swaps) refresh the people list too.
  const { data: people, error } = useQuery<Person[], Error>(
    ['taxonomy', 'people', initSessionId],
    () => fetchPeople(authToken),
    { enabled: !!initSessionId, staleTime: 60_000 }
  );

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

  const handleCluster = async () => {
    try {
      const res = await fetch(`${mediaServerBase}/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(authToken),
        },
        credentials: 'include',
        body: JSON.stringify({ input: 'faces-cluster' }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'info',
          title: 'Clustering faces',
          message: 'Grouping scanned faces into people…',
        },
      });
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
    <div
      key={person.id}
      className={`person-card${
        selectedTags.includes(person.name) ? ' active' : ''
      }${isDisabled ? ' disabled' : ''}`}
      onClick={() => handleSelect(person)}
      title={`${person.name} — ${person.faceCount} face${
        person.faceCount === 1 ? '' : 's'
      } in ${person.mediaCount} item${person.mediaCount === 1 ? '' : 's'}`}
    >
      <FaceCrop faceId={person.coverFaceId} authToken={authToken} />
      <div className="person-card-info">
        <span className="person-card-name">{person.name}</span>
        <span className="person-card-count">{person.mediaCount}</span>
      </div>
      <button
        type="button"
        className="person-card-edit"
        onClick={(e) => {
          e.stopPropagation();
          setEditing(person);
          setEditOpen(true);
        }}
        title="Rename, merge, or delete"
        aria-label={`Edit ${person.name}`}
      >
        ✎
      </button>
    </div>
  );

  return (
    <div className="people-grid-wrap">
      <div className="people-grid-toolbar">
        <button
          type="button"
          className="people-cluster-btn"
          onClick={handleCluster}
          title="Group scanned faces into people (never overwrites your manual labels)"
        >
          Group new faces
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
