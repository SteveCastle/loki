// SelectionTags — bulk tag editing for the context palette's multi-item
// selection. Rendered under the Merge action in the "Selection" block.
//
// Two halves:
//  - Shared tags: the intersection of tags across EVERY selected item, shown
//    as removable chips. Removing is two-click (arm → confirm, the palette's
//    destructive-action pattern) and deletes every occurrence of the label
//    from each selected path.
//  - Add: the shared 'curated' tag type-ahead (same scope + worker index the
//    command palette uses). Choosing a result assigns it to ALL selected
//    items in one create-assignment call (the IPC/HTTP contract takes a path
//    array natively).
//
// Everything is gated on `active` so a closed palette (this component stays
// mounted with the palette) costs nothing: no tag fetches, no search index.
import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  invoke,
  isElectron,
  mediaServerBase,
  mediaServerConfigured,
} from '../../platform';
import { displayTagLabel } from '../../tag-display';
import { useTagSearch, type TagConcept } from '../../hooks/useTagSearch';

interface AssignedTag {
  tag_label: string;
  category_label?: string;
  time_stamp?: number;
}

interface SelectionTagsProps {
  selection: string[];
  // Palette open + multi-selection + the same server/Electron gate as the
  // surrounding Selection block.
  active: boolean;
  applyTagPreview: boolean;
  authToken: string | null;
  libraryService: any;
}

// Keep the suggestion list palette-sized; the full ranked set is capped
// upstream anyway (useTagSearch).
const SUGGESTION_CAP = 8;

// A shared tag: the label present on every selected item, with the category
// of its first-seen occurrence (categories can differ per item; the label is
// what removal targets).
interface SharedTag {
  label: string;
  category?: string;
}

// Removing a person's tag from an item means "this person is not in this
// item": their faces must leave the group too (veto + cannot-links on the
// server). Web mode's assignment DELETE does this itself; Electron deletes go
// straight to SQLite over IPC, so the face discard needs this explicit,
// best-effort server call per path. (Same contract as metadata/tags.tsx.)
async function rejectPersonFaces(
  path: string,
  personName: string,
  authToken: string | null
) {
  const res = await fetch(`${mediaServerBase}/api/media/reject-person`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
    },
    credentials: 'include',
    body: JSON.stringify({ path, name: personName }),
  });
  // 404 = tag label isn't a person (or already gone) — nothing to discard.
  if (!res.ok && res.status !== 404) {
    throw new Error((await res.text()) || `HTTP ${res.status}`);
  }
}

export default function SelectionTags({
  selection,
  active,
  applyTagPreview,
  authToken,
  libraryService,
}: SelectionTagsProps) {
  const queryClient = useQueryClient();

  const [text, setText] = useState('');
  const [highlightIdx, setHighlightIdx] = useState(0);
  // Label currently being added/removed — disables the whole strip so a slow
  // bulk write (create-assignment sleeps ~3s on multi-path) can't be stacked.
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  // Two-click remove: first click arms the chip (turns red), second removes.
  const [armedLabel, setArmedLabel] = useState<string | null>(null);

  // Disarm + clear the draft whenever the palette closes or the selection
  // changes — a confirm click must never land on a different set than the one
  // that was armed.
  useEffect(() => {
    setArmedLabel(null);
    setText('');
  }, [active, selection]);
  // Auto-disarm, like the merge button.
  useEffect(() => {
    if (!armedLabel) return;
    const t = window.setTimeout(() => setArmedLabel(null), 4000);
    return () => window.clearTimeout(t);
  }, [armedLabel]);

  // Per-item tags for the whole selection, fetched together. The key is
  // prefixed 'tags-by-path' so every existing invalidation of that family
  // (tag edits anywhere in the app) refreshes this too.
  const selectionKey = useMemo(() => selection.join('\n'), [selection]);
  const { data: perItemTags } = useQuery<AssignedTag[][], Error>(
    ['tags-by-path', 'selection', selectionKey],
    async () => {
      const metas = await Promise.all(
        selection.map(
          (path) =>
            invoke('load-tags-by-media-path', [{ path }]) as Promise<{
              tags?: AssignedTag[];
            } | null>
        )
      );
      return metas.map((m) => m?.tags ?? []);
    },
    { enabled: active && selection.length > 1 }
  );

  // Tags shared by ALL selected items: label-set intersection, categories
  // remembered from the first occurrence for display + the People check.
  const sharedTags: SharedTag[] = useMemo(() => {
    if (!perItemTags || perItemTags.length === 0) return [];
    const category = new Map<string, string | undefined>();
    let shared: Set<string> | null = null;
    for (const tags of perItemTags) {
      const labels = new Set<string>();
      for (const t of tags) {
        if (!t?.tag_label) continue;
        labels.add(t.tag_label);
        if (!category.has(t.tag_label)) {
          category.set(t.tag_label, t.category_label);
        }
      }
      if (shared === null) {
        shared = labels;
      } else {
        const prev: Set<string> = shared;
        shared = new Set([...prev].filter((l) => labels.has(l)));
      }
      if (shared.size === 0) break;
    }
    return [...(shared ?? [])]
      .sort((a, b) => displayTagLabel(a).localeCompare(displayTagLabel(b)))
      .map((label) => ({ label, category: category.get(label) }));
  }, [perItemTags]);
  const sharedLabels = useMemo(
    () => new Set(sharedTags.map((t) => t.label)),
    [sharedTags]
  );

  // Type-ahead over the curated scope (the palette standard — the ~183K
  // machine-suggested bucket is noise here). Enabled only while typing so the
  // index fetch never runs for a palette that isn't using this feature.
  const { results } = useTagSearch(text, active && text.length > 0, 'curated');
  // A tag already on every item is a no-op to add — drop it from the list.
  const suggestions = useMemo(
    () =>
      results.filter((t) => !sharedLabels.has(t.label)).slice(0, SUGGESTION_CAP),
    [results, sharedLabels]
  );
  useEffect(() => {
    setHighlightIdx(0);
  }, [text]);

  const invalidateTagCaches = () => {
    queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
    queryClient.invalidateQueries({ queryKey: ['metadata'] });
    queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
  };

  const toast = (
    type: 'success' | 'error',
    title: string,
    message?: string
  ) => {
    libraryService.send({
      type: 'ADD_TOAST',
      data: { type, title, message, durationMs: 2500 },
    });
  };

  const handleAdd = async (t: TagConcept) => {
    if (busyLabel) return;
    setBusyLabel(t.label);
    try {
      // One call — the create-assignment contract takes the path array.
      await invoke('create-assignment', [
        selection,
        t.label,
        t.category,
        null,
        applyTagPreview,
      ]);
      invalidateTagCaches();
      setText('');
      toast(
        'success',
        `Applied "${displayTagLabel(t.label)}"`,
        `Tagged ${selection.length} items`
      );
    } catch (err) {
      toast(
        'error',
        'Tag failed',
        err instanceof Error ? err.message : 'Could not apply tag'
      );
    } finally {
      setBusyLabel(null);
    }
  };

  const handleRemove = async (tag: SharedTag) => {
    if (busyLabel) return;
    if (armedLabel !== tag.label) {
      setArmedLabel(tag.label);
      return;
    }
    setArmedLabel(null);
    setBusyLabel(tag.label);
    const faceFailures: string[] = [];
    try {
      // Sequential: delete-assignment has no lock-retry wrapper, and the Go
      // server writes to the same SQLite file — don't pile writes up.
      for (const path of selection) {
        // time_stamp 0 → every occurrence of the label on the path.
        await invoke('delete-assignment', [
          path,
          { tag_label: tag.label, time_stamp: 0 },
        ]);
        if (
          isElectron &&
          mediaServerConfigured &&
          tag.category === 'People'
        ) {
          try {
            await rejectPersonFaces(path, tag.label, authToken);
          } catch {
            faceFailures.push(path);
          }
        }
      }
      libraryService.send({ type: 'DELETED_ASSIGNMENT' });
      invalidateTagCaches();
      toast(
        'success',
        `Removed "${displayTagLabel(tag.label)}"`,
        `From ${selection.length} items`
      );
      if (faceFailures.length > 0) {
        toast(
          'error',
          'Faces not discarded from group',
          `The tag was removed, but discarding faces from "${displayTagLabel(
            tag.label
          )}" failed on ${faceFailures.length} item${
            faceFailures.length === 1 ? '' : 's'
          }`
        );
      }
    } catch (err) {
      toast(
        'error',
        'Remove failed',
        err instanceof Error ? err.message : 'Could not remove tag'
      );
    } finally {
      setBusyLabel(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Keep palette-global hotkeys (and the app's) away from typing, but let
    // Escape through — it closes the palette from anywhere.
    if (e.key !== 'Escape') e.stopPropagation();
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightIdx((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      const pick = suggestions[Math.min(highlightIdx, suggestions.length - 1)];
      if (pick) handleAdd(pick);
    }
  };

  return (
    <div className="selection-tags">
      <span className="selection-tags-label">
        Tags on all {selection.length} items
      </span>
      <div className="selection-tags-shared">
        {sharedTags.length === 0 && (
          <span className="selection-tags-empty">
            {perItemTags ? 'No shared tags' : 'Loading…'}
          </span>
        )}
        {sharedTags.map((tag) => {
          const armed = armedLabel === tag.label;
          const busy = busyLabel === tag.label;
          return (
            <span
              key={tag.label}
              className={`selection-tag-chip${armed ? ' armed' : ''}${
                busy ? ' busy' : ''
              }`}
              title={
                tag.category ? `${tag.category} tag on every selected item` : undefined
              }
            >
              {displayTagLabel(tag.label)}
              <button
                type="button"
                className="selection-tag-remove"
                disabled={!!busyLabel}
                onClick={() => handleRemove(tag)}
                title={
                  armed
                    ? `Click again to remove from all ${selection.length} items`
                    : `Remove "${displayTagLabel(tag.label)}" from all ${
                        selection.length
                      } items (asks to confirm)`
                }
                aria-label={
                  armed
                    ? `Confirm removing ${displayTagLabel(tag.label)}`
                    : `Remove ${displayTagLabel(tag.label)} from all items`
                }
              >
                {busy ? '…' : armed ? '✓?' : '×'}
              </button>
            </span>
          );
        })}
      </div>
      <input
        className="selection-tags-input"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        onKeyUp={(e) => e.stopPropagation()}
        placeholder={`Add tag to all ${selection.length} items…`}
        aria-label="Add tag to all selected items"
        disabled={!!busyLabel}
      />
      {text.length > 0 && suggestions.length > 0 && (
        <div className="selection-tags-results" role="listbox">
          {suggestions.map((t, i) => (
            <div
              key={`${t.category}:${t.label}`}
              role="option"
              aria-selected={i === highlightIdx}
              className={`selection-tag-row${
                i === highlightIdx ? ' highlighted' : ''
              }`}
              onMouseEnter={() => setHighlightIdx(i)}
              onClick={() => handleAdd(t)}
              title={`Apply "${displayTagLabel(t.label)}" to all ${
                selection.length
              } selected items`}
            >
              <span className="selection-tag-row-prefix">#</span>
              <span className="selection-tag-row-value">
                {displayTagLabel(t.label)}
              </span>
              {t.category && t.category !== 'Suggested' && (
                <span className="selection-tag-row-meta">{t.category}</span>
              )}
            </div>
          ))}
        </div>
      )}
      {text.length > 1 && suggestions.length === 0 && (
        <div className="selection-tags-empty">No matching tags</div>
      )}
    </div>
  );
}
