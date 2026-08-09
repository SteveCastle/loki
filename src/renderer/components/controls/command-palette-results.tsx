// CommandPaletteResults — the command palette's type-ahead ENGINE.
//
// Everything data-bound behind the palette's search input lives here: the
// shared tag-index lookup (useTagSearch), the categories fetch, and the
// suggestion sections. It is deliberately a separate component from
// CommandPaletteSearch so the palette can mount it LATE: opening the palette
// renders the shell + input in the first commit, and this subtree — with its
// query subscriptions, effects and IPC — mounts in the task after that paint
// (or immediately on the first keystroke, whichever comes first). See
// useAfterPaint.
//
// It reports its ordered, navigable rows up to the parent (tags first, then the
// suggestion rows) so the parent can own the keyboard highlight across both
// groups while the results themselves stay down here.
import { useEffect, useMemo, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Predicate } from '../../query/types';
import { invoke } from '../../platform';
import { useTagSearch } from '../../hooks/useTagSearch';
import { displayTagLabel } from '../../tag-display';
import type { TagConcept } from '../../hooks/useTagSearch';
import { DEFAULT_TAG_SCOPE, type TagScope } from '../../search/tag-scopes';
import { isWarm } from '../../search/tag-search-service';
import { markPalette, endPaletteOpen } from '../../palette-trace';
import SuggestionSections from '../taxonomy/suggestion-sections';
import type { SuggestionItem } from '../taxonomy/suggestion-sections';
import TagPlusIcon from '../icons/tag-plus-icon';

// Compact cap for the palette's Tags section — the sidebar shows far more, but
// the floating palette is tight on space.
const PALETTE_TAG_CAP = 12;

interface CategoryLite {
  label: string;
  weight?: number;
  description?: string;
}

async function loadCategories(): Promise<CategoryLite[]> {
  const result = await invoke('load-categories', []);
  return (result as CategoryLite[]) ?? [];
}

interface CommandPaletteResultsProps {
  // The raw (un-debounced) search text.
  text: string;
  // Whether the rows should render. False while the engine is only warming
  // (palette open, nothing typed) or while "search by meaning" is on — the
  // hooks still run, but nothing is drawn.
  active: boolean;
  libraryService: any;
  // Path of the media item under the cursor; the apply-tag button targets it.
  currentPath?: string;
  // Keyboard highlight, owned by the parent so it can span tags + suggestions.
  highlightedKey: string | null;
  onHighlightKey: (key: string) => void;
  // Report the ordered navigable rows (tags first, then suggestions).
  onNavItemsChange: (items: SuggestionItem[]) => void;
  onCommit: (predicate: Predicate) => void;
  // Which slice of the tag table to fetch and fuzzy-search. See
  // ../../search/tag-scopes: 'curated' skips the autotagger's "Suggested"
  // bucket, which is ~97% of the table and the reason the palette used to lag.
  tagScope?: TagScope;
}

export default function CommandPaletteResults({
  text,
  active,
  libraryService,
  currentPath,
  highlightedKey,
  onHighlightKey,
  onNavItemsChange,
  onCommit,
  tagScope = DEFAULT_TAG_SCOPE,
}: CommandPaletteResultsProps) {
  const queryClient = useQueryClient();
  const applyTagPreview = useSelector(
    libraryService,
    (s: any) => s.context.settings.applyTagPreview
  );
  const initSessionId = useSelector(
    libraryService,
    (s: any) => s.context.initSessionId
  );

  const { results: tagResults } = useTagSearch(
    text,
    active && text.length > 0,
    tagScope
  );

  const { data: categories } = useQuery<CategoryLite[], Error>(
    ['taxonomy', 'categories', initSessionId],
    loadCategories,
    { staleTime: Infinity, cacheTime: Infinity }
  );

  // The palette is "ready entirely" once its engine has its data: the
  // categories list resolved and the fuzzy index for this scope is built. Both
  // are one-time per session, which is exactly why the FIRST open is the slow
  // one — see ../../palette-trace.
  useEffect(() => {
    markPalette('results-mounted');
  }, []);
  useEffect(() => {
    if (categories) markPalette('categories-ready');
  }, [categories]);
  useEffect(() => {
    if (!categories) return;
    // Whether the fuzzy index for this scope already exists. It is normally
    // warmed at startup, so on a healthy session this is true before the user
    // ever right-clicks; if it is false here, the first keystroke will pay for
    // the fetch + index build and THAT is the lag being felt.
    if (isWarm(tagScope)) markPalette('index-warm');
    markPalette('data-ready');
    endPaletteOpen(isWarm(tagScope) ? 'ready' : 'ready-index-cold');
  }, [categories, tagScope]);

  // Apply a tag to the current media item (an assignment) — distinct from
  // clicking the row, which adds the tag to the search query.
  const applyTagToCurrent = async (t: TagConcept) => {
    if (!currentPath) return;
    try {
      await invoke('create-assignment', [
        [currentPath],
        t.label,
        t.category,
        null,
        applyTagPreview,
      ]);
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      queryClient.invalidateQueries({ queryKey: ['taxonomy', 'tag', t.label] });
      queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: `Applied "${t.label}"`,
          message: currentPath.split(/[\\/]/).pop(),
          durationMs: 2000,
        },
      });
    } catch (err) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Tag failed',
          message: err instanceof Error ? err.message : 'Could not apply tag',
        },
      });
    }
  };

  // Prioritize curated tags (any category other than the catch-all "Suggested"
  // autotag bucket) — they're far more useful, so surface them first. Stable
  // sort preserves fuzzy-match relevance order within each group.
  const sortedTags = useMemo(
    () =>
      [...tagResults].sort(
        (a, b) =>
          (a.category === 'Suggested' ? 1 : 0) -
          (b.category === 'Suggested' ? 1 : 0)
      ),
    [tagResults]
  );

  // The tag rows shown in the palette (capped). They lead the navigable list.
  const cappedTags = useMemo(
    () => sortedTags.slice(0, PALETTE_TAG_CAP),
    [sortedTags]
  );

  // Ordered suggestion rows reported up by SuggestionSections.
  const [suggestionItems, setSuggestionItems] = useState<SuggestionItem[]>([]);

  // The full, ordered set the keyboard moves through: tags first (in render
  // order), then the suggestion rows.
  const navItems: SuggestionItem[] = useMemo(() => {
    if (!active) return [];
    const tagItems: SuggestionItem[] = cappedTags.map((t) => ({
      key: `tag:${t.label}`,
      predicate: { type: 'tag', value: t.label, exclude: false },
    }));
    return [...tagItems, ...suggestionItems];
  }, [active, cappedTags, suggestionItems]);

  useEffect(() => {
    onNavItemsChange(navItems);
  }, [navItems, onNavItemsChange]);

  // The parent's highlight must not outlive this subtree — it unmounts as soon
  // as the palette closes, and a stale count would keep the input in
  // result-navigation mode.
  useEffect(
    () => () => {
      onNavItemsChange([]);
    },
    [onNavItemsChange]
  );

  // Warming only (nothing typed, or "search by meaning" is on): the hooks above
  // still run — priming the categories cache and the shared tag index — but
  // there is nothing to draw.
  if (!active) return null;

  return (
    <div className="commandPaletteSearchResults">
      {cappedTags.length > 0 && (
        <div className="suggestion-section">
          <div className="suggestion-section-label">Tags</div>
          {cappedTags.map((t) => {
            const key = `tag:${t.label}`;
            return (
              <div
                key={t.label}
                className={`suggestion-row${
                  highlightedKey === key ? ' highlighted' : ''
                }`}
                title={displayTagLabel(t.label)}
                onMouseEnter={() => onHighlightKey(key)}
                onClick={() =>
                  onCommit({
                    type: 'tag',
                    value: t.label,
                    exclude: false,
                  })
                }
              >
                <span className="suggestion-prefix">#</span>
                <span className="suggestion-value">
                  {displayTagLabel(t.label)}
                </span>
                {t.category && t.category !== 'Suggested' && (
                  <span className="suggestion-meta">{t.category}</span>
                )}
                {currentPath && (
                  <button
                    type="button"
                    className="suggestion-apply"
                    title={`Apply "${t.label}" to the current item`}
                    onClick={(e) => {
                      e.stopPropagation();
                      applyTagToCurrent(t);
                    }}
                  >
                    <TagPlusIcon />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      <SuggestionSections
        text={text}
        categories={categories ?? []}
        onAdd={(predicate) => onCommit(predicate)}
        onItemsChange={setSuggestionItems}
        highlightedKey={highlightedKey}
        onHighlightKey={onHighlightKey}
      />
    </div>
  );
}
