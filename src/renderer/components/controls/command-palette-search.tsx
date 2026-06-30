// CommandPaletteSearch — the unified query surface for the command palette.
//
// Replaces the old ListContextDisplay tag/search pill row. Renders the chip
// QueryInput plus an in-place type-ahead results surface (Tags + Categories /
// Paths / Description / Hash). Selecting a result adds a predicate to the SAME
// query state the taxonomy sidebar drives, so the palette and sidebar stay in
// lockstep. Kept compact + scrollable to fit the floating palette.
import { useEffect, useMemo, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { Predicate } from '../../query/types';
import { getNextFilterMode } from '../../../settings';
import { invoke } from '../../platform';
import { useTagSearch } from '../../hooks/useTagSearch';
import type { TagConcept } from '../../hooks/useTagSearch';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import QueryInput from '../query-input/QueryInput';
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

interface CommandPaletteSearchProps {
  // InterpreterFrom<typeof libraryMachine>; typed loosely to match the rest of
  // command-palette.tsx which threads libraryService as `any`.
  libraryService: any;
  // The media item under the cursor; the apply-tag button assigns to its path.
  currentItem?: { path?: string } | null;
}

export default function CommandPaletteSearch({
  libraryService,
  currentItem,
}: CommandPaletteSearchProps) {
  const queryClient = useQueryClient();
  const query = useSelector(libraryService, (s: any) => s.context.query);
  const filteringMode = useSelector(
    libraryService,
    (s: any) => s.context.settings.filteringMode
  );
  const applyTagPreview = useSelector(
    libraryService,
    (s: any) => s.context.settings.applyTagPreview
  );
  const initSessionId = useSelector(
    libraryService,
    (s: any) => s.context.initSessionId
  );

  const currentPath = currentItem?.path;

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

  const { addSearch } = useSearchHistory();

  const [text, setText] = useState('');
  // "Search by meaning" mode: typed text commits as a visual: predicate and the
  // tag-suggestion surface is suppressed (it's irrelevant to semantic search).
  const [meaningMode, setMeaningMode] = useState(false);
  // Index into `navItems` of the currently highlighted result. Enter commits
  // it; arrow keys move it; the top result is highlighted by default.
  const [highlightIndex, setHighlightIndex] = useState(0);
  // Ordered suggestion rows reported up by SuggestionSections, so the highlight
  // can span tags *and* suggestions as one list.
  const [suggestionItems, setSuggestionItems] = useState<SuggestionItem[]>([]);

  const { results: tagResults } = useTagSearch(text, text.length > 0);

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

  const { data: categories } = useQuery<CategoryLite[], Error>(
    ['taxonomy', 'categories', initSessionId],
    loadCategories,
    { staleTime: Infinity }
  );

  const join: 'AND' | 'OR' = filteringMode === 'OR' ? 'OR' : 'AND';

  const hasText = text.length > 0;

  const clearText = () => setText('');

  // The tag rows shown in the palette (capped). They lead the navigable list.
  const cappedTags = useMemo(
    () => sortedTags.slice(0, PALETTE_TAG_CAP),
    [sortedTags]
  );

  // The full, ordered set the keyboard moves through: tags first (in render
  // order), then the suggestion rows. Empty unless the user is searching.
  const navItems: SuggestionItem[] = useMemo(() => {
    if (!hasText || meaningMode) return [];
    const tagItems: SuggestionItem[] = cappedTags.map((t) => ({
      key: `tag:${t.label}`,
      predicate: { type: 'tag', value: t.label, exclude: false },
    }));
    return [...tagItems, ...suggestionItems];
  }, [hasText, cappedTags, suggestionItems, meaningMode]);

  // Clamp the stored index to the live list — the result count changes as the
  // user types and as async suggestions arrive.
  const safeIndex =
    navItems.length === 0
      ? -1
      : Math.min(Math.max(highlightIndex, 0), navItems.length - 1);
  const highlightedKey = safeIndex >= 0 ? navItems[safeIndex].key : null;

  // Snap the highlight back to the top result whenever the query text changes.
  useEffect(() => {
    setHighlightIndex(0);
  }, [text]);

  // Commit a chosen result: add it to the query AND record it in recent
  // searches (the user selected it — that's the search worth remembering),
  // then clear the typed text.
  const commitPredicate = (predicate: Predicate) => {
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: { predicate: { ...predicate, join } },
    });
    addSearch(predicate.value);
    clearText();
    setHighlightIndex(0);
  };

  const moveHighlight = (delta: 1 | -1) => {
    if (navItems.length === 0) return;
    setHighlightIndex((prev) => {
      const base = prev < 0 ? 0 : prev;
      return (base + delta + navItems.length) % navItems.length;
    });
  };

  const highlightByKey = (key: string) => {
    const idx = navItems.findIndex((n) => n.key === key);
    if (idx >= 0) setHighlightIndex(idx);
  };

  return (
    <div className="commandPaletteSearch">
      <QueryInput
        autoFocus
        query={query}
        textValue={text}
        onTextChange={setText}
        filteringMode={filteringMode}
        onCycleFilterMode={() =>
          libraryService.send({
            type: 'CHANGE_SETTING',
            data: { filteringMode: getNextFilterMode(filteringMode) },
          })
        }
        onSubmitText={() => {
          // Fallback for the brief window before suggestions populate (when
          // resultNavCount is still 0): commit the top tag if there is one.
          const top = sortedTags[0];
          if (top) {
            commitPredicate({ type: 'tag', value: top.label, exclude: false });
          }
        }}
        onRemovePredicate={(key) =>
          libraryService.send({ type: 'REMOVE_PREDICATE', data: { key } })
        }
        onToggleExclude={(key) =>
          libraryService.send({ type: 'TOGGLE_EXCLUDE', data: { key } })
        }
        onSetPredicateJoin={(key, j) =>
          libraryService.send({
            type: 'SET_PREDICATE_JOIN',
            data: { key, join: j },
          })
        }
        onClearText={clearText}
        onClearAll={() => {
          libraryService.send({ type: 'CLEAR_QUERY' });
          clearText();
        }}
        onMeaningModeChange={setMeaningMode}
        onSubmitVisual={(t) => {
          libraryService.send({
            type: 'ADD_PREDICATE',
            data: {
              predicate: { type: 'visual', value: t, exclude: false, join },
            },
          });
          clearText();
          setHighlightIndex(0);
        }}
        resultNavCount={navItems.length}
        onResultNavMove={moveHighlight}
        onResultNavSubmit={() => {
          if (safeIndex >= 0) commitPredicate(navItems[safeIndex].predicate);
        }}
      />

      {meaningMode && (
        <div className="commandPaletteMeaningHint">
          {hasText ? (
            <>
              Press <kbd>↵</kbd> to search images by meaning:{' '}
              <span className="meaning-hint-query">“{text.trim()}”</span>
            </>
          ) : (
            <>✨ Search by meaning — describe what an image looks like, then press <kbd>↵</kbd></>
          )}
        </div>
      )}

      {hasText && !meaningMode && (
        <div className="commandPaletteSearchResults">
          {cappedTags.length > 0 && (
            <div className="suggestion-section">
              <div className="suggestion-section-label">Tags</div>
              {cappedTags.map((t, i) => (
                <div
                  key={t.label}
                  className={`suggestion-row${
                    safeIndex === i ? ' highlighted' : ''
                  }`}
                  title={t.label}
                  onMouseEnter={() => setHighlightIndex(i)}
                  onClick={() =>
                    commitPredicate({
                      type: 'tag',
                      value: t.label,
                      exclude: false,
                    })
                  }
                >
                  <span className="suggestion-prefix">#</span>
                  <span className="suggestion-value">{t.label}</span>
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
              ))}
            </div>
          )}

          <SuggestionSections
            text={text}
            categories={categories ?? []}
            onAdd={(predicate) => commitPredicate(predicate)}
            onItemsChange={setSuggestionItems}
            highlightedKey={highlightedKey}
            onHighlightKey={highlightByKey}
          />
        </div>
      )}
    </div>
  );
}
