// CommandPaletteSearch — the unified query surface for the command palette.
//
// Replaces the old ListContextDisplay tag/search pill row. Renders the chip
// QueryInput plus an in-place type-ahead results surface (Tags + Categories /
// Paths / Description / Hash). Selecting a result adds a predicate to the SAME
// query state the taxonomy sidebar drives, so the palette and sidebar stay in
// lockstep. Kept compact + scrollable to fit the floating palette.
//
// OPEN-LATENCY CONTRACT: the palette is the most frequently opened surface in
// the app, so this component's FIRST commit must stay cheap — chips + input and
// nothing else. Every data-bound piece (the shared tag index, the categories
// fetch, the suggestion sections and their per-category counts) lives in
// CommandPaletteResults, which mounts only once the palette has painted, or on
// the first keystroke if the user types before that. Nothing here may await, or
// subscribe to, library-sized data on mount.
import { useCallback, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import type { Predicate } from '../../query/types';
import type { QueryHistoryEntry } from '../../query/history';
import { getNextFilterMode } from '../../../settings';
import { useFilterHistory } from '../../hooks/useFilterHistory';
import { useMeaningMode } from '../../hooks/useMeaningMode';
import useVisualSearchAvailable from '../../hooks/useVisualSearchAvailable';
import { useAfterPaint } from '../../hooks/useAfterPaint';
import QueryInput from '../query-input/QueryInput';
import CommandPaletteResults from './command-palette-results';
import type { SuggestionItem } from '../taxonomy/suggestion-sections';
import type { TagScope } from '../../search/tag-scopes';
import { markPalette } from '../../palette-trace';

interface CommandPaletteSearchProps {
  // InterpreterFrom<typeof libraryMachine>; typed loosely to match the rest of
  // command-palette.tsx which threads libraryService as `any`.
  libraryService: any;
  // The media item under the cursor; the apply-tag button assigns to its path.
  currentItem?: { path?: string } | null;
  // Which slice of the tag table the type-ahead searches. Defaults to
  // 'curated': the palette skips the autotagger's "Suggested" bucket, which is
  // ~183K of ~189K tags. Loading and Fuse-indexing those was what made this
  // surface laggy, and the palette never surfaced them prominently anyway (it
  // already sorted them last). The taxonomy sidebar passes nothing and gets the
  // complete set. See ../../search/tag-scopes.
  tagScope?: TagScope;
}

export default function CommandPaletteSearch({
  libraryService,
  currentItem,
  tagScope = 'curated',
}: CommandPaletteSearchProps) {
  const query = useSelector(libraryService, (s: any) => s.context.query);
  const filteringMode = useSelector(
    libraryService,
    (s: any) => s.context.settings.filteringMode
  );
  const authToken = useSelector(
    libraryService,
    (s: any) => s.context.authToken
  );

  const currentPath = currentItem?.path;

  // Session filter-state history + base row, rendered as a section of the
  // results surface below (QueryHistorySection inside CommandPaletteResults).
  const filterHistory = useFilterHistory(libraryService);

  const [text, setText] = useState('');
  // "Search by meaning" mode: typed text commits as a visual: predicate and the
  // tag-suggestion surface is suppressed (it's irrelevant to semantic search).
  // Shared + sticky (useMeaningMode) — survives the palette closing/reopening.
  const { meaningMode } = useMeaningMode();
  // Vector search needs a reachable, logged-in media server (Electron); the
  // ✨ toggle is hidden and meaning mode force-disabled without one.
  const visualSearchAvailable = useVisualSearchAvailable(authToken);
  // Index into `navItems` of the currently highlighted result. Enter commits
  // it; arrow keys move it; the top result is highlighted by default.
  const [highlightIndex, setHighlightIndex] = useState(0);
  // Ordered rows (tags then suggestions) reported up by the results engine, so
  // the highlight can span both groups as one list.
  const [navItems, setNavItems] = useState<SuggestionItem[]>([]);

  const join: 'AND' | 'OR' = filteringMode === 'OR' ? 'OR' : 'AND';

  const hasText = text.length > 0;
  const resultsActive = hasText && !meaningMode;

  // Keep the search engine out of the commit that opens the palette: it mounts
  // in the task after the shell paints, or immediately if the user has already
  // started typing (a keystroke can't arrive before that first paint in
  // practice, but the palette must never swallow one if it does).
  const afterPaint = useAfterPaint();
  const engineMounted = afterPaint || resultsActive;
  if (engineMounted) markPalette('engine-mounted');

  const clearText = () => setText('');

  // Clamp the stored index to the live list — the result count changes as the
  // user types and as async suggestions arrive. A negative index means "no
  // highlight": the resting state while nothing is typed, so opening the
  // palette doesn't arm Enter on a history row the user never chose.
  const safeIndex =
    navItems.length === 0 || highlightIndex < 0
      ? -1
      : Math.min(highlightIndex, navItems.length - 1);
  const highlightedKey = safeIndex >= 0 ? navItems[safeIndex].key : null;

  // Snap the highlight back to the top result whenever the query text changes;
  // with no text there is no top result to arm (only the history section).
  useEffect(() => {
    setHighlightIndex(text ? 0 : -1);
  }, [text]);

  // Commit a chosen result: add it to the query, then clear the typed text.
  // (The resulting filter STATE is recorded by the machine once it loads —
  // see queryHistory in state.tsx.)
  const commitPredicate = useCallback(
    (predicate: Predicate) => {
      libraryService.send({
        type: 'ADD_PREDICATE',
        data: { predicate: { ...predicate, join } },
      });
      setText('');
      setHighlightIndex(0);
    },
    [libraryService, join]
  );

  // Commit whichever navigable row is chosen: history rows carry an action
  // (re-apply that filter state), every other row carries a predicate.
  const commitItem = useCallback(
    (item: SuggestionItem) => {
      if (item.action) item.action();
      else if (item.predicate) commitPredicate(item.predicate);
    },
    [commitPredicate]
  );

  // Applying a history entry (or the base row) replaces the whole query, so
  // any typed text is stale — clear it alongside.
  const applyHistoryEntry = useCallback(
    (entry: QueryHistoryEntry) => {
      filterHistory.onApplyHistory(entry);
      setText('');
    },
    [filterHistory.onApplyHistory]
  );

  const applyBase = useCallback(() => {
    filterHistory.onApplyBase();
    setText('');
  }, [filterHistory.onApplyBase]);

  const moveHighlight = (delta: 1 | -1) => {
    if (navItems.length === 0) return;
    setHighlightIndex((prev) => {
      // From the no-highlight resting state, ArrowDown enters at the top and
      // ArrowUp at the bottom.
      if (prev < 0) return delta === 1 ? 0 : navItems.length - 1;
      return (prev + delta + navItems.length) % navItems.length;
    });
  };

  const highlightByKey = useCallback(
    (key: string) => {
      const idx = navItems.findIndex((n) => n.key === key);
      if (idx >= 0) setHighlightIndex(idx);
    },
    [navItems]
  );

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
          // Fallback for the brief window before results populate (when
          // resultNavCount is still 0): commit the top row if there is one.
          const top = navItems[0];
          if (top) commitItem(top);
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
        onUpdatePredicateBlend={(key, patch) =>
          libraryService.send({
            type: 'UPDATE_PREDICATE_BLEND',
            data: { key, patch },
          })
        }
        onAddBlendNode={(key, node) =>
          libraryService.send({ type: 'ADD_BLEND_NODE', data: { key, node } })
        }
        onRemoveBlendNode={(key, index) =>
          libraryService.send({
            type: 'REMOVE_BLEND_NODE',
            data: { key, index },
          })
        }
        onUpdateBlendNode={(key, index, patch) =>
          libraryService.send({
            type: 'UPDATE_BLEND_NODE',
            data: { key, index, patch },
          })
        }
        onSetBlendMode={(key, mode) =>
          libraryService.send({
            type: 'SET_BLEND_MODE',
            data: { key, mode },
          })
        }
        onClearText={clearText}
        onClearAll={() => {
          libraryService.send({ type: 'CLEAR_QUERY' });
          clearText();
        }}
        onSubmitVisual={
          visualSearchAvailable
            ? (t) => {
                libraryService.send({
                  type: 'ADD_PREDICATE',
                  data: {
                    predicate: {
                      type: 'visual',
                      value: t,
                      exclude: false,
                      join,
                    },
                  },
                });
                clearText();
                setHighlightIndex(0);
              }
            : undefined
        }
        resultNavCount={navItems.length}
        onResultNavMove={moveHighlight}
        onResultNavSubmit={() => {
          if (safeIndex >= 0) commitItem(navItems[safeIndex]);
        }}
      />

      {meaningMode && visualSearchAvailable && (
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

      {engineMounted && (
        <CommandPaletteResults
          text={text}
          active={resultsActive}
          libraryService={libraryService}
          currentPath={currentPath}
          highlightedKey={highlightedKey}
          onHighlightKey={highlightByKey}
          onNavItemsChange={setNavItems}
          onCommit={commitPredicate}
          tagScope={tagScope}
          historyActive={!meaningMode}
          query={query}
          history={filterHistory.history}
          onApplyHistory={applyHistoryEntry}
          baseLabel={filterHistory.baseLabel}
          baseCount={filterHistory.baseCount}
          onApplyBase={applyBase}
        />
      )}
    </div>
  );
}
