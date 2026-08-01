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
import { getNextFilterMode } from '../../../settings';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import { useMeaningMode } from '../../hooks/useMeaningMode';
import useVisualSearchAvailable from '../../hooks/useVisualSearchAvailable';
import { useAfterPaint } from '../../hooks/useAfterPaint';
import QueryInput from '../query-input/QueryInput';
import CommandPaletteResults from './command-palette-results';
import type { SuggestionItem } from '../taxonomy/suggestion-sections';

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

  const { addSearch } = useSearchHistory();

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

  const clearText = () => setText('');

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
  const commitPredicate = useCallback(
    (predicate: Predicate) => {
      libraryService.send({
        type: 'ADD_PREDICATE',
        data: { predicate: { ...predicate, join } },
      });
      addSearch(predicate.value);
      setText('');
      setHighlightIndex(0);
    },
    [libraryService, join, addSearch]
  );

  const moveHighlight = (delta: 1 | -1) => {
    if (navItems.length === 0) return;
    setHighlightIndex((prev) => {
      const base = prev < 0 ? 0 : prev;
      return (base + delta + navItems.length) % navItems.length;
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
          if (top) commitPredicate(top.predicate);
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
          if (safeIndex >= 0) commitPredicate(navItems[safeIndex].predicate);
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
        />
      )}
    </div>
  );
}
