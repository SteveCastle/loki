// CommandPaletteSearch — the unified query surface for the command palette.
//
// Replaces the old ListContextDisplay tag/search pill row. Renders the chip
// QueryInput plus an in-place type-ahead results surface (Tags + Categories /
// Paths / Description / Hash). Selecting a result adds a predicate to the SAME
// query state the taxonomy sidebar drives, so the palette and sidebar stay in
// lockstep. Kept compact + scrollable to fit the floating palette.
import { useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import type { Predicate } from '../../query/types';
import { invoke } from '../../platform';
import { useTagSearch } from '../../hooks/useTagSearch';
import QueryInput from '../query-input/QueryInput';
import SuggestionSections from '../taxonomy/suggestion-sections';

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
}

export default function CommandPaletteSearch({
  libraryService,
}: CommandPaletteSearchProps) {
  const query = useSelector(libraryService, (s: any) => s.context.query);
  const filteringMode = useSelector(
    libraryService,
    (s: any) => s.context.settings.filteringMode
  );
  const initSessionId = useSelector(
    libraryService,
    (s: any) => s.context.initSessionId
  );

  const [text, setText] = useState('');

  const { results: tagResults } = useTagSearch(text, text.length > 0);

  const { data: categories } = useQuery<CategoryLite[], Error>(
    ['taxonomy', 'categories', initSessionId],
    loadCategories,
    { staleTime: Infinity }
  );

  const join: 'AND' | 'OR' = filteringMode === 'OR' ? 'OR' : 'AND';

  const clearText = () => setText('');

  const addPredicate = (predicate: Predicate) => {
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: { predicate: { ...predicate, join } },
    });
  };

  const addTag = (label: string) => {
    addPredicate({ type: 'tag', value: label, exclude: false });
    clearText();
  };

  const hasText = text.length > 0;

  return (
    <div className="commandPaletteSearch">
      <QueryInput
        autoFocus
        query={query}
        textValue={text}
        onTextChange={setText}
        onSubmitText={() => {
          // Commit the top tag suggestion as a predicate, then clear text.
          const top = tagResults[0];
          if (top) {
            addTag(top.label);
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
        onClearAll={() => {
          libraryService.send({ type: 'CLEAR_QUERY' });
          clearText();
        }}
      />

      {hasText && (
        <div className="commandPaletteSearchResults">
          {tagResults.length > 0 && (
            <div className="suggestion-section">
              <div className="suggestion-section-label">Tags</div>
              {tagResults.slice(0, PALETTE_TAG_CAP).map((t) => (
                <div
                  key={t.label}
                  className="suggestion-row"
                  onClick={() => addTag(t.label)}
                >
                  <span className="suggestion-prefix">#</span>
                  <span className="suggestion-value">{t.label}</span>
                </div>
              ))}
            </div>
          )}

          <SuggestionSections
            text={text}
            categories={categories ?? []}
            onAdd={(predicate) => addPredicate(predicate)}
          />
        </div>
      )}
    </div>
  );
}
