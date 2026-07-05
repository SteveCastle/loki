import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { debounce } from 'lodash';
import { invoke } from '../../platform';
import type { Predicate } from '../../query/types';
import { SEARCH_DEBOUNCE_MS } from '../../search/search-config';
import './taxonomy.css';

interface CategoryLite {
  label: string;
}

// One navigable suggestion row: a stable key (unique across the whole results
// surface) plus the predicate it commits when chosen.
export interface SuggestionItem {
  key: string;
  predicate: Predicate;
}

interface SuggestionSectionsProps {
  text: string; // active search term (non-empty when shown)
  categories: CategoryLite[];
  onAdd: (predicate: Predicate) => void;
  // Optional keyboard-navigation hooks (command palette). When omitted (e.g.
  // the taxonomy sidebar) the section is click-only and behaves as before.
  onItemsChange?: (items: SuggestionItem[]) => void; // report ordered rows
  highlightedKey?: string | null; // row to render as highlighted
  onHighlightKey?: (key: string) => void; // hover → move highlight here
}

const SECTION_CAP = 8;

// Distinct-media counts stop counting here. Counting DISTINCT media_path is a
// per-row table walk in SQLite, and the huge autotag "Suggested" bucket
// (hundreds of thousands of items) took 20+ seconds per keystroke, stalling
// every other query on the shared connection. Every curated category is well
// under this cap, so their counts stay exact; capped ones render as "25000+".
const CATEGORY_COUNT_CAP = 25000;

// Lazy media count for a single rendered category row. Only mounted for
// categories that are actually shown, so we never fetch counts for the
// filtered-out / capped categories. staleTime: Infinity — the count is
// decoration, so one (possibly ~0.5s) query per category per session.
function CategoryCount({ label }: { label: string }) {
  const { data: count } = useQuery<number, Error>(
    ['suggest', 'category-count', label],
    () =>
      invoke('get-category-count', [label, CATEGORY_COUNT_CAP]) as Promise<number>,
    { refetchOnWindowFocus: false, staleTime: Infinity }
  );
  if (count === undefined) return null;
  return (
    <span className="suggestion-meta">
      {count >= CATEGORY_COUNT_CAP ? `${CATEGORY_COUNT_CAP}+` : count}
    </span>
  );
}

export default function SuggestionSections({
  text,
  categories,
  onAdd,
  onItemsChange,
  highlightedKey,
  onHighlightKey,
}: SuggestionSectionsProps) {
  // Debounce the term that drives the category matching (and with it the
  // per-category count IPCs mounted below) so they don't fire on every
  // keystroke — the command palette passes the raw, un-debounced text. The
  // "contains X" add-rows keep using the live `text` so the echoed term stays
  // instant.
  const [debouncedText, setDebouncedText] = useState<string>(text);
  const debouncedSet = useRef(
    debounce((value: string) => setDebouncedText(value), SEARCH_DEBOUNCE_MS)
  );
  useEffect(() => {
    debouncedSet.current(text);
  }, [text]);
  useEffect(() => {
    const d = debouncedSet.current;
    return () => d.cancel();
  }, []);

  const term = debouncedText.toLowerCase();

  // 1. Categories — substring match on label, capped. Driven by the debounced
  // term so the per-category count IPC (CategoryCount) only fires once typing
  // settles.
  const matchedCategories = categories
    .filter((c) => c.label.toLowerCase().includes(term))
    .slice(0, SECTION_CAP);

  // Row keys for highlight matching + navigation. These also prefix the rows
  // below so the rendered DOM order matches the reported item order exactly.
  const CAT_KEY = (label: string) => `cat:${label}`;
  const PATH_ADD_KEY = 'path:add';
  const DESC_ADD_KEY = 'desc:add';
  const HASH_ADD_KEY = 'hash:add';

  // Ordered, navigable items — must mirror the render order below so arrow-key
  // navigation lines up with what the user sees.
  const items: SuggestionItem[] = [
    ...matchedCategories.map((c) => ({
      key: CAT_KEY(c.label),
      predicate: { type: 'category', value: c.label, exclude: false } as Predicate,
    })),
    {
      key: PATH_ADD_KEY,
      predicate: { type: 'path', value: text, exclude: false } as Predicate,
    },
    {
      key: DESC_ADD_KEY,
      predicate: { type: 'description', value: text, exclude: false } as Predicate,
    },
    {
      key: HASH_ADD_KEY,
      predicate: { type: 'hash', value: text, exclude: false } as Predicate,
    },
  ];

  // Report the ordered items to the parent. Keyed on the (key + value)
  // signature so it only fires when the set actually changes — a fresh `items`
  // array every render would otherwise loop with the parent's setState. The
  // callback is read through a ref so its identity changing doesn't re-run it.
  const onItemsChangeRef = useRef(onItemsChange);
  onItemsChangeRef.current = onItemsChange;
  const itemsSignature = items
    .map((i) => `${i.key}=${i.predicate.value}`)
    .join('|');
  useEffect(() => {
    onItemsChangeRef.current?.(items);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemsSignature]);

  const rowClass = (key: string) =>
    `suggestion-row${highlightedKey === key ? ' highlighted' : ''}`;

  return (
    <div className="suggestion-sections">
      {matchedCategories.length > 0 && (
        <div className="suggestion-section">
          <div className="suggestion-section-label">Categories</div>
          {matchedCategories.map((c) => (
            <div
              key={c.label}
              className={`${rowClass(CAT_KEY(c.label))} suggestion-row-category`}
              onMouseEnter={() => onHighlightKey?.(CAT_KEY(c.label))}
              onClick={() =>
                onAdd({ type: 'category', value: c.label, exclude: false })
              }
            >
              <span className="suggestion-prefix">in:</span>
              <span className="suggestion-value">{c.label}</span>
              <CategoryCount label={c.label} />
            </div>
          ))}
        </div>
      )}

      <div className="suggestion-section">
        <div className="suggestion-section-label">Paths</div>
        <div
          className={`${rowClass(PATH_ADD_KEY)} suggestion-add-row`}
          onMouseEnter={() => onHighlightKey?.(PATH_ADD_KEY)}
          onClick={() => onAdd({ type: 'path', value: text, exclude: false })}
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            path contains &quot;{text}&quot;
          </span>
        </div>
      </div>

      <div className="suggestion-section">
        <div className="suggestion-section-label">Description</div>
        <div
          className={`${rowClass(DESC_ADD_KEY)} suggestion-add-row`}
          onMouseEnter={() => onHighlightKey?.(DESC_ADD_KEY)}
          onClick={() =>
            onAdd({ type: 'description', value: text, exclude: false })
          }
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            description contains &quot;{text}&quot;
          </span>
        </div>
      </div>

      <div className="suggestion-section">
        <div className="suggestion-section-label">Hash</div>
        <div
          className={`${rowClass(HASH_ADD_KEY)} suggestion-add-row`}
          onMouseEnter={() => onHighlightKey?.(HASH_ADD_KEY)}
          onClick={() => onAdd({ type: 'hash', value: text, exclude: false })}
        >
          <span className="suggestion-add-badge">+</span>
          <span className="suggestion-value">
            hash contains &quot;{text}&quot;
          </span>
        </div>
      </div>
    </div>
  );
}
