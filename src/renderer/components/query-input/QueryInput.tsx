import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import type { Query, Predicate } from '../../query/types';
import { predicateKey } from '../../query/types';
import type { FilterModeOption } from '../../../settings';
import clear from '../../../../assets/cancel.svg';
import union from '../../../../assets/union.svg';
import intersect from '../../../../assets/intersect.svg';
import selective from '../../../../assets/selective.svg';
import './query-input.css';

interface QueryInputProps {
  query: Query;
  textValue: string;
  onTextChange: (value: string) => void;
  onSubmitText: () => void; // Enter pressed with text present (taxonomy decides what to commit)
  onRemovePredicate: (key: string) => void;
  onToggleExclude: (key: string) => void;
  onSetPredicateJoin: (key: string, join: 'AND' | 'OR') => void;
  onClearAll: () => void; // clear chips + text (resets the library)
  onClearText: () => void; // clear only the typed text (no-op on the library)
  onFocus?: () => void;
  autoFocus?: boolean; // focus the text input on mount (fast palette workflow)
  disabled?: boolean;
  // Result-navigation bridge. When the parent renders its own results surface
  // (the command palette) and passes resultNavCount > 0, the input forwards
  // Arrow Up/Down and Enter to the parent so the user can keyboard-navigate the
  // highlighted result, and the internal history dropdown is suppressed (the
  // results surface is what the user is navigating). Omitted everywhere else
  // (e.g. the taxonomy sidebar), which keeps the original history behaviour.
  resultNavCount?: number;
  onResultNavMove?: (delta: 1 | -1) => void;
  onResultNavSubmit?: () => void;
  // Tag-filtering behaviour toggle. When both are supplied, an icon button is
  // rendered in the input that cycles Intersection → Union → Exclusive (the
  // same `filteringMode` setting the taxonomy sidebar toggles). Omitted →
  // the toggle is not rendered, so callers that don't drive the setting are
  // unaffected.
  filteringMode?: FilterModeOption;
  onCycleFilterMode?: () => void;
}

const CHEAT_SHEET = [
  { syntax: '"quoted phrase"', desc: 'Exact match' },
  { syntax: 'tag:name', desc: 'Search tags' },
  { syntax: 'in:category', desc: 'Filter by category' },
  { syntax: 'path:dir', desc: 'Search paths' },
  { syntax: 'description:txt', desc: 'Search descriptions' },
  { syntax: 'hash:abc', desc: 'Search by hash' },
  { syntax: 'visual:"red car"', desc: 'Visual search by text', visualOnly: true },
  { syntax: 'similar:path', desc: 'Find visually similar media', visualOnly: true },
  { syntax: '-term', desc: 'Exclude term' },
];

// Glyph prefix shown on a chip for each predicate type.
const TYPE_GLYPH: Record<Predicate['type'], string> = {
  tag: '#',
  category: 'in:',
  path: 'path:',
  description: 'description:',
  hash: 'hash:',
  similar: 'similar:',
  visual: 'visual:',
};

// Icons + labels for the three tag-filtering behaviours, mirroring the toggle
// in the taxonomy sidebar (taxonomy.tsx). The same `filteringMode` setting
// drives both, so the icon shown here always matches the sidebar toggle.
const FILTER_MODE_ICONS: Record<FilterModeOption, string> = {
  OR: union,
  AND: intersect,
  EXCLUSIVE: selective,
};

const FILTER_MODE_LABELS: Record<FilterModeOption, string> = {
  OR: 'Union',
  AND: 'Intersection',
  EXCLUSIVE: 'Exclusive',
};

const MAX_VISIBLE_RECENT = 5;
const MAX_VISIBLE_FILTERED = 10;

export default function QueryInput({
  query,
  textValue,
  onTextChange,
  onSubmitText,
  onRemovePredicate,
  onToggleExclude,
  onSetPredicateJoin,
  onClearAll,
  onClearText,
  onFocus,
  autoFocus = false,
  disabled = false,
  resultNavCount = 0,
  onResultNavMove,
  onResultNavSubmit,
  filteringMode,
  onCycleFilterMode,
}: QueryInputProps) {
  const { history, addSearch, removeSearch, clearAll } = useSearchHistory();
  // The parent owns a navigable results list (command palette). While it has
  // items, arrow/enter drive that list instead of the history dropdown.
  const resultNavActive = resultNavCount > 0;
  const [isOpen, setIsOpen] = useState(false);
  const [showCheatSheet, setShowCheatSheet] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Focus the text input on mount when requested (e.g. the command palette
  // opens) so the user can start typing a query immediately.
  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filteredHistory = textValue.trim()
    ? history
        .filter((item) =>
          item.toLowerCase().includes(textValue.trim().toLowerCase())
        )
        .slice(0, MAX_VISIBLE_FILTERED)
    : history.slice(0, MAX_VISIBLE_RECENT);

  const hasItems = showCheatSheet || filteredHistory.length > 0;

  // Reset highlight when input changes
  useEffect(() => {
    setHighlightIndex(-1);
  }, [textValue]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
        setShowCheatSheet(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleFocus = useCallback(() => {
    if (blurTimeoutRef.current) {
      clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }
    // Deliberately do NOT open the dropdown on focus. The command palette
    // programmatically focuses this input on mount (autoFocus), and opening on
    // focus made the history dropdown appear before the user interacted at all.
    // The dropdown opens only on genuine intent: typing, an intentional
    // mouse-down on the input, or ArrowDown (see below).
    onFocus?.();
  }, [onFocus]);

  const handleBlur = useCallback(() => {
    blurTimeoutRef.current = setTimeout(() => {
      setIsOpen(false);
      setShowCheatSheet(false);
    }, 200);
  }, []);

  // Select a search-history entry: push its text into the input, record it,
  // then commit it via the taxonomy-owned submit handler.
  const selectItem = useCallback(
    (item: string) => {
      onTextChange(item);
      addSearch(item);
      onSubmitText();
      setIsOpen(false);
      setShowCheatSheet(false);
      setHighlightIndex(-1);
    },
    [onTextChange, addSearch, onSubmitText]
  );

  const handleSubmit = useCallback(() => {
    const trimmed = textValue.trim();
    if (trimmed) {
      addSearch(trimmed);
      onSubmitText();
      setIsOpen(false);
      setShowCheatSheet(false);
      setHighlightIndex(-1);
    }
  }, [textValue, addSearch, onSubmitText]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      e.stopPropagation();

      // Result-navigation mode: the parent's results surface owns the
      // highlight, so arrow/enter drive it instead of the history dropdown.
      if (resultNavActive) {
        switch (e.key) {
          case 'ArrowDown':
            e.preventDefault();
            onResultNavMove?.(1);
            return;
          case 'ArrowUp':
            e.preventDefault();
            onResultNavMove?.(-1);
            return;
          case 'Enter':
            e.preventDefault();
            onResultNavSubmit?.();
            return;
          default:
            return; // typing and everything else: let the input handle it
        }
      }

      if (!isOpen || !hasItems || showCheatSheet) {
        if (e.key === 'Enter') {
          handleSubmit();
          return;
        }
        if (e.key === 'Escape') {
          setIsOpen(false);
          setShowCheatSheet(false);
          return;
        }
        // Keyboard affordance: open (and highlight the first row) on ArrowDown
        // so keyboard-only users can still reach history now that focus alone
        // no longer opens the dropdown.
        if (e.key === 'ArrowDown' && hasItems && !showCheatSheet) {
          e.preventDefault();
          setIsOpen(true);
          setHighlightIndex(0);
          return;
        }
        return;
      }

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setHighlightIndex((prev) =>
            prev < filteredHistory.length - 1 ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setHighlightIndex((prev) =>
            prev > 0 ? prev - 1 : filteredHistory.length - 1
          );
          break;
        case 'Enter':
          if (highlightIndex >= 0 && highlightIndex < filteredHistory.length) {
            selectItem(filteredHistory[highlightIndex]);
          } else {
            handleSubmit();
          }
          break;
        case 'Escape':
          setIsOpen(false);
          setShowCheatSheet(false);
          setHighlightIndex(-1);
          break;
        case 'Delete':
        case 'Backspace':
          if (
            !textValue &&
            highlightIndex >= 0 &&
            highlightIndex < filteredHistory.length
          ) {
            e.preventDefault();
            const itemToRemove = filteredHistory[highlightIndex];
            removeSearch(itemToRemove);
            setHighlightIndex((prev) =>
              prev >= filteredHistory.length - 1 ? prev - 1 : prev
            );
          }
          break;
      }
    },
    [
      isOpen,
      hasItems,
      showCheatSheet,
      highlightIndex,
      filteredHistory,
      textValue,
      handleSubmit,
      selectItem,
      removeSearch,
      resultNavActive,
      onResultNavMove,
      onResultNavSubmit,
    ]
  );

  const handleClear = useCallback(() => {
    // Clearing filters resets the library to its pre-filter state. Clearing
    // only typed text must NOT touch the library — so when there are no
    // predicates, clear the text alone.
    if (query.predicates.length > 0) {
      onClearAll();
    } else {
      onClearText();
    }
    setHighlightIndex(-1);
  }, [query.predicates.length, onClearAll, onClearText]);

  // Suppress the history dropdown while the parent's results surface is the
  // active navigation target, so the two don't overlap or fight for arrow keys.
  const dropdownOpen = !resultNavActive && isOpen && hasItems;

  return (
    <div className="query-input" ref={containerRef}>
      {query.predicates.length > 0 && (
        <div className="query-chips">
          {query.predicates.map((p, index) => {
            const key = predicateKey(p);
            const join = p.join ?? 'AND';
            const chipClass = `query-chip${p.exclude ? ' exclude' : ''}${
              p.type === 'category' ? ' category' : ''
            }`;
            return (
              <span
                className={chipClass}
                key={key}
                onClick={() => onToggleExclude(key)}
                title={
                  p.exclude ? 'Click to include' : 'Click to exclude'
                }
              >
                {index > 0 && (
                  <button
                    type="button"
                    className={`query-chip-join${
                      join === 'OR' ? ' query-chip-join--or' : ''
                    }`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSetPredicateJoin(key, join === 'AND' ? 'OR' : 'AND');
                    }}
                    title="Toggle AND/OR"
                  >
                    {join}
                  </button>
                )}
                <span className="query-chip-label">
                  {p.exclude ? '−' : ''}
                  {TYPE_GLYPH[p.type]}
                  {p.value}
                </span>
                <button
                  className="query-chip-remove"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemovePredicate(key);
                  }}
                  title="Remove"
                >
                  &times;
                </button>
              </span>
            );
          })}
        </div>
      )}
      <div className="query-input-field">
        <input
          ref={inputRef}
          type="text"
          placeholder="Search & filter"
          value={textValue}
          onChange={(e) => {
            // Typing is intent — open the dropdown so matching history shows.
            setIsOpen(true);
            onTextChange(e.currentTarget.value);
          }}
          onMouseDown={() => {
            // An intentional click into the input opens the dropdown (focus
            // alone no longer does, to keep the palette's autoFocus quiet).
            setIsOpen(true);
          }}
          onKeyDown={handleKeyDown}
          onKeyUp={(e) => e.stopPropagation()}
          onFocus={handleFocus}
          onBlur={handleBlur}
          disabled={disabled}
        />
        {filteringMode && onCycleFilterMode && (
          <button
            className="query-input-filter-mode"
            title={`Tag filtering: ${FILTER_MODE_LABELS[filteringMode]} (click to cycle Intersection / Union / Exclusive)`}
            onMouseDown={(e) => e.preventDefault()}
            onClick={onCycleFilterMode}
          >
            <img
              src={FILTER_MODE_ICONS[filteringMode]}
              alt={FILTER_MODE_LABELS[filteringMode]}
            />
          </button>
        )}
        <button
          className="query-input-help"
          title="Query syntax help"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            setShowCheatSheet((prev) => !prev);
            setIsOpen(true);
            inputRef.current?.focus();
          }}
        >
          ?
        </button>
        <button
          className="query-input-submit"
          onClick={handleSubmit}
          disabled={!textValue.trim() || disabled}
          title="Search"
        >
          &rarr;
        </button>
        <button className="query-input-clear" onClick={handleClear}>
          <img src={clear} />
        </button>
      </div>
      {dropdownOpen && (
        <div className="query-input-dropdown">
          {showCheatSheet ? (
            <div className="query-input-cheatsheet">
              <div className="query-input-section-header">Query Syntax</div>
              {CHEAT_SHEET.map((entry) => (
                <div className="query-input-cheatsheet-row" key={entry.syntax}>
                  <span className="query-input-cheatsheet-syntax">
                    {entry.syntax}
                  </span>
                  <span className="query-input-cheatsheet-desc">
                    {entry.desc}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="query-input-history">
              <div className="query-input-section-header">
                <span>
                  {textValue.trim() ? 'Search History' : 'Recent Searches'}
                </span>
                {history.length > 0 && (
                  <button
                    className="query-input-clear-all"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      clearAll();
                      setHighlightIndex(-1);
                    }}
                  >
                    Clear All
                  </button>
                )}
              </div>
              {filteredHistory.length === 0 ? (
                <div className="query-input-empty">No recent searches</div>
              ) : (
                filteredHistory.map((item, index) => (
                  <div
                    className={`query-input-item${index === highlightIndex ? ' highlighted' : ''}`}
                    key={item}
                    onMouseEnter={() => setHighlightIndex(index)}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => selectItem(item)}
                  >
                    <span className="query-input-item-text">{item}</span>
                    <button
                      className="query-input-item-remove"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={(e) => {
                        e.stopPropagation();
                        removeSearch(item);
                        setHighlightIndex(-1);
                      }}
                      title="Remove"
                    >
                      &times;
                    </button>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
