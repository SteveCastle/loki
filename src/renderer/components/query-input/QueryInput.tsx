import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import type { Query, Predicate } from '../../query/types';
import { predicateKey } from '../../query/types';
import clear from '../../../../assets/cancel.svg';
import './query-input.css';

interface QueryInputProps {
  query: Query;
  textValue: string;
  onTextChange: (value: string) => void;
  onSubmitText: () => void; // Enter pressed with text present (taxonomy decides what to commit)
  onRemovePredicate: (key: string) => void;
  onToggleExclude: (key: string) => void;
  onSetPredicateJoin: (key: string, join: 'AND' | 'OR') => void;
  onClearAll: () => void; // clear chips + text
  onFocus?: () => void;
  disabled?: boolean;
}

const CHEAT_SHEET = [
  { syntax: '"quoted phrase"', desc: 'Exact match' },
  { syntax: 'tag:name', desc: 'Search tags' },
  { syntax: 'in:category', desc: 'Filter by category' },
  { syntax: 'path:dir', desc: 'Search paths' },
  { syntax: 'description:txt', desc: 'Search descriptions' },
  { syntax: 'hash:abc', desc: 'Search by hash' },
  { syntax: '-term', desc: 'Exclude term' },
];

// Glyph prefix shown on a chip for each predicate type.
const TYPE_GLYPH: Record<Predicate['type'], string> = {
  tag: '#',
  category: 'in:',
  path: 'path:',
  description: 'description:',
  hash: 'hash:',
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
  onFocus,
  disabled = false,
}: QueryInputProps) {
  const { history, addSearch, removeSearch, clearAll } = useSearchHistory();
  const [isOpen, setIsOpen] = useState(false);
  const [showCheatSheet, setShowCheatSheet] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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
    setIsOpen(true);
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
    ]
  );

  const handleClear = useCallback(() => {
    onClearAll();
    setHighlightIndex(-1);
  }, [onClearAll]);

  const dropdownOpen = isOpen && hasItems;

  return (
    <div className="query-input" ref={containerRef}>
      {query.predicates.length > 0 && (
        <div className="query-chips">
          {query.predicates.map((p) => {
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
          onChange={(e) => onTextChange(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          onKeyUp={(e) => e.stopPropagation()}
          onFocus={handleFocus}
          onBlur={handleBlur}
          disabled={disabled}
        />
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
