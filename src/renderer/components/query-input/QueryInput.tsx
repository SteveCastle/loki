import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import clear from '../../../../assets/cancel.svg';
import './query-input.css';

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  onClear: () => void;
  disabled?: boolean;
}

const CHEAT_SHEET = [
  { syntax: '"quoted phrase"', desc: 'Exact match' },
  { syntax: 'tag:name', desc: 'Search tags' },
  { syntax: 'path:dir', desc: 'Search paths' },
  { syntax: 'description:txt', desc: 'Search descriptions' },
  { syntax: 'hash:abc', desc: 'Search by hash' },
  { syntax: '-term', desc: 'Exclude term' },
];

const MAX_VISIBLE_RECENT = 5;
const MAX_VISIBLE_FILTERED = 10;

export default function QueryInput({
  value,
  onChange,
  onSubmit,
  onClear,
  disabled = false,
}: QueryInputProps) {
  const { history, addSearch, removeSearch, clearAll } = useSearchHistory();
  const [isOpen, setIsOpen] = useState(false);
  const [showCheatSheet, setShowCheatSheet] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const filteredHistory = value.trim()
    ? history
        .filter((item) =>
          item.toLowerCase().includes(value.trim().toLowerCase())
        )
        .slice(0, MAX_VISIBLE_FILTERED)
    : history.slice(0, MAX_VISIBLE_RECENT);

  const hasItems = showCheatSheet || filteredHistory.length > 0;

  // Reset highlight when input changes
  useEffect(() => {
    setHighlightIndex(-1);
  }, [value]);

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
  }, []);

  const handleBlur = useCallback(() => {
    blurTimeoutRef.current = setTimeout(() => {
      setIsOpen(false);
      setShowCheatSheet(false);
    }, 200);
  }, []);

  const selectItem = useCallback(
    (query: string) => {
      onChange(query);
      addSearch(query);
      onSubmit(query);
      setIsOpen(false);
      setShowCheatSheet(false);
      setHighlightIndex(-1);
    },
    [onChange, addSearch, onSubmit]
  );

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed) {
      addSearch(trimmed);
      onSubmit(trimmed);
      setIsOpen(false);
      setShowCheatSheet(false);
      setHighlightIndex(-1);
    }
  }, [value, addSearch, onSubmit]);

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
            !value &&
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
      value,
      handleSubmit,
      selectItem,
      removeSearch,
    ]
  );

  const handleClear = useCallback(() => {
    onChange('');
    onClear();
    setHighlightIndex(-1);
  }, [onChange, onClear]);

  const dropdownOpen = isOpen && hasItems;

  return (
    <div className="query-input" ref={containerRef}>
      <div className="query-input-field">
        <input
          ref={inputRef}
          type="text"
          placeholder="Search Content"
          value={value}
          onChange={(e) => onChange(e.currentTarget.value)}
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
          disabled={!value.trim() || disabled}
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
                <span>{value.trim() ? 'Search History' : 'Recent Searches'}</span>
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
