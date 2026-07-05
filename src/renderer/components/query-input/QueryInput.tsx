import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchHistory } from '../../hooks/useSearchHistory';
import { useMeaningMode } from '../../hooks/useMeaningMode';
import { mediaUrl } from '../../platform';
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
  // Blend an image predicate (similar/clip) with text. When supplied, hovering
  // an image chip opens a popover with a text input and an image↔text weight
  // slider; commits patch the predicate in place (text: '' clears the blend).
  onUpdatePredicateBlend?: (
    key: string,
    patch: { text?: string; textWeight?: number }
  ) => void;
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
  // Semantic ("search by meaning") support. When onSubmitVisual is provided a
  // ✨ toggle is shown; with it ON, submitting commits the typed text as a
  // `visual:` (text→image embedding) predicate instead of the normal parse.
  // The toggle state itself is shared via useMeaningMode — parents that need
  // to react (e.g. hide tag suggestions) read the same hook.
  onSubmitVisual?: (text: string) => void;
}

// Glyph prefix shown on a chip for each predicate type.
const TYPE_GLYPH: Record<Predicate['type'], string> = {
  tag: '#',
  category: 'in:',
  path: 'path:',
  description: 'description:',
  hash: 'hash:',
  similar: 'similar:',
  visual: 'visual:',
  clip: 'clip:', // never shown — clip chips render a thumbnail instead
  face: 'face:', // never shown — face chips render a thumbnail instead
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
  onUpdatePredicateBlend,
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
  onSubmitVisual,
}: QueryInputProps) {
  const { history, addSearch, removeSearch, clearAll } = useSearchHistory();
  // "Search by meaning" mode: typed text commits as a visual: predicate.
  // Shared + sticky: stays on across palette open/close until toggled off.
  const { meaningMode, setMeaningMode } = useMeaningMode();
  const toggleMeaningMode = useCallback(() => {
    setMeaningMode(!meaningMode);
    inputRef.current?.focus();
  }, [meaningMode, setMeaningMode]);
  // The parent owns a navigable results list (command palette). While it has
  // items, arrow/enter drive that list instead of the history dropdown.
  const resultNavActive = resultNavCount > 0;
  const [isOpen, setIsOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const blurTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Blend popover (image chips only): which chip is open, plus draft text /
  // text-weight (%) being edited. Drafts commit on Enter, input blur, or
  // slider release — never per keystroke/drag-tick, since a commit re-runs the
  // query (an embed-subprocess call per run on the server).
  const [blendKey, setBlendKey] = useState<string | null>(null);
  const [blendText, setBlendText] = useState('');
  const [blendPct, setBlendPct] = useState(50);
  const blendCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Keep the popover open while its controls have focus or the pointer is
  // over the chip/popover (the two signals cover typing and mousing apart).
  const blendFocusRef = useRef(false);
  const blendHoverRef = useRef(false);

  const cancelBlendClose = useCallback(() => {
    if (blendCloseTimer.current) {
      clearTimeout(blendCloseTimer.current);
      blendCloseTimer.current = null;
    }
  }, []);

  // Commit a blend edit for the chip at `key`. Values are passed explicitly
  // (not read from state) so slider-release and input-blur commits can't race
  // a pending setState. No-ops when nothing actually changed.
  const commitBlend = useCallback(
    (key: string, text: string, pct: number) => {
      if (!onUpdatePredicateBlend) return;
      const p = query.predicates.find((x) => predicateKey(x) === key);
      if (!p) return;
      const trimmed = text.trim();
      const curText = p.text ?? '';
      const curPct = Math.round((p.textWeight ?? 0.5) * 100);
      if (trimmed === curText && (trimmed === '' || pct === curPct)) return;
      onUpdatePredicateBlend(
        key,
        trimmed
          ? { text: trimmed, textWeight: pct / 100 }
          : { text: '', textWeight: undefined }
      );
    },
    [onUpdatePredicateBlend, query.predicates]
  );

  const openBlend = useCallback(
    (p: Predicate) => {
      cancelBlendClose();
      blendHoverRef.current = true;
      const key = predicateKey(p);
      if (blendKey === key) return;
      // Flush any uncommitted draft on the previously open chip.
      if (blendKey !== null) commitBlend(blendKey, blendText, blendPct);
      setBlendKey(key);
      setBlendText(p.text ?? '');
      setBlendPct(Math.round((p.textWeight ?? 0.5) * 100));
    },
    [cancelBlendClose, blendKey, blendText, blendPct, commitBlend]
  );

  const scheduleBlendClose = useCallback(() => {
    blendHoverRef.current = false;
    cancelBlendClose();
    blendCloseTimer.current = setTimeout(() => {
      if (blendFocusRef.current || blendHoverRef.current) return;
      setBlendKey(null);
    }, 250);
  }, [cancelBlendClose]);

  // Clear any pending popover-close timer on unmount.
  useEffect(() => cancelBlendClose, [cancelBlendClose]);

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

  const hasItems = filteredHistory.length > 0;

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
      setHighlightIndex(-1);
    },
    [onTextChange, addSearch, onSubmitText]
  );

  const handleSubmit = useCallback(() => {
    const trimmed = textValue.trim();
    if (trimmed) {
      addSearch(trimmed);
      if (meaningMode && onSubmitVisual) {
        // Commit the raw text as a visual (text→image) embedding search.
        onSubmitVisual(trimmed);
      } else {
        onSubmitText();
      }
      setIsOpen(false);
      setHighlightIndex(-1);
    }
  }, [textValue, addSearch, onSubmitText, meaningMode, onSubmitVisual]);

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
            // In meaning mode, Enter commits the visual search rather than the
            // highlighted tag suggestion.
            if (meaningMode && onSubmitVisual) handleSubmit();
            else onResultNavSubmit?.();
            return;
          default:
            return; // typing and everything else: let the input handle it
        }
      }

      if (!isOpen || !hasItems) {
        if (e.key === 'Enter') {
          handleSubmit();
          return;
        }
        if (e.key === 'Escape') {
          setIsOpen(false);
          return;
        }
        // Keyboard affordance: open (and highlight the first row) on ArrowDown
        // so keyboard-only users can still reach history now that focus alone
        // no longer opens the dropdown.
        if (e.key === 'ArrowDown' && hasItems) {
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
      highlightIndex,
      filteredHistory,
      textValue,
      handleSubmit,
      selectItem,
      removeSearch,
      resultNavActive,
      onResultNavMove,
      onResultNavSubmit,
      meaningMode,
      onSubmitVisual,
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
            const isVisual = p.type === 'visual';
            const isSimilar = p.type === 'similar';
            // A captured screen region (PNG data URL) — renders like a similar:
            // chip but with the clip itself as the thumbnail.
            const isClip = p.type === 'clip';
            // Face-identity search: value is a media path or a captured
            // region data URL; renders like similar/clip with a person icon.
            const isFace = p.type === 'face';
            const isFaceClip = isFace && p.value.startsWith('data:');
            const chipClass = `query-chip${p.exclude ? ' exclude' : ''}${
              p.type === 'category' ? ' category' : ''
            }${isVisual ? ' visual' : ''}${
              isSimilar || isClip || isFace ? ' similar' : ''
            }`;
            const baseName =
              isSimilar || (isFace && !isFaceClip)
                ? p.value.split(/[/\\]/).pop() || p.value
                : '';
            // Image chips (similar/clip) can blend in a text query: hovering
            // opens a popover with a text input + image↔text weight slider.
            const isImage = isSimilar || isClip;
            const canBlend = isImage && !!onUpdatePredicateBlend;
            const blendOpen = canBlend && blendKey === key;
            const hasBlend = isImage && !!p.text;
            const chipPct = Math.round((p.textWeight ?? 0.5) * 100);
            return (
              <span
                className="query-chip-wrap"
                key={key}
                onMouseEnter={canBlend ? () => openBlend(p) : undefined}
                onMouseLeave={canBlend ? scheduleBlendClose : undefined}
              >
              <span
                className={chipClass}
                onClick={() => onToggleExclude(key)}
                title={`${p.exclude ? 'Click to include' : 'Click to exclude'}${
                  canBlend ? ' · hover to blend with text' : ''
                }`}
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
                <span
                  className="query-chip-label"
                  title={
                    isClip || isFaceClip
                      ? isFace
                        ? 'Face clip'
                        : 'Screen clip'
                      : p.value
                  }
                >
                  {p.exclude ? '−' : ''}
                  {isVisual ? (
                    <>
                      <span className="query-chip-icon" aria-hidden="true">
                        ✨
                      </span>
                      {p.value}
                    </>
                  ) : isFace ? (
                    <>
                      <span className="query-chip-icon" aria-hidden="true">
                        👤
                      </span>
                      {isFaceClip ? (
                        <>
                          <img
                            className="query-chip-thumb"
                            src={p.value}
                            alt=""
                          />
                          Face clip
                        </>
                      ) : (
                        <>
                          <img
                            className="query-chip-thumb"
                            src={mediaUrl(p.value)}
                            alt=""
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                          {baseName}
                        </>
                      )}
                    </>
                  ) : isClip ? (
                    <>
                      <img
                        className="query-chip-thumb"
                        src={p.value}
                        alt=""
                      />
                      Screen clip
                    </>
                  ) : isSimilar ? (
                    <>
                      <img
                        className="query-chip-thumb"
                        src={mediaUrl(p.value)}
                        alt=""
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                      {baseName}
                    </>
                  ) : (
                    <>
                      {TYPE_GLYPH[p.type]}
                      {p.value}
                    </>
                  )}
                  {hasBlend && (
                    <span
                      className="query-chip-blend"
                      title={`Blended with “${p.text}” (${chipPct}% text)`}
                    >
                      <span aria-hidden="true">✨</span>
                      {p.text}
                      <span className="query-chip-blend-pct">{chipPct}%</span>
                    </span>
                  )}
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
              {blendOpen && (
                <span
                  className="query-chip-blend-pop"
                  onClick={(e) => e.stopPropagation()}
                >
                  <input
                    className="query-chip-blend-input"
                    type="text"
                    placeholder="Blend with text… (e.g. “at night”)"
                    value={blendText}
                    onChange={(e) => setBlendText(e.currentTarget.value)}
                    onKeyDown={(e) => {
                      e.stopPropagation();
                      if (e.key === 'Enter') {
                        commitBlend(key, e.currentTarget.value, blendPct);
                        blendFocusRef.current = false;
                        setBlendKey(null);
                      } else if (e.key === 'Escape') {
                        // Revert the draft; keep whatever was last committed.
                        setBlendText(p.text ?? '');
                        setBlendPct(chipPct);
                        blendFocusRef.current = false;
                        setBlendKey(null);
                      }
                    }}
                    onKeyUp={(e) => e.stopPropagation()}
                    onFocus={() => {
                      blendFocusRef.current = true;
                    }}
                    onBlur={(e) => {
                      blendFocusRef.current = false;
                      commitBlend(key, e.currentTarget.value, blendPct);
                      scheduleBlendClose();
                    }}
                  />
                  <span className="query-chip-blend-slider">
                    <span title="All image" aria-hidden="true">
                      🖼
                    </span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={5}
                      value={blendPct}
                      disabled={!blendText.trim()}
                      aria-label="Image vs. text weight"
                      onChange={(e) =>
                        setBlendPct(Number(e.currentTarget.value))
                      }
                      onPointerUp={(e) =>
                        commitBlend(
                          key,
                          blendText,
                          Number((e.target as HTMLInputElement).value)
                        )
                      }
                      onKeyUp={(e) => {
                        e.stopPropagation();
                        commitBlend(
                          key,
                          blendText,
                          Number(e.currentTarget.value)
                        );
                      }}
                      onFocus={() => {
                        blendFocusRef.current = true;
                      }}
                      onBlur={() => {
                        blendFocusRef.current = false;
                        scheduleBlendClose();
                      }}
                    />
                    <span title="All text" aria-hidden="true">
                      ✨
                    </span>
                    <span className="query-chip-blend-pct-label">
                      {blendPct}% text
                    </span>
                  </span>
                </span>
              )}
              </span>
            );
          })}
        </div>
      )}
      <div
        className={`query-input-field${meaningMode ? ' meaning-mode' : ''}`}
      >
        <input
          ref={inputRef}
          type="text"
          placeholder={
            meaningMode
              ? 'Describe what you’re looking for…'
              : 'Search & filter'
          }
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
        {onSubmitVisual && (
          <button
            type="button"
            className={`query-input-meaning${meaningMode ? ' active' : ''}`}
            title={
              meaningMode
                ? 'Search by meaning is ON — type a description and press Enter (off: normal filter)'
                : 'Search by meaning — describe images in words (SigLIP 2 text→image)'
            }
            aria-pressed={meaningMode}
            onMouseDown={(e) => e.preventDefault()}
            onClick={toggleMeaningMode}
          >
            ✨
          </button>
        )}
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
          {
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
          }
        </div>
      )}
    </div>
  );
}
