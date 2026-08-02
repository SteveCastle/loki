import { useState, useRef, useEffect, useCallback } from 'react';
import { useMeaningMode } from '../../hooks/useMeaningMode';
import { mediaUrl } from '../../platform';
import type { Query, Predicate, BlendNode } from '../../query/types';
import { predicateKey } from '../../query/types';
import {
  queryStateKey,
  type QueryHistoryEntry,
} from '../../query/history';
import { effectiveBlendNodes } from '../../query/reducer';
import { displayTagLabel } from '../../tag-display';
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
  // Composite blend nodes (similar/clip): manage the multi-node latent-space
  // query on a chip — add/remove text and image nodes, adjust weights, flip a
  // node negative. When supplied, the hover popover becomes a node editor.
  onAddBlendNode?: (key: string, node: BlendNode) => void;
  onRemoveBlendNode?: (key: string, index: number) => void;
  onUpdateBlendNode?: (
    key: string,
    index: number,
    patch: Partial<Pick<BlendNode, 'weight' | 'negative'>>
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
  // Session filter-state history (machine context.queryHistory): the last few
  // committed DB query states, newest first. Selecting one RE-RUNS that query
  // via onApplyHistory. The dropdown also pins a base row that restores the
  // session's starting library (the folder scan) via onApplyBase.
  history: QueryHistoryEntry[];
  onApplyHistory: (entry: QueryHistoryEntry) => void;
  // Label for the pinned base row (e.g. the folder's basename) and, when
  // known, how many items the base library holds.
  baseLabel: string;
  baseCount?: number;
  onApplyBase: () => void;
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
  faces: 'faces:',
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

// Compact "how long ago did this state last run" formatter for history rows.
function timeAgo(at: number): string {
  const s = Math.max(0, Math.floor((Date.now() - at) / 1000));
  if (s < 60) return 'just now';
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// Short display label for a predicate in a history row's mini-chip.
function predicateMiniLabel(p: Predicate): string {
  if (p.type === 'clip') return 'Screen clip';
  if (p.type === 'face' && p.value.startsWith('data:')) return 'Face clip';
  if (p.type === 'similar' || p.type === 'face') {
    return p.value.split(/[/\\]/).pop() || p.value;
  }
  return p.type === 'tag' ? displayTagLabel(p.value) : p.value;
}

// Substring match across everything a filter state is made of, so typing
// narrows the history the same way it narrows tag suggestions.
function entryMatchesText(entry: QueryHistoryEntry, needle: string): boolean {
  return entry.query.predicates.some(
    (p) =>
      predicateMiniLabel(p).toLowerCase().includes(needle) ||
      (p.text ?? '').toLowerCase().includes(needle) ||
      (p.nodes ?? []).some((n) => n.value.toLowerCase().includes(needle))
  );
}

// One predicate of a history entry, rendered as a read-only mini-chip:
// join connector, thumbnail/glyph, truncated value, blend-node count.
function HistoryMiniChip({ p, first }: { p: Predicate; first: boolean }) {
  const isImage =
    p.type === 'clip' ||
    (p.type === 'face' && p.value.startsWith('data:')) ||
    p.type === 'similar' ||
    (p.type === 'face' && !p.value.startsWith('data:'));
  const src =
    p.type === 'clip' || p.value.startsWith('data:')
      ? p.value
      : mediaUrl(p.value);
  const nodeCount = effectiveBlendNodes(p).length;
  const typeClass =
    p.type === 'visual'
      ? ' visual'
      : p.type === 'similar' || p.type === 'clip' || p.type === 'face'
      ? ' similar'
      : p.type === 'category'
      ? ' category'
      : '';
  return (
    <>
      {!first && (
        <span
          className={`query-history-join${
            (p.join ?? 'AND') === 'OR' ? ' or' : ''
          }`}
        >
          {p.join ?? 'AND'}
        </span>
      )}
      <span
        className={`query-history-chip${p.exclude ? ' exclude' : ''}${typeClass}`}
      >
        {p.exclude && <span className="query-history-chip-not">−</span>}
        {p.type === 'visual' && (
          <span className="query-history-chip-glyph" aria-hidden="true">
            ✨
          </span>
        )}
        {p.type === 'face' && (
          <span className="query-history-chip-glyph" aria-hidden="true">
            👤
          </span>
        )}
        {isImage && (
          <img
            className="query-history-chip-thumb"
            decoding="async"
            src={src}
            alt=""
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
        )}
        {!isImage && p.type !== 'visual' && (
          <span className="query-history-chip-glyph">{TYPE_GLYPH[p.type]}</span>
        )}
        <span className="query-history-chip-value">
          {predicateMiniLabel(p)}
        </span>
        {nodeCount > 0 && (
          <span
            className="query-history-chip-nodes"
            title={`${nodeCount} blended concept${nodeCount === 1 ? '' : 's'}`}
          >
            ⊕{nodeCount}
          </span>
        )}
      </span>
    </>
  );
}

export default function QueryInput({
  query,
  textValue,
  onTextChange,
  onSubmitText,
  onRemovePredicate,
  onToggleExclude,
  onSetPredicateJoin,
  onUpdatePredicateBlend,
  onAddBlendNode,
  onRemoveBlendNode,
  onUpdateBlendNode,
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
  history,
  onApplyHistory,
  baseLabel,
  baseCount,
  onApplyBase,
}: QueryInputProps) {
  // "Search by meaning" mode: typed text commits as a visual: predicate.
  // Shared + sticky: stays on across palette open/close until toggled off.
  const { meaningMode, setMeaningMode } = useMeaningMode();
  // Meaning mode only takes effect while the parent offers visual search —
  // the gate (server reachable + logged in) can close while the sticky
  // shared mode is still on, and the field must not keep advertising it.
  const meaningActive = meaningMode && Boolean(onSubmitVisual);
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

  // Blend popover (image chips only): which chip's node editor is open, plus
  // the draft text for the "add concept" input. Node weights commit on slider
  // release (never per drag-tick — every commit re-runs the query, an
  // embed-subprocess call per run on the server); toggles commit instantly.
  const [blendKey, setBlendKey] = useState<string | null>(null);
  const [addText, setAddText] = useState('');
  const blendCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelBlendClose = useCallback(() => {
    if (blendCloseTimer.current) {
      clearTimeout(blendCloseTimer.current);
      blendCloseTimer.current = null;
    }
  }, []);

  const openBlend = useCallback(
    (p: Predicate) => {
      cancelBlendClose();
      const key = predicateKey(p);
      if (blendKey === key) return;
      setBlendKey(key);
      setAddText('');
    },
    [cancelBlendClose, blendKey]
  );

  // The open chip's wrap (chip + popover), located by attribute rather than a
  // conditionally-attached ref: the popover can move between chips in one
  // commit, and React's detach-order on a switched ref object can null out
  // the newly attached element.
  const openBlendWrap = useCallback(
    () =>
      containerRef.current?.querySelector<HTMLElement>(
        '.query-chip-wrap[data-blend-open="true"]'
      ) ?? null,
    []
  );

  // Deferred close that the LIVE DOM must confirm. Mouseleave/blur latches are
  // not trusted here: a chip that re-renders under a new key while hovered
  // (any node edit commits → the query re-runs → predicateKey changes) is
  // replaced without ever firing mouseleave, and a latch that misses that
  // event used to wedge the popover open forever. Instead the timer checks
  // whether the wrap is actually hovered or holds focus, and RE-ARMS while it
  // is — a skipped close is always retried, so no stale signal is permanent.
  const scheduleBlendClose = useCallback(() => {
    cancelBlendClose();
    const attempt = () => {
      blendCloseTimer.current = null;
      const wrap = openBlendWrap();
      if (
        wrap &&
        (wrap.matches(':hover') || wrap.contains(document.activeElement))
      ) {
        blendCloseTimer.current = setTimeout(attempt, 250);
        return;
      }
      setBlendKey(null);
    };
    blendCloseTimer.current = setTimeout(attempt, 250);
  }, [cancelBlendClose, openBlendWrap]);

  // Hard close on any press outside the open chip+popover — the guaranteed
  // manual escape even if every hover/focus signal above has gone stale.
  useEffect(() => {
    if (!blendKey) return;
    const onPointerDown = (e: PointerEvent) => {
      const wrap = openBlendWrap();
      if (!wrap || !wrap.contains(e.target as Node)) {
        cancelBlendClose();
        setBlendKey(null);
      }
    };
    document.addEventListener('pointerdown', onPointerDown, true);
    return () =>
      document.removeEventListener('pointerdown', onPointerDown, true);
  }, [blendKey, cancelBlendClose, openBlendWrap]);

  // Clear any pending popover-close timer on unmount.
  useEffect(() => cancelBlendClose, [cancelBlendClose]);

  // Focus the text input on mount when requested (e.g. the command palette
  // opens) so the user can start typing a query immediately.
  useEffect(() => {
    if (autoFocus) inputRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Identity of the CURRENT filter state, so the dropdown can mark the entry
  // (or the base row, when the query is empty) the user is already on.
  const currentKey = queryStateKey(query);
  const atBase = query.predicates.length === 0;

  const needle = textValue.trim().toLowerCase();
  const filteredHistory = needle
    ? history.filter((entry) => entryMatchesText(entry, needle))
    : history;

  // The base row is always present, so the dropdown always has something to
  // show; keyboard navigation spans the history rows then the base row.
  const navCount = filteredHistory.length + 1;
  const baseIndex = filteredHistory.length;

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

  // Re-apply a filter state from the dropdown (re-runs the query), or return
  // to the base library. Both clear any typed text — the applied state is now
  // the whole query.
  const applyEntry = useCallback(
    (entry: QueryHistoryEntry) => {
      onApplyHistory(entry);
      onTextChange('');
      setIsOpen(false);
      setHighlightIndex(-1);
    },
    [onApplyHistory, onTextChange]
  );

  const applyBase = useCallback(() => {
    onApplyBase();
    onTextChange('');
    setIsOpen(false);
    setHighlightIndex(-1);
  }, [onApplyBase, onTextChange]);

  const applyRow = useCallback(
    (index: number) => {
      if (index === baseIndex) applyBase();
      else if (index >= 0 && index < filteredHistory.length)
        applyEntry(filteredHistory[index]);
    },
    [baseIndex, applyBase, applyEntry, filteredHistory]
  );

  const handleSubmit = useCallback(() => {
    const trimmed = textValue.trim();
    if (trimmed) {
      if (meaningMode && onSubmitVisual) {
        // Commit the raw text as a visual (text→image) embedding search.
        onSubmitVisual(trimmed);
      } else {
        onSubmitText();
      }
      setIsOpen(false);
      setHighlightIndex(-1);
    }
  }, [textValue, onSubmitText, meaningMode, onSubmitVisual]);

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

      if (!isOpen) {
        if (e.key === 'Enter') {
          handleSubmit();
          return;
        }
        if (e.key === 'Escape') {
          setIsOpen(false);
          return;
        }
        // Keyboard affordance: open (and highlight the first row) on ArrowDown
        // so keyboard-only users can still reach the filter history now that
        // focus alone no longer opens the dropdown.
        if (e.key === 'ArrowDown') {
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
          setHighlightIndex((prev) => (prev < navCount - 1 ? prev + 1 : 0));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setHighlightIndex((prev) => (prev > 0 ? prev - 1 : navCount - 1));
          break;
        case 'Enter':
          if (highlightIndex >= 0 && highlightIndex < navCount) {
            applyRow(highlightIndex);
          } else {
            handleSubmit();
          }
          break;
        case 'Escape':
          setIsOpen(false);
          setHighlightIndex(-1);
          break;
      }
    },
    [
      isOpen,
      highlightIndex,
      navCount,
      applyRow,
      handleSubmit,
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
  // active navigation target, so the two don't overlap or fight for arrow
  // keys. The dropdown always has content — the base row is always there.
  const dropdownOpen = !resultNavActive && isOpen;

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
            // Chip thumbnails point at the FULL media file (there is no
            // thumbnail route for an arbitrary predicate value), so every one
            // of them is decoded with `decoding="async"` below — a synchronous
            // decode of a large image would otherwise land in the same frame
            // that opens the command palette and delay its first paint.
            //
            // Embedding chips (similar/clip images and visual text) carry a
            // composite blend: hovering opens the node editor (stack images,
            // add/remove text concepts, weights, negative steering).
            const isBlendable = isSimilar || isClip || isVisual;
            const canBlend =
              isBlendable &&
              !!onAddBlendNode &&
              !!onRemoveBlendNode &&
              !!onUpdateBlendNode;
            const blendOpen = canBlend && blendKey === key;
            const effNodes = isBlendable ? effectiveBlendNodes(p) : [];
            return (
              <span
                className="query-chip-wrap"
                key={key}
                data-blend-open={blendOpen ? 'true' : undefined}
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
                        decoding="async"
                            src={p.value}
                            alt=""
                          />
                          Face clip
                        </>
                      ) : (
                        <>
                          <img
                            className="query-chip-thumb"
                        decoding="async"
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
                        decoding="async"
                        src={p.value}
                        alt=""
                      />
                      Screen clip
                    </>
                  ) : isSimilar ? (
                    <>
                      <img
                        className="query-chip-thumb"
                        decoding="async"
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
                      {p.type === 'tag' ? displayTagLabel(p.value) : p.value}
                    </>
                  )}
                  {effNodes.length > 0 && (
                    <span
                      className="query-chip-blend"
                      title={effNodes
                        .map(
                          (n) =>
                            `${n.negative ? '−' : '+'} ${
                              n.kind === 'text' ? `“${n.value}”` : n.kind
                            } (${Math.round((n.weight ?? 1) * 100)}%)`
                        )
                        .join('\n')}
                    >
                      {effNodes.length === 1 && effNodes[0].kind === 'text' ? (
                        <>
                          <span aria-hidden="true">
                            {effNodes[0].negative ? '−✨' : '✨'}
                          </span>
                          {effNodes[0].value}
                        </>
                      ) : (
                        <>
                          <span aria-hidden="true">⊕</span>
                          {effNodes.length}
                        </>
                      )}
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
                  {/* Base component — always weight 1, the anchor of the query. */}
                  <span className="query-blend-node query-blend-node--base">
                    {isVisual ? (
                      <span
                        className="query-blend-node-thumb query-blend-node-thumb--text"
                        aria-hidden="true"
                      >
                        ✨
                      </span>
                    ) : (
                      <img
                        className="query-blend-node-thumb"
                        src={isClip ? p.value : mediaUrl(p.value)}
                        alt=""
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    )}
                    <span className="query-blend-node-label">
                      {isVisual ? p.value : isClip ? 'Screen clip' : baseName}
                    </span>
                    <span className="query-blend-node-hint">base</span>
                  </span>
                  {effNodes.map((n, ni) => {
                    const pct = Math.round((n.weight ?? 1) * 100);
                    return (
                      <span
                        className={`query-blend-node${
                          n.negative ? ' negative' : ''
                        }`}
                        key={`${n.kind}:${n.value}:${ni}`}
                      >
                        {n.kind === 'text' ? (
                          <span
                            className="query-blend-node-thumb query-blend-node-thumb--text"
                            aria-hidden="true"
                          >
                            ✨
                          </span>
                        ) : (
                          <img
                            className="query-blend-node-thumb"
                            src={n.kind === 'clip' ? n.value : mediaUrl(n.value)}
                            alt=""
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                        )}
                        <span
                          className="query-blend-node-label"
                          title={n.value}
                        >
                          {n.kind === 'text'
                            ? n.value
                            : n.kind === 'clip'
                            ? 'Screen clip'
                            : n.value.split(/[/\\]/).pop() || n.value}
                        </span>
                        <input
                          type="range"
                          min={5}
                          max={100}
                          step={5}
                          // Uncontrolled + remount on committed weight: drag is
                          // local, commit happens on release only.
                          key={`w${pct}`}
                          defaultValue={pct}
                          aria-label="Node weight"
                          onPointerUp={(e) =>
                            onUpdateBlendNode?.(key, ni, {
                              weight:
                                Number((e.target as HTMLInputElement).value) /
                                100,
                            })
                          }
                          onKeyUp={(e) => {
                            e.stopPropagation();
                            onUpdateBlendNode?.(key, ni, {
                              weight: Number(e.currentTarget.value) / 100,
                            });
                          }}
                          onFocus={cancelBlendClose}
                          onBlur={scheduleBlendClose}
                        />
                        <span className="query-blend-node-pct">{pct}%</span>
                        <button
                          type="button"
                          className={`query-blend-node-neg${
                            n.negative ? ' active' : ''
                          }`}
                          title={
                            n.negative
                              ? 'Steering AWAY from this — click to attract'
                              : 'Click to steer away from this (negative vector)'
                          }
                          onClick={() =>
                            onUpdateBlendNode?.(key, ni, {
                              negative: !n.negative,
                            })
                          }
                        >
                          −
                        </button>
                        <button
                          type="button"
                          className="query-blend-node-remove"
                          title="Remove from blend"
                          onClick={() => onRemoveBlendNode?.(key, ni)}
                        >
                          ×
                        </button>
                      </span>
                    );
                  })}
                  <input
                    className="query-chip-blend-input"
                    type="text"
                    placeholder="Add concept… (e.g. “at night”), Enter to add"
                    value={addText}
                    onChange={(e) => setAddText(e.currentTarget.value)}
                    onKeyDown={(e) => {
                      e.stopPropagation();
                      if (e.key === 'Enter' && addText.trim()) {
                        onAddBlendNode?.(key, {
                          kind: 'text',
                          value: addText.trim(),
                          weight: 0.5,
                        });
                        setAddText('');
                      } else if (e.key === 'Escape') {
                        setAddText('');
                        setBlendKey(null);
                      }
                    }}
                    onKeyUp={(e) => e.stopPropagation()}
                    onFocus={cancelBlendClose}
                    onBlur={scheduleBlendClose}
                  />
                  <span className="query-blend-pop-hint">
                    Similar images added in ∩/∪ mode stack here as nodes.
                  </span>
                </span>
              )}
              </span>
            );
          })}
        </div>
      )}
      <div
        className={`query-input-field${meaningActive ? ' meaning-mode' : ''}`}
      >
        <input
          ref={inputRef}
          type="text"
          placeholder={
            meaningActive
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
          <div className="query-input-history">
            <div className="query-input-section-header">
              <span>
                {needle ? 'Matching filters' : 'Session filters'}
              </span>
              <span className="query-input-section-hint">
                click to re-run
              </span>
            </div>
            {filteredHistory.length === 0 ? (
              <div className="query-input-empty">
                {history.length === 0
                  ? 'Filters you apply will appear here'
                  : 'No filters match the typed text'}
              </div>
            ) : (
              filteredHistory.map((entry, index) => {
                const isCurrent = !atBase && entry.key === currentKey;
                return (
                  <div
                    className={`query-history-row${
                      index === highlightIndex ? ' highlighted' : ''
                    }${isCurrent ? ' current' : ''}`}
                    key={entry.key}
                    onMouseEnter={() => setHighlightIndex(index)}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => applyEntry(entry)}
                    title={
                      isCurrent
                        ? 'This is the active filter — click to re-run it'
                        : 'Apply this filter state (re-runs the query)'
                    }
                  >
                    <span className="query-history-chips">
                      {entry.query.predicates.map((p, pi) => (
                        <HistoryMiniChip
                          key={`${predicateKey(p)}:${pi}`}
                          p={p}
                          first={pi === 0}
                        />
                      ))}
                    </span>
                    <span className="query-history-meta">
                      {isCurrent && (
                        <span className="query-history-badge">current</span>
                      )}
                      <span className="query-history-count">
                        {entry.count.toLocaleString()}{' '}
                        {entry.count === 1 ? 'item' : 'items'}
                      </span>
                      <span className="query-history-time">
                        {timeAgo(entry.at)}
                      </span>
                    </span>
                  </div>
                );
              })
            )}
            <div
              className={`query-history-base${
                highlightIndex === baseIndex ? ' highlighted' : ''
              }${atBase ? ' current' : ''}`}
              onMouseEnter={() => setHighlightIndex(baseIndex)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={applyBase}
              title={
                atBase
                  ? 'You are on the starting library'
                  : 'Back to the starting library (clears all filters)'
              }
            >
              <span className="query-history-base-icon" aria-hidden="true">
                ⌂
              </span>
              <span className="query-history-base-text">
                <span className="query-history-base-label">{baseLabel}</span>
                <span className="query-history-base-sub">
                  Starting library
                </span>
              </span>
              <span className="query-history-meta">
                {atBase && (
                  <span className="query-history-badge">current</span>
                )}
                {typeof baseCount === 'number' && baseCount > 0 && (
                  <span className="query-history-count">
                    {baseCount.toLocaleString()}{' '}
                    {baseCount === 1 ? 'item' : 'items'}
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
