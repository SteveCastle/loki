import { useState, useRef, useEffect, useCallback } from 'react';
import { useMeaningMode } from '../../hooks/useMeaningMode';
import { mediaUrl } from '../../platform';
import type { Query, Predicate, BlendNode } from '../../query/types';
import { predicateKey } from '../../query/types';
import { effectiveBlendNodes } from '../../query/reducer';
import { displayTagLabel } from '../../tag-display';
import { TYPE_GLYPH } from './query-history-section';
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
  // Composite scoring mode: 'blend' averages the components into one query
  // vector; 'shared' requires candidates to match EVERY positive component
  // (zeroes in on what the components have in common).
  onSetBlendMode?: (key: string, mode: 'blend' | 'shared') => void;
  onClearAll: () => void; // clear chips + text (resets the library)
  onClearText: () => void; // clear only the typed text (no-op on the library)
  onFocus?: () => void;
  autoFocus?: boolean; // focus the text input on mount (fast palette workflow)
  disabled?: boolean;
  // Result-navigation bridge. When the parent renders its own results surface
  // (the command palette) and passes resultNavCount > 0, the input forwards
  // Arrow Up/Down and Enter to the parent so the user can keyboard-navigate
  // the highlighted result. Omitted everywhere else (e.g. the taxonomy
  // sidebar), where the input is plain: Enter submits, arrows do nothing.
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
// NOTE: the session filter-state history used to render as a dropdown under
// this input. It overlapped the live results beneath, so it now lives in
// QueryHistorySection (./query-history-section), which each host renders
// in-flow inside its own type-ahead results surface.

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
  onSetBlendMode,
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
  // items, arrow/enter drive that list instead of the input itself.
  const resultNavActive = resultNavCount > 0;
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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

  const handleSubmit = useCallback(() => {
    const trimmed = textValue.trim();
    if (trimmed) {
      if (meaningMode && onSubmitVisual) {
        // Commit the raw text as a visual (text→image) embedding search.
        onSubmitVisual(trimmed);
      } else {
        onSubmitText();
      }
    }
  }, [textValue, onSubmitText, meaningMode, onSubmitVisual]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      e.stopPropagation();

      // Result-navigation mode: the parent's results surface owns the
      // highlight, so arrow/enter drive it instead of the input itself.
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

      if (e.key === 'Enter') handleSubmit();
    },
    [
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
  }, [query.predicates.length, onClearAll, onClearText]);

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
                  {effNodes.length > 0 && onSetBlendMode && (
                    <span
                      className="query-blend-mode"
                      role="group"
                      aria-label="How components combine"
                    >
                      <button
                        type="button"
                        className={`query-blend-mode-btn${
                          p.blendMode !== 'shared' ? ' active' : ''
                        }`}
                        title="Blend: rank by AVERAGE similarity to the components (one combined query)"
                        onClick={() => onSetBlendMode(key, 'blend')}
                      >
                        Blend
                      </button>
                      <button
                        type="button"
                        className={`query-blend-mode-btn${
                          p.blendMode === 'shared' ? ' active' : ''
                        }`}
                        title="Match all: results must resemble EVERY component — zeroes in on what they share"
                        onClick={() => onSetBlendMode(key, 'shared')}
                      >
                        Match all
                      </button>
                    </span>
                  )}
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
          onChange={(e) => onTextChange(e.currentTarget.value)}
          onKeyDown={handleKeyDown}
          onKeyUp={(e) => e.stopPropagation()}
          onFocus={() => onFocus?.()}
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
    </div>
  );
}
