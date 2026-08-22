// QueryHistorySection — the session filter-state history (plus the pinned
// "starting library" base row) rendered IN-FLOW as one more type-ahead
// section, exactly like the Tags / Categories / Paths sections around it.
//
// This used to be an overlay dropdown anchored under QueryInput, but the
// overlay covered the very results that update live as the user types (the
// sidebar's tag grid, the palette's suggestion rows). Now both hosts render it
// inside their normal results layout instead: the command palette appends it
// to its results surface (and folds its rows into the shared keyboard
// navigation via onItemsChange), the taxonomy sidebar appends it to the
// suggestions pane (click-only).
import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../platform';
import type { Query, Predicate } from '../../query/types';
import { queryStateKey, type QueryHistoryEntry } from '../../query/history';
import { effectiveBlendNodes } from '../../query/reducer';
import { displayTagLabel } from '../../tag-display';
import type { SuggestionItem } from '../taxonomy/suggestion-sections';
import './query-input.css';

// Glyph prefix shown on a mini-chip for each predicate type. Also used by
// QueryInput for the live (full-size) chips.
export const TYPE_GLYPH: Record<Predicate['type'], string> = {
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
  orientation: 'orientation:',
};

// Navigation keys for the rows this section contributes to a host's shared
// keyboard-navigable list (command palette).
export const HISTORY_BASE_KEY = 'hist:base';
export const historyRowKey = (entry: QueryHistoryEntry) => `hist:${entry.key}`;

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

interface QueryHistorySectionProps {
  // The LIVE query, so the row matching the active filter state (or the base
  // row, when the query is empty) can be marked "current".
  query: Query;
  // The typed search text. Non-empty → history rows are filtered to entries
  // that contain it; the base row is always shown.
  text: string;
  // Session filter-state history (machine context.queryHistory), newest first.
  // Selecting a row RE-RUNS that query via onApplyHistory.
  history: QueryHistoryEntry[];
  onApplyHistory: (entry: QueryHistoryEntry) => void;
  // Label for the pinned base row (e.g. the folder's basename) and, when
  // known, how many items the base library holds. Selecting it restores the
  // session's starting library via onApplyBase.
  baseLabel: string;
  baseCount?: number;
  onApplyBase: () => void;
  // Optional keyboard-navigation hooks (command palette), mirroring
  // SuggestionSections: report the ordered rows so the host's highlight can
  // span them, and render/steer the highlight by key. Omitted in the taxonomy
  // sidebar, where the section is click-only.
  highlightedKey?: string | null;
  onHighlightKey?: (key: string) => void;
  onItemsChange?: (items: SuggestionItem[]) => void;
  // Render the section header as a toggle and start COLLAPSED (the command
  // palette, where vertical space is tight). While collapsed no rows render
  // and none are reported to onItemsChange, so keyboard navigation can never
  // land on a hidden row. Omitted (the sidebar) → always expanded, no toggle.
  collapsible?: boolean;
}

export default function QueryHistorySection({
  query,
  text,
  history,
  onApplyHistory,
  baseLabel,
  baseCount,
  onApplyBase,
  highlightedKey,
  onHighlightKey,
  onItemsChange,
  collapsible = false,
}: QueryHistorySectionProps) {
  // Collapsible hosts start collapsed; expanding is a per-mount choice (the
  // palette re-collapses on its next open, which is the space-saving point).
  const [expanded, setExpanded] = useState(!collapsible);

  const needle = text.trim().toLowerCase();
  const filteredHistory = needle
    ? history.filter((entry) => entryMatchesText(entry, needle))
    : history;

  // Identity of the CURRENT filter state, so the section can mark the entry
  // (or the base row, when the query is empty) the user is already on.
  const currentKey = queryStateKey(query);
  const atBase = query.predicates.length === 0;

  // Ordered, navigable rows for the host's shared keyboard navigation: the
  // history rows then the always-present base row — or nothing while
  // collapsed, since hidden rows must not be reachable by arrow keys.
  // Reported through the same signature-keyed effect pattern as
  // SuggestionSections so a fresh array per render doesn't loop with the
  // host's setState.
  const items: SuggestionItem[] = expanded
    ? [
        ...filteredHistory.map((entry) => ({
          key: historyRowKey(entry),
          action: () => onApplyHistory(entry),
        })),
        { key: HISTORY_BASE_KEY, action: onApplyBase },
      ]
    : [];
  const onItemsChangeRef = useRef(onItemsChange);
  onItemsChangeRef.current = onItemsChange;
  const itemsSignature = items.map((i) => i.key).join('|');
  useEffect(() => {
    onItemsChangeRef.current?.(items);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [itemsSignature]);

  const label = needle ? 'Matching filters' : 'Session filters';
  // Plain label, or — for collapsible hosts — a toggle whose mousedown is
  // swallowed so expanding/collapsing never blurs the palette's input. The
  // count keeps a collapsed section honest about what it's hiding.
  const header = collapsible ? (
    <button
      type="button"
      className="suggestion-section-label query-history-toggle"
      aria-expanded={expanded}
      onMouseDown={(e) => e.preventDefault()}
      onClick={() => setExpanded((v) => !v)}
      title={expanded ? 'Collapse' : 'Expand'}
    >
      <span className="query-history-caret" aria-hidden="true">
        {expanded ? '▾' : '▸'}
      </span>
      {label}
      <span className="query-history-toggle-count">
        {filteredHistory.length}
      </span>
    </button>
  ) : (
    <div className="suggestion-section-label">{label}</div>
  );

  if (!expanded) {
    return (
      <div className="suggestion-section query-history-section">{header}</div>
    );
  }

  return (
    <div className="suggestion-section query-history-section">
      {header}
      {history.length === 0 && !needle && (
        <div className="query-input-empty">
          Filters you apply will appear here
        </div>
      )}
      {filteredHistory.map((entry) => {
        const key = historyRowKey(entry);
        const isCurrent = !atBase && entry.key === currentKey;
        return (
          <div
            className={`query-history-row${
              highlightedKey === key ? ' highlighted' : ''
            }${isCurrent ? ' current' : ''}`}
            key={entry.key}
            onMouseEnter={() => onHighlightKey?.(key)}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onApplyHistory(entry)}
            title={
              isCurrent
                ? 'This is the active filter — click to re-run it'
                : 'Apply this filter state (re-runs the query)'
            }
          >
            <span className="query-history-chips">
              {entry.query.predicates.map((p, pi) => (
                <HistoryMiniChip
                  key={`${p.type}:${p.value}:${pi}`}
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
              <span className="query-history-time">{timeAgo(entry.at)}</span>
            </span>
          </div>
        );
      })}
      <div
        className={`query-history-base${
          highlightedKey === HISTORY_BASE_KEY ? ' highlighted' : ''
        }${atBase ? ' current' : ''}`}
        onMouseEnter={() => onHighlightKey?.(HISTORY_BASE_KEY)}
        onMouseDown={(e) => e.preventDefault()}
        onClick={onApplyBase}
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
          <span className="query-history-base-sub">Starting library</span>
        </span>
        <span className="query-history-meta">
          {atBase && <span className="query-history-badge">current</span>}
          {typeof baseCount === 'number' && baseCount > 0 && (
            <span className="query-history-count">
              {baseCount.toLocaleString()}{' '}
              {baseCount === 1 ? 'item' : 'items'}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}
