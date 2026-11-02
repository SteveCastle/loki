// SearchStatus — what the library search in flight is doing, how long it has
// been at it, and a way out. Rendered under the query input (command palette,
// taxonomy sidebar) and as the floating pill over the list.
//
// Mount it only while a search is running. It fades in after a short delay
// (CSS), so a fast search never flashes it, and only this component ticks the
// elapsed clock — its hosts don't re-render per tick.
import { useEffect, useState } from 'react';
import type { SearchActivity } from '../../hooks/useSearchActivity';
import './search-status.css';

// The "why is this slow" hint waits this long: most searches land before it.
const SLOW_HINT_AFTER_MS = 3500;
// Under this, the clock would just flicker 0.1s, 0.2s… on fast searches.
const SHOW_ELAPSED_AFTER_MS = 700;

const formatElapsed = (ms: number) => {
  const s = ms / 1000;
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.floor(s)}s`;
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

interface SearchStatusProps {
  activity: SearchActivity;
  variant?: 'inline' | 'floating';
}

export default function SearchStatus({
  activity,
  variant = 'inline',
}: SearchStatusProps) {
  const { description, startedAt, cancel } = activity;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(timer);
  }, []);
  if (!description) return null;
  const elapsed = startedAt ? Math.max(0, now - startedAt) : 0;
  const { kind, verb, subject, slowHint } = description;
  return (
    <div
      className={`search-status search-status--${variant} search-status--${kind}`}
      role="status"
      aria-live="polite"
    >
      <span className="search-status-orb" aria-hidden="true" />
      <span className="search-status-text">
        {verb}
        {subject ? (
          <>
            {' '}
            <span className="search-status-subject">“{subject}”</span>
          </>
        ) : null}
        <span className="search-status-ellipsis" aria-hidden="true" />
      </span>
      {elapsed >= SHOW_ELAPSED_AFTER_MS && (
        <span className="search-status-elapsed">{formatElapsed(elapsed)}</span>
      )}
      <button
        type="button"
        className="search-status-cancel"
        // Keep focus in the query input (palette / sidebar typing).
        onMouseDown={(e) => e.preventDefault()}
        onClick={cancel}
        title="Stop this search and keep the current results"
      >
        Cancel
      </button>
      {slowHint && elapsed >= SLOW_HINT_AFTER_MS && (
        <span className="search-status-hint">{slowHint}</span>
      )}
    </div>
  );
}
