// SearchOverlay — the library search in flight, shown over the results it is
// about to replace. A sweep runs along the top edge; after a beat the stale
// results dim (still scrollable — the overlay never takes pointer events
// except on its own pill) and, in the list, a floating status pill with
// Cancel appears. The host element must be position: relative.
//
// `minimal` (the detail panel) shows only the sweep, so the pill appears once
// on screen even with list and detail side by side.
import { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import { useSearchActivity } from '../../hooks/useSearchActivity';
import SearchStatus from '../controls/search-status';
import './search-overlay.css';

export default function SearchOverlay({ minimal = false }: { minimal?: boolean }) {
  const { libraryService } = useContext(GlobalStateContext);
  const search = useSearchActivity(libraryService);
  if (!search.busy) return null;
  const kind = search.description?.kind ?? 'filter';
  return (
    <div
      className={`search-overlay search-overlay--${kind}${
        minimal ? ' search-overlay--minimal' : ''
      }`}
    >
      {!minimal && <div className="search-overlay-veil" />}
      <div className="search-overlay-bar" />
      {!minimal && (
        <div className="search-overlay-pill">
          <SearchStatus activity={search} variant="floating" />
        </div>
      )}
    </div>
  );
}
