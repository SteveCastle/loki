// Live view of the library search in flight (see ../search-activity.ts).
// Selectors are cheap and stable, so idle components re-render only when a
// search starts or lands.
import { useCallback, useMemo } from 'react';
import { useSelector } from '@xstate/react';
import {
  describeSearch,
  isSearchInFlight,
  pendingPredicateKeys,
  type SearchDescription,
} from '../search-activity';

const NO_KEYS: ReadonlySet<string> = new Set();

export interface SearchActivity {
  busy: boolean;
  startedAt: number;
  // Chips the results on screen don't reflect yet.
  pendingKeys: ReadonlySet<string>;
  description: SearchDescription | null;
  // Stop the search and keep the results (and chips) currently on screen.
  cancel: () => void;
}

export function useSearchActivity(libraryService: any): SearchActivity {
  const busy = useSelector(libraryService, isSearchInFlight);
  const query = useSelector(libraryService, (s: any) => s.context.query);
  const settled = useSelector(
    libraryService,
    (s: any) => s.context.settledQuery
  );
  const startedAt = useSelector(
    libraryService,
    (s: any) => s.context.queryStartedAt as number
  );
  const pendingKeys = useMemo(
    () => (busy ? pendingPredicateKeys(query, settled) : NO_KEYS),
    [busy, query, settled]
  );
  const description = useMemo(
    () => (busy ? describeSearch(query, settled) : null),
    [busy, query, settled]
  );
  const cancel = useCallback(
    () => libraryService.send('CANCEL_QUERY'),
    [libraryService]
  );
  return { busy, startedAt, pendingKeys, description, cancel };
}
