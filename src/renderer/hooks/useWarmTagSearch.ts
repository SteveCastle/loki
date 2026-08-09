// useWarmTagSearch — builds the command palette's tag-search index in the
// background, once the app has finished the work the user is actually waiting on.
//
// Mounted once near the app root. It warms the CURATED scope only: that is what
// the palette — the app's most frequently opened surface — searches, and it is
// ~5.6K tags instead of the ~189K in the full set. The taxonomy sidebar's
// complete index is deliberately NOT warmed here; the sidebar fetches it lazily
// when its search input gains focus, so the cost lands on the surface that needs
// it rather than on every launch.
import { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { GlobalStateContext } from '../state';
import { invoke } from '../platform';
import { indexTags, type TagConcept } from '../search/tag-search-service';
import { loadTagsForScope, tagScopeQueryKey } from './useTagSearch';
import { onIdleAfterFirstPaint } from '../first-paint';

// What the palette searches. See ../search/tag-scopes.
const WARM_SCOPE = 'curated' as const;

export function useWarmTagSearch(): void {
  const { libraryService } = useContext(GlobalStateContext);
  const initSessionId = useSelector(
    libraryService,
    (state) => state.context.initSessionId
  );

  // Held off the startup critical path. Even at the curated scope this is a DB
  // query, an IPC clone and a Fuse build, and none of it is on the path to
  // showing the user the file they opened. Nothing on screen needs it: it only
  // makes the FIRST type-ahead instant, and the user cannot type before the app
  // has painted.
  const [warmable, setWarmable] = useState(false);
  useEffect(() => onIdleAfterFirstPaint(() => setWarmable(true)), []);

  // Same key + loader as useTagSearch so this primes the shared cache. Also
  // gated on initSessionId (assigned only once the machine reaches its post-DB
  // `init` state) so the fetch never races ahead of the load-db handler
  // registration — firing earlier just yields a guaranteed "No handler
  // registered for 'load-all-tags'" rejection on every launch. staleTime:
  // Infinity means it then runs at most once per session (re-keyed on
  // initSessionId).
  const { data: allTagsData } = useQuery<TagConcept[], Error>(
    tagScopeQueryKey(WARM_SCOPE, initSessionId),
    () => loadTagsForScope(WARM_SCOPE),
    { enabled: !!initSessionId && warmable, staleTime: Infinity }
  );

  // Build the shared index ahead of first use, off the RAW React Query array so
  // the same reference is shared with every other consumer of this scope. That
  // shared reference is what lets indexTags clone to the worker exactly once —
  // here at startup — instead of again on the palette's first search.
  useEffect(() => {
    if (allTagsData) indexTags(allTagsData, WARM_SCOPE);
  }, [allTagsData]);

  // The categories list, on the same idle window and under the SAME query key
  // the palette uses. It is a trivial query (a couple of dozen rows) but it is
  // the last thing the palette waits on to be fully ready, and asking for it
  // at open time means queueing behind whatever the main process is still
  // doing — measured at ~194ms on a first open. Prefetching costs nothing here
  // and makes it a cache hit later.
  const queryClient = useQueryClient();
  useEffect(() => {
    if (!initSessionId || !warmable) return;
    queryClient.prefetchQuery({
      queryKey: ['taxonomy', 'categories', initSessionId],
      queryFn: async () => (await invoke('load-categories', [])) ?? [],
      staleTime: Infinity,
      cacheTime: Infinity,
    });
  }, [initSessionId, warmable, queryClient]);
}

export default useWarmTagSearch;
