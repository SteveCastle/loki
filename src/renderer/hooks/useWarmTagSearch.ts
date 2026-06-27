// useWarmTagSearch — builds the shared tag-search index at app startup.
//
// Mounted once near the app root (after the library DB is available) so the
// full all-tags list is fetched eagerly and pushed into the shared singleton
// index. By the time the user opens the taxonomy sidebar or the command
// palette, the index is already warm and the first search is instant.
import { useContext, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { GlobalStateContext } from '../state';
import { invoke } from '../platform';
import { indexTags, type TagConcept } from '../search/tag-search-service';

async function loadAllTags(): Promise<TagConcept[]> {
  const result = await invoke('load-all-tags', []);
  return (result as TagConcept[]) ?? [];
}

export function useWarmTagSearch(): void {
  const { libraryService } = useContext(GlobalStateContext);
  const initSessionId = useSelector(
    libraryService,
    (state) => state.context.initSessionId
  );

  // Same key + loader as useTagSearch / taxonomy so this primes the shared
  // cache. Gated on initSessionId (assigned only once the machine reaches its
  // post-DB `init` state) so the fetch never races ahead of the load-db handler
  // registration — firing earlier just yields a guaranteed "No handler
  // registered for 'load-all-tags'" rejection on every launch. staleTime:
  // Infinity means it then runs at most once per session (re-keyed on
  // initSessionId).
  const { data: allTagsData } = useQuery<TagConcept[], Error>(
    ['taxonomy', 'all-tags', initSessionId],
    loadAllTags,
    { enabled: !!initSessionId, staleTime: Infinity }
  );

  // Build the shared index ahead of first use, off the RAW React Query array so
  // the same reference is shared with every other consumer (taxonomy, palette).
  // That shared reference is what lets indexTags clone the library to the worker
  // exactly once — here at startup — instead of again on each surface's first
  // search. The defensive label filter lives inside indexTags.
  useEffect(() => {
    if (allTagsData) indexTags(allTagsData);
  }, [allTagsData]);
}

export default useWarmTagSearch;
