// useTagSearch — reusable tag type-ahead over the SHARED, pre-warmed index.
//
// Both the taxonomy sidebar and the command palette consume this hook. It no
// longer owns a worker or Fuse instance: all indexing + matching is routed
// through the module-level singleton in ../search/tag-search-service, so the
// whole app shares ONE worker + ONE index (warmed at startup by
// useWarmTagSearch). This keeps the first search from either surface instant.
import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { debounce } from 'lodash';
import { GlobalStateContext } from '../state';
import { invoke } from '../platform';
import {
  indexTags,
  searchTags,
  type TagConcept,
} from '../search/tag-search-service';

export type { TagConcept } from '../search/tag-search-service';

// Mirrors the historical cap: each rendered result can trigger downstream IPC
// work, so we cap the match list even though Fuse ranks across the full set.
const MAX_SEARCH_RESULTS = 200;

async function loadAllTags(): Promise<TagConcept[]> {
  const result = await invoke('load-all-tags', []);
  return (result as TagConcept[]) ?? [];
}

/**
 * Off-thread fuzzy tag search backed by the shared singleton index.
 *
 * @param text    The raw (un-debounced) search text.
 * @param enabled Gates the active search so it only runs while the consuming
 *                surface wants results.
 */
export function useTagSearch(
  text: string,
  enabled: boolean
): { results: TagConcept[] } {
  const { libraryService } = useContext(GlobalStateContext);
  const initSessionId = useSelector(
    libraryService,
    (state) => state.context.initSessionId
  );

  // Debounce the raw text 150ms into the value actually dispatched to search.
  const [debouncedText, setDebouncedText] = useState<string>('');
  const debouncedSet = useRef(
    debounce((value: string) => setDebouncedText(value), 150)
  );
  useEffect(() => {
    debouncedSet.current(text);
  }, [text]);
  useEffect(() => {
    const d = debouncedSet.current;
    return () => d.cancel();
  }, []);

  // Full-tag fetch — same React Query key + loader as the warmer and taxonomy
  // so all consumers share a single cached fetch per session.
  const { data: allTagsData } = useQuery<TagConcept[], Error>(
    ['taxonomy', 'all-tags', initSessionId],
    loadAllTags,
    { enabled, staleTime: Infinity }
  );

  // Defensive filter: drop label-less tags; Fuse and the row renderers assume a
  // non-empty string and would otherwise throw.
  const allTags = useMemo(() => {
    if (!allTagsData) return [] as TagConcept[];
    return allTagsData.filter(
      (t) => t && typeof t.label === 'string' && t.label.length > 0
    );
  }, [allTagsData]);

  // Keep the shared index in sync with the loaded tag set (idempotent by ref).
  useEffect(() => {
    indexTags(allTags);
  }, [allTags]);

  const [results, setResults] = useState<TagConcept[]>([]);

  // Stable callback that guards against late results after unmount.
  const aliveRef = useRef(true);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);
  const setResultsSafe = useRef((items: TagConcept[]) => {
    if (aliveRef.current) setResults(items);
  });

  useEffect(() => {
    searchTags(
      enabled ? debouncedText : '',
      MAX_SEARCH_RESULTS,
      setResultsSafe.current
    );
  }, [debouncedText, enabled, allTags]);

  return { results };
}

export default useTagSearch;
