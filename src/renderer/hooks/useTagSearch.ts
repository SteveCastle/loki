// useTagSearch — reusable tag type-ahead over the SHARED, pre-warmed indexes.
//
// Both the taxonomy sidebar and the command palette consume this hook. It does
// not own a worker or Fuse instance: all indexing + matching is routed through
// the module-level singleton in ../search/tag-search-service, so the whole app
// shares ONE worker.
//
// Callers pick a SCOPE (see ../search/tag-scopes). The palette asks for
// 'curated' and the sidebar for 'all'; each scope is fetched and indexed
// independently, so the palette never pays for the ~183K machine-suggested tags
// it doesn't show. Scope is part of the React Query key, so the two datasets
// cache side by side rather than evicting each other.
import { useContext, useEffect, useRef, useState } from 'react';
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
import {
  DEFAULT_TAG_SCOPE,
  applyScope,
  excludedCategories,
  type TagScope,
} from '../search/tag-scopes';
import { SEARCH_DEBOUNCE_MS } from '../search/search-config';

export type { TagConcept } from '../search/tag-search-service';
export type { TagScope } from '../search/tag-scopes';

// Mirrors the historical cap: each rendered result can trigger downstream IPC
// work, so we cap the match list even though Fuse ranks across the full set.
const MAX_SEARCH_RESULTS = 200;

// Deliberately thin rows: label + category + weight (see loadAllTags in
// src/main/taxonomy.ts and the /api/taxonomy/tags handler). This is a whole
// slice of the tag table on both hops — IPC/HTTP and then the worker's
// postMessage — so don't add fields here; fetch per-tag detail with `get-tag`.
//
// The scope's excluded categories are pushed DOWN to the backend (a SQL NOT IN
// on Electron, a query param on web) so they are never fetched or serialized;
// applyScope is the client-side backstop for a backend that ignores the hint.
export async function loadTagsForScope(scope: TagScope): Promise<TagConcept[]> {
  const excluded = excludedCategories(scope);
  const result = await invoke('load-all-tags', excluded.length ? [excluded] : []);
  return applyScope((result as TagConcept[]) ?? [], scope);
}

/** The shared React Query key for a scope, so every consumer hits one fetch. */
export function tagScopeQueryKey(scope: TagScope, initSessionId: string) {
  return ['taxonomy', 'all-tags', scope, initSessionId] as const;
}

/**
 * Off-thread fuzzy tag search backed by the shared singleton index.
 *
 * @param text    The raw (un-debounced) search text.
 * @param enabled Gates the active search so it only runs while the consuming
 *                surface wants results.
 * @param scope   Which slice of the tag table to search. Defaults to everything.
 */
export function useTagSearch(
  text: string,
  enabled: boolean,
  scope: TagScope = DEFAULT_TAG_SCOPE
): { results: TagConcept[] } {
  const { libraryService } = useContext(GlobalStateContext);
  const initSessionId = useSelector(
    libraryService,
    (state) => state.context.initSessionId
  );

  // Debounce the raw text into the value actually dispatched to search.
  const [debouncedText, setDebouncedText] = useState<string>('');
  const debouncedSet = useRef(
    debounce((value: string) => setDebouncedText(value), SEARCH_DEBOUNCE_MS)
  );
  useEffect(() => {
    debouncedSet.current(text);
  }, [text]);
  useEffect(() => {
    const d = debouncedSet.current;
    return () => d.cancel();
  }, []);

  // Scope-tag fetch — same key + loader as the warmer and taxonomy so all
  // consumers of a scope share a single cached fetch per session.
  const { data: allTagsData } = useQuery<TagConcept[], Error>(
    tagScopeQueryKey(scope, initSessionId),
    () => loadTagsForScope(scope),
    { enabled, staleTime: Infinity }
  );

  // Keep the shared index in sync. Index off the RAW React Query array (not a
  // locally-filtered copy): every consumer reads the same cache entry, so this
  // shared reference lets indexTags dedupe to a single worker clone instead of
  // re-cloning on this surface's first search.
  useEffect(() => {
    if (allTagsData) indexTags(allTagsData, scope);
  }, [allTagsData, scope]);

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
      setResultsSafe.current,
      scope
    );
  }, [debouncedText, enabled, allTagsData, scope]);

  return { results };
}

export default useTagSearch;
