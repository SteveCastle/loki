// useTagSearch — reusable, off-thread tag type-ahead.
//
// Encapsulates the worker + Fuse-fallback + debounce tag search that lives
// inline in components/taxonomy/taxonomy.tsx. Lifted faithfully so the command
// palette can reuse it without depending on the taxonomy panel. The taxonomy
// panel intentionally keeps its own copy (to avoid regressions); a duplicate
// worker instance per consumer is fine.
import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery } from '@tanstack/react-query';
import { debounce } from 'lodash';
import Fuse from 'fuse.js';
import { GlobalStateContext } from '../state';
import { invoke } from '../platform';

export interface TagConcept {
  label: string;
  category: string;
  weight: number;
  description?: string;
}

// Mirrors taxonomy.tsx: each rendered result can trigger downstream IPC work,
// so we cap the match list even though Fuse ranks across the full set.
const MAX_SEARCH_RESULTS = 200;

async function loadAllTags(): Promise<TagConcept[]> {
  const result = await invoke('load-all-tags', []);
  return (result as TagConcept[]) ?? [];
}

/**
 * Off-thread fuzzy tag search.
 *
 * @param text    The raw (un-debounced) search text.
 * @param enabled Gates the all-tags fetch + search so it only runs while the
 *                consuming surface is active.
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

  // Debounce the raw text into the value actually dispatched to the worker.
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

  // Lazy full-tag fetch — same React Query key + loader as taxonomy.tsx so the
  // two consumers share a single cached fetch per session.
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

  const workerRef = useRef<Worker | null>(null);
  const searchSeq = useRef(0);
  const [results, setResults] = useState<TagConcept[]>([]);
  // Optimistically assume worker support; flip to false if construction fails
  // (e.g. jsdom under tests) so the synchronous fallback takes over.
  const [workerReady, setWorkerReady] = useState<boolean>(
    typeof Worker !== 'undefined'
  );

  useEffect(() => {
    if (typeof Worker === 'undefined') {
      setWorkerReady(false);
      return undefined;
    }
    let worker: Worker;
    try {
      worker = new Worker(
        new URL('../components/taxonomy/tag-search.worker.ts', import.meta.url)
      );
    } catch {
      setWorkerReady(false);
      return undefined;
    }
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg?.type === 'result' && msg.id === searchSeq.current) {
        setResults(msg.items as TagConcept[]);
      }
    };
    workerRef.current = worker;
    setWorkerReady(true);
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Keep the worker index synced with the loaded tag set. The worker re-runs
  // the outstanding query after re-indexing, so results refresh automatically.
  useEffect(() => {
    workerRef.current?.postMessage({ type: 'index', tags: allTags });
  }, [allTags]);

  // Synchronous Fuse fallback — only built when no worker is available.
  // Keep these options in sync with tag-search.worker.ts.
  const fallbackFuse = useMemo(() => {
    if (workerReady) return null;
    return new Fuse(allTags, {
      keys: [
        { name: 'label', weight: 2 },
        { name: 'category', weight: 1 },
      ],
      threshold: 0.4,
      ignoreLocation: true,
      minMatchCharLength: 1,
    });
  }, [allTags, workerReady]);

  useEffect(() => {
    const id = (searchSeq.current += 1);
    const activeQuery = enabled ? debouncedText : '';
    if (!activeQuery) {
      setResults([]);
    }
    if (workerRef.current) {
      workerRef.current.postMessage({
        type: 'search',
        id,
        query: activeQuery,
        limit: MAX_SEARCH_RESULTS,
      });
    } else if (fallbackFuse && activeQuery) {
      setResults(
        fallbackFuse
          .search(activeQuery, { limit: MAX_SEARCH_RESULTS })
          .map((r) => r.item)
      );
    }
  }, [debouncedText, enabled, fallbackFuse]);

  return { results };
}

export default useTagSearch;
