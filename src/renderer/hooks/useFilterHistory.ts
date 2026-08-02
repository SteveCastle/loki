import { useCallback } from 'react';
import { useSelector } from '@xstate/react';
import type { QueryHistoryEntry } from '../query/history';

// Wires the state machine's session filter history to QueryInput's dropdown:
// the recorded query states, the pinned base row's label/count, and the two
// apply handlers. Applying a history entry RE-RUNS its query
// (APPLY_QUERY_STATE); applying the base clears the query, which restores the
// in-memory folder-scan snapshot (CLEAR_QUERY → loadingFromPreviousLibrary).
// Shared by the taxonomy sidebar and the command palette so both dropdowns
// stay in lockstep.
export function useFilterHistory(libraryService: any) {
  const history = useSelector(
    libraryService,
    (s: any) => s.context.queryHistory ?? []
  ) as QueryHistoryEntry[];

  // In FS mode the base is the CURRENT view; in DB mode it's the snapshot the
  // queries started from. No folder at all (web full-library sessions) reads
  // as "All media".
  const baseLabel = useSelector(libraryService, (s: any) => {
    const c = s.context;
    const file =
      (c.currentStateType === 'fs'
        ? c.initialFile
        : c.previousInitialFile || c.initialFile) || '';
    if (!file) return 'All media';
    return file.split(/[/\\]/).filter(Boolean).pop() || file;
  }) as string;

  const baseCount = useSelector(libraryService, (s: any) => {
    const c = s.context;
    return c.currentStateType === 'fs'
      ? c.library?.length ?? 0
      : c.previousLibrary?.length ?? 0;
  }) as number;

  const onApplyHistory = useCallback(
    (entry: QueryHistoryEntry) => {
      libraryService.send({
        type: 'APPLY_QUERY_STATE',
        data: { predicates: entry.query.predicates },
      });
    },
    [libraryService]
  );

  const onApplyBase = useCallback(() => {
    libraryService.send({ type: 'CLEAR_QUERY' });
  }, [libraryService]);

  return { history, baseLabel, baseCount, onApplyHistory, onApplyBase };
}
