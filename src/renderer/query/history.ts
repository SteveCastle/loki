// src/renderer/query/history.ts
// Session history of committed filter states. The state machine records one
// entry per successfully-loaded DB query (see loadedFromDB's entry action);
// the QueryInput dropdown renders them and re-applies one by RE-RUNNING the
// query (APPLY_QUERY_STATE) — results are never cached here, only the state
// needed to reproduce them. The filesystem base state is NOT an entry: it is
// the machine's previousLibrary snapshot, pinned separately in the dropdown,
// because a folder scan is the one load too slow to just re-run.
import type { Query, Predicate, BlendNode } from './types';

export interface QueryHistoryEntry {
  // Canonical identity of the filter state (queryStateKey). Re-running a
  // state already in history moves it to the top instead of duplicating it.
  key: string;
  // Deep-copied snapshot — later chip edits must not mutate history.
  query: Query;
  // Result count when this state last ran, shown in the dropdown.
  count: number;
  // Timestamp of the last run, for the dropdown's "how stale" hint.
  at: number;
}

// Cap keeps the dropdown scannable; the FS base is pinned outside the cap.
export const MAX_QUERY_HISTORY = 5;

// Canonical identity for a filter state: every field that changes what a run
// returns (type/value/exclude/join, legacy text blends, composite nodes).
// Order-sensitive on purpose — predicate order changes left-to-right joins.
export function queryStateKey(q: Query): string {
  return JSON.stringify(
    (q?.predicates ?? []).map((p) => [
      p.type,
      p.value,
      !!p.exclude,
      p.join ?? '',
      p.text ?? '',
      p.textWeight ?? null,
      (p.nodes ?? []).map((n) => [n.kind, n.value, n.weight ?? 1, !!n.negative]),
      p.blendMode ?? 'blend',
    ])
  );
}

function cloneNode(n: BlendNode): BlendNode {
  return { ...n };
}

function clonePredicate(p: Predicate): Predicate {
  return { ...p, nodes: p.nodes ? p.nodes.map(cloneNode) : undefined };
}

export function cloneQuery(q: Query): Query {
  return { predicates: (q?.predicates ?? []).map(clonePredicate) };
}

// Record a committed query state: dedupe by key (a re-run refreshes count and
// timestamp and moves the entry to the top), cap at MAX_QUERY_HISTORY. Empty
// queries are never recorded — the empty state IS the base, handled separately.
export function pushQueryHistory(
  history: QueryHistoryEntry[],
  query: Query,
  count: number,
  at: number = Date.now()
): QueryHistoryEntry[] {
  if (!query || query.predicates.length === 0) return history;
  const key = queryStateKey(query);
  const entry: QueryHistoryEntry = { key, query: cloneQuery(query), count, at };
  return [entry, ...history.filter((e) => e.key !== key)].slice(
    0,
    MAX_QUERY_HISTORY
  );
}
