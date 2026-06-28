// src/renderer/query/types.ts
// Structured, composable media query shared by the chip input, the state
// machine, and the backend query engine. A query is a flat list of
// predicates joined by the global filtering mode (see settings.FilterModeOption).

export type PredicateType =
  | 'tag'
  | 'category'
  | 'path'
  | 'description'
  | 'hash'
  | 'similar'
  | 'visual';

export interface Predicate {
  type: PredicateType;
  // Exact value for 'tag'/'category'; substring for 'path'/'description'/'hash'.
  // 'similar' = a media path (find visually similar media via embedding backend).
  // 'visual' = free text query (text→image search via embedding backend).
  value: string;
  // Per-predicate include (false) / exclude (true).
  exclude: boolean;
  // Faceted combine bucket; undefined falls back to the global mode.
  join?: 'AND' | 'OR';
}

export interface Query {
  predicates: Predicate[];
}

export const EMPTY_QUERY: Query = { predicates: [] };

// Stable identity for a predicate, used for dedupe and React keys.
export function predicateKey(p: Predicate): string {
  return `${p.exclude ? '-' : ''}${p.type}:${p.value}`;
}

export function queryIsEmpty(q: Query): boolean {
  return !q || q.predicates.length === 0;
}
