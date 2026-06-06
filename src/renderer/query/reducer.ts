// src/renderer/query/reducer.ts
import type { Predicate, Query } from './types';
import { predicateKey } from './types';

export function addPredicate(q: Query, p: Predicate): Query {
  const key = predicateKey(p);
  if (q.predicates.some((x) => predicateKey(x) === key)) return q;
  return { predicates: [...q.predicates, p] };
}

export function removePredicate(q: Query, key: string): Query {
  return { predicates: q.predicates.filter((x) => predicateKey(x) !== key) };
}

export function toggleExclude(q: Query, key: string): Query {
  return {
    predicates: q.predicates.map((x) =>
      predicateKey(x) === key ? { ...x, exclude: !x.exclude } : x
    ),
  };
}

// Legacy tag-click behavior unified into the query model.
// AND/OR: toggle the tag in place. EXCLUSIVE: clicking replaces the entire
// query with that single tag (or clears it if it was the only active tag).
export function applyTagClick(q: Query, tag: string, mode: string): Query {
  const join: 'AND' | 'OR' = mode === 'OR' ? 'OR' : 'AND';
  const key = predicateKey({ type: 'tag', value: tag, exclude: false });
  const exists = q.predicates.some((x) => predicateKey(x) === key);
  if (mode === 'EXCLUSIVE') {
    if (exists) return { predicates: [] };
    return { predicates: [{ type: 'tag', value: tag, exclude: false, join }] };
  }
  if (exists) return removePredicate(q, key);
  return addPredicate(q, { type: 'tag', value: tag, exclude: false, join });
}

export function setPredicateJoin(q: Query, key: string, join: 'AND' | 'OR'): Query {
  return {
    predicates: q.predicates.map((x) =>
      predicateKey(x) === key ? { ...x, join } : x
    ),
  };
}

// Project the active (included) tag values from a query — the legacy
// dbQuery.tags view derives from this so it can never desync from `query`.
export function tagsFromQuery(q: Query): string[] {
  return q.predicates
    .filter((p) => p.type === 'tag' && !p.exclude)
    .map((p) => p.value);
}
