// src/renderer/query/reducer.ts
import type { BlendNode, Predicate, Query } from './types';
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
  const target = q.predicates.find((x) => predicateKey(x) === key);
  if (!target) return q;
  // Guard: never produce an all-exclude query (no positive/include anchor).
  // Such a query has nothing to drive from and scans the entire media table
  // ("everything except X"), which can hang/crash the app on a large library.
  // Block toggling the last remaining include predicate to exclude.
  if (!target.exclude) {
    const hasOtherInclude = q.predicates.some(
      (x) => predicateKey(x) !== key && !x.exclude
    );
    if (!hasOtherInclude) return q; // would leave zero includes — no-op
  }
  return {
    predicates: q.predicates.map((x) =>
      predicateKey(x) === key ? { ...x, exclude: !x.exclude } : x
    ),
  };
}

// A tag was renamed server-side (person rename cascades to taxonomy rows):
// any tag chip filtering on the old label must follow it, or the active view
// silently becomes "matches nothing". Returns the SAME query object when no
// chip referenced the old name, so callers can skip the re-query.
export function renameTagPredicate(q: Query, from: string, to: string): Query {
  if (!from || !to || from === to) return q;
  let changed = false;
  const predicates = q.predicates.map((p) => {
    if (p.type === 'tag' && p.value === from) {
      changed = true;
      return { ...p, value: to };
    }
    return p;
  });
  return changed ? { predicates } : q;
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

// Adding a predicate honors EXCLUSIVE mode: in EXCLUSIVE, selecting ANY filter
// (tag, path, category, description, hash) replaces the entire query with just
// that predicate. In AND/OR it appends (deduped).
export function addPredicateWithMode(
  q: Query,
  p: Predicate,
  mode: string
): Query {
  if (mode === 'EXCLUSIVE') return { predicates: [p] };
  return addPredicate(q, p);
}

// Adding an EMBEDDING-SPACE predicate (similar/clip image or visual text)
// treats the filtering mode specially so components can be STACKED into one
// latent-space query: EXCLUSIVE replaces the whole query (the classic
// find-similar), but in AND/OR, when an embedding predicate already exists,
// the new image/text is merged into it as a blend node — the server combines
// the vectors into one — instead of intersecting independent top-N path sets
// (which mostly returns nothing). Works in every direction: image onto text,
// text onto image, image onto image. Face predicates never merge (different
// embedding space).
const EMBEDDING_TYPES = ['similar', 'clip', 'visual'] as const;

function isEmbeddingType(t: Predicate['type']): boolean {
  return (EMBEDDING_TYPES as readonly string[]).includes(t);
}

export function addOrMergeSimilarityPredicate(
  q: Query,
  p: Predicate,
  mode: string
): Query {
  if (mode === 'EXCLUSIVE' || !isEmbeddingType(p.type)) {
    return addPredicateWithMode(q, p, mode);
  }
  const target = q.predicates.find(
    (x) => isEmbeddingType(x.type) && !x.exclude
  );
  if (!target) return addPredicate(q, p);
  const node: BlendNode =
    p.type === 'visual'
      ? // Text joins at half strength (same default as the popover's add-
        // concept input) so it steers rather than overwhelms the base.
        { kind: 'text', value: p.value, weight: 0.5 }
      : { kind: p.type === 'similar' ? 'image' : 'clip', value: p.value };
  return addBlendNode(q, predicateKey(target), node);
}

// migrateLegacyBlend folds the old single-text blend fields into `nodes` so
// every node mutation works on one uniform list. Index 0 is always the legacy
// text when it existed, matching what the popover renders.
function migrateLegacyBlend(p: Predicate): Predicate {
  if (!p.text || !p.text.trim()) return p;
  const next: Predicate = {
    ...p,
    nodes: [
      { kind: 'text', value: p.text.trim(), weight: p.textWeight ?? 0.5 },
      ...(p.nodes ?? []),
    ],
  };
  delete next.text;
  delete next.textWeight;
  return next;
}

// effectiveBlendNodes is the uniform node view of a predicate for rendering:
// the migrated legacy text (if any) followed by the explicit nodes.
export function effectiveBlendNodes(p: Predicate): BlendNode[] {
  return migrateLegacyBlend(p).nodes ?? [];
}

// Append a blend node to the similarity predicate at `key` (deduped against
// the base value and existing nodes).
export function addBlendNode(q: Query, key: string, node: BlendNode): Query {
  return {
    predicates: q.predicates.map((x) => {
      if (predicateKey(x) !== key) return x;
      const p = migrateLegacyBlend(x);
      if (p.value === node.value) return p; // already the base image
      const nodes = p.nodes ?? [];
      if (nodes.some((n) => n.kind === node.kind && n.value === node.value)) {
        return p;
      }
      return { ...p, nodes: [...nodes, node] };
    }),
  };
}

// Remove the blend node at `index` (indices match effectiveBlendNodes).
export function removeBlendNode(q: Query, key: string, index: number): Query {
  return {
    predicates: q.predicates.map((x) => {
      if (predicateKey(x) !== key) return x;
      const p = migrateLegacyBlend(x);
      const nodes = (p.nodes ?? []).filter((_, i) => i !== index);
      if (nodes.length === 0) {
        const rest: Predicate = { ...p };
        delete rest.nodes;
        return rest;
      }
      return { ...p, nodes };
    }),
  };
}

// Patch the blend node at `index` (weight / negative toggle).
export function updateBlendNode(
  q: Query,
  key: string,
  index: number,
  patch: Partial<Pick<BlendNode, 'weight' | 'negative' | 'value'>>
): Query {
  return {
    predicates: q.predicates.map((x) => {
      if (predicateKey(x) !== key) return x;
      const p = migrateLegacyBlend(x);
      return {
        ...p,
        nodes: (p.nodes ?? []).map((n, i) =>
          i === index ? { ...n, ...patch } : n
        ),
      };
    }),
  };
}

// Patch a predicate's blend fields (text / textWeight) in place, keyed by
// predicateKey (which ignores blend fields, so the chip identity is stable
// while its blend is edited). Clearing the text drops the whole blend.
export function updatePredicateBlend(
  q: Query,
  key: string,
  patch: { text?: string; textWeight?: number }
): Query {
  return {
    predicates: q.predicates.map((x) => {
      if (predicateKey(x) !== key) return x;
      const next = { ...x, ...patch };
      if (!next.text || !next.text.trim()) {
        delete next.text;
        delete next.textWeight;
      }
      return next;
    }),
  };
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
