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
  | 'visual'
  | 'clip'
  | 'face';

// One extra component of a composite similarity query, merged with the
// predicate's base value into a single query vector server-side:
//   'image' = another library item (media path)
//   'clip'  = another captured region (PNG data URL)
//   'text'  = a free-text concept
// Negative nodes steer the query AWAY from the concept ("… − 'blurry'").
export interface BlendNode {
  kind: 'image' | 'clip' | 'text';
  value: string;
  // 0..1 magnitude; undefined = 1 (full strength).
  weight?: number;
  negative?: boolean;
}

export interface Predicate {
  type: PredicateType;
  // Exact value for 'tag'/'category'; substring for 'path'/'description'/'hash'.
  // 'similar' = a media path (find visually similar media via embedding backend).
  // 'visual' = free text query (text→image search via embedding backend).
  // 'clip' = a captured screen region as a PNG data URL (image→image search
  //   via embedding backend); the data URL doubles as the chip thumbnail.
  // 'face' = person search by face identity: a media path ("find this person")
  //   or a captured region as a PNG data URL; matched against the face
  //   embedding index, not the whole-image one.
  value: string;
  // Per-predicate include (false) / exclude (true).
  exclude: boolean;
  // Faceted combine bucket; undefined falls back to the global mode.
  join?: 'AND' | 'OR';
  // Blended search ('similar'/'clip' only): free text mixed into the image
  // query. The server combines the image and text embeddings into one query
  // vector, so both live in the same SigLIP 2 space. Absent = pure image.
  // LEGACY single-text form — node mutations migrate it into `nodes`.
  text?: string;
  // Text share of the blend, 0..1 (0 = pure image, 1 = pure text). Only
  // meaningful when `text` is set; the UI defaults new blends to 0.5.
  textWeight?: number;
  // Composite similarity ('similar'/'clip'/'visual'): extra image/clip/text
  // nodes merged with the base value into ONE query vector server-side. The
  // chip popover manages these (add/remove text, stack images, negative
  // toggles) — a visual (text) chip composes exactly like an image chip.
  nodes?: BlendNode[];
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
