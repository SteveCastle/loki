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
  | 'face'
  | 'faces'
  | 'dupe'
  | 'orientation';

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
  // 'faces' = face-presence filter; value 'ungrouped' = media holding at
  //   least one detected face not assigned to any person yet (the People
  //   panel's Ungrouped pool).
  // 'dupe' = duplicate-candidate group filter (the Duplicates panel): a group
  //   id (that group's active members), 'pending' (every item still waiting
  //   on review) or 'any' (every grouped item, dismissed groups included).
  // 'orientation' = dimension filter on media.width vs media.height; value
  //   'landscape' | 'portrait' | 'square'. Items without known dimensions
  //   never match an include and are kept by an exclude.
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
  // How a multi-node composite is scored server-side:
  //   'blend' (or undefined) = one combined centroid vector — ranks by the
  //     AVERAGE similarity to the components (historical behavior).
  //   'shared' = must-match-all — ranks by similarity to EVERY positive
  //     component at once, zeroing in on what they have in common (five
  //     football photos surface footballs, not one photo's grassy park).
  blendMode?: 'blend' | 'shared';
  // Similarity predicates ('similar'/'visual'/'clip'/'face'): how many of the
  // most similar items the search returns. Undefined = SIMILARITY_LIMIT (the
  // server's default); the chip's widen button adds SIMILARITY_LIMIT_STEP.
  limit?: number;
}

/** Results a similarity chip returns until widened (the server's default cap). */
export const SIMILARITY_LIMIT = 1000;
/** How much one press of a chip's widen button adds. */
export const SIMILARITY_LIMIT_STEP = 1000;
/** The server's ceiling for a widened chip. */
export const SIMILARITY_LIMIT_MAX = 1_000_000;

/** Whether a predicate is resolved by embedding similarity (and so has a result cap). */
export function isSimilarityPredicate(p: Predicate): boolean {
  return p.type === 'similar' || p.type === 'visual' || p.type === 'clip' || p.type === 'face';
}

/** A similarity predicate's result cap. */
export function similarityLimit(p: Predicate): number {
  return p.limit && p.limit > 0 ? p.limit : SIMILARITY_LIMIT;
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
