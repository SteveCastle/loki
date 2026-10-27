// Scene layer references: extra images / videos / audio a layer's generation is
// given besides its own crop, and the tokens its prompt names them by. The
// numbering mirrors sceneRefTokens in media-server/tasks/scene.go:
//   Edit (loki-retouch):   <image1> = the crop, then <image2>, <image3>… for images
//   Motion (loki-reshoot): <Picture 1> = the crop, then <Picture 2>… for images,
//                          <Video 1>… for videos, <Audio N>… for audio, numbered
//                          after the dialogue voice when there is one (it is <Audio 1>)
import { mediaKind } from '../transform/intents';
import type { LayerKind, LayerOptions } from './scene-api';

export type RefKind = 'image' | 'video' | 'audio';

export interface SceneRefItem {
  /** Stable id, so prompt tokens can follow a reference when the list changes. */
  id: string;
  /** The file ('' for a paint layer). */
  path: string;
  /** A paint layer of the scene instead of a file: the server renders it over the picture, aligned with the crop. */
  layer?: string;
  /** What it is, in the user's words (motion prompts define the reference with it). */
  note?: string;
}

export function refKind(path: string): RefKind | null {
  const k = mediaKind(path);
  return k === 'image' || k === 'video' || k === 'audio' ? k : null;
}

/** What kind of reference an item is (a paint layer is an image). */
export function itemKind(r: SceneRefItem): RefKind | null {
  return r.layer ? 'image' : refKind(r.path);
}

/** The reference kinds a layer can use. */
export function refKinds(kind: LayerKind): RefKind[] {
  return kind === 'motion' ? ['image', 'video', 'audio'] : ['image'];
}

/** The token for the crop itself. */
export function selfToken(kind: LayerKind): string {
  return kind === 'motion' ? '<Picture 1>' : '<image1>';
}

/** Whether a dialogue voice takes <Audio 1>. */
export function hasVoice(o: LayerOptions | undefined): boolean {
  return !!o?.voiceRef && (o.dialogue || []).some((l) => l.text.trim());
}

/** Each reference's token (null = this layer can't use it). */
export function refTokens(kind: LayerKind, refs: SceneRefItem[], voice: boolean): Array<string | null> {
  let img = 2;
  let vid = 1;
  let aud = voice ? 2 : 1;
  return refs.map((r) => {
    const k = itemKind(r);
    if (kind !== 'motion') {
      if (k !== 'image') return null;
      img += 1;
      return `<image${img - 1}>`;
    }
    if (k === 'image') {
      img += 1;
      return `<Picture ${img - 1}>`;
    }
    if (k === 'video') {
      vid += 1;
      return `<Video ${vid - 1}>`;
    }
    if (k === 'audio') {
      aud += 1;
      return `<Audio ${aud - 1}>`;
    }
    return null;
  });
}

const TOKEN_RE = /<image\d+>|<(?:Picture|Video|Audio) \d+>/gi;
const REMOVED_RE = /<removed: [^>]*>/g;

/** Tokens in a prompt that name nothing this layer has (a removed reference, a typo, the other engine's style). */
export function tokenIssues(prompt: string, kind: LayerKind, refs: SceneRefItem[], voice: boolean): string[] {
  const valid = new Set<string>([selfToken(kind)]);
  if (kind === 'motion' && voice) valid.add('<Audio 1>');
  refTokens(kind, refs, voice).forEach((t) => t && valid.add(t));
  const bad = new Set<string>();
  (prompt.match(TOKEN_RE) || []).forEach((t) => {
    if (!valid.has(t)) bad.add(t);
  });
  (prompt.match(REMOVED_RE) || []).forEach((t) => bad.add(t));
  return [...bad];
}

/**
 * Renumber a prompt's tokens after the reference list changed (removed, reordered): every reference keeps the
 * words that named it. A removed reference's token becomes <removed: its name>, which tokenIssues flags: left as a
 * number it could silently name whichever reference took that number.
 */
export function remapRefTokens(prompt: string, kind: LayerKind, before: SceneRefItem[], after: SceneRefItem[], voice: boolean): string {
  const oldT = refTokens(kind, before, voice);
  const newById = new Map<string, string | null>();
  const nt = refTokens(kind, after, voice);
  after.forEach((r, i) => newById.set(r.id, nt[i]));
  const map = new Map<string, string>();
  before.forEach((r, i) => {
    const o = oldT[i];
    if (!o) return;
    if (!newById.has(r.id)) {
      map.set(o, `<removed: ${(r.note || '').trim() || (r.layer ? 'paint layer' : '') || r.path.split(/[\\/]/).pop()}>`);
      return;
    }
    const n = newById.get(r.id);
    if (n && o !== n) map.set(o, n);
  });
  if (!map.size) return prompt;
  return prompt.replace(TOKEN_RE, (t) => map.get(t) ?? t);
}

export function newRefId(): string {
  return `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}
