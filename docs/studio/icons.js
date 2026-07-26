/*
 * slangfx studio — layer-type icons.
 *
 * One small set, shared by every place a layer announces what it is: the
 * ＋ Layer menu, the inspector's title badge, and the clip chips in the
 * timeline. They're drawn to survive 11–16 px, which is the only size
 * they're ever used at:
 *
 *   - a 16×16 box with a 1px safety margin, so nothing clips when the
 *     browser rounds a fractional layout position
 *   - one idea per icon, no incidental detail (a film frame gets sprocket
 *     holes OR a play mark, not both)
 *   - 1.5px strokes on a 16px grid, round joins, `currentColor` throughout
 *     so each context colours them by inheritance
 *   - silhouettes that stay distinct when they blur together: a wide flat
 *     rectangle (media), stacked bars (audio), a split disc (adjustment),
 *     a letterform (text), two overlapping outlines (shape)
 */

const svg = (body, extra = '') =>
  `<svg class="lyr-ico" viewBox="0 0 16 16" width="16" height="16" fill="none"
     stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
     stroke-linejoin="round" aria-hidden="true"${extra}>${body}</svg>`;

export const LAYER_ICONS = {
  /* Video: a frame with a play mark — the frame alone reads as "image". */
  video: svg(`
    <rect x="1.6" y="3.1" width="12.8" height="9.8" rx="2"/>
    <path d="M6.6 6.2 L10.6 8 L6.6 9.8 Z" fill="currentColor" stroke="none"/>`),

  /* Still: the universal picture — horizon, sun, frame. */
  image: svg(`
    <rect x="1.6" y="3.1" width="12.8" height="9.8" rx="2"/>
    <circle cx="5.6" cy="6.5" r="1.15" fill="currentColor" stroke="none"/>
    <path d="M2.4 11.6 L6.1 8.4 L8.7 10.7 L10.7 9 L13.6 11.6"/>`),

  /* Sound: a level meter. Bars survive small sizes where a quaver's
     tail and notehead merge into a smudge. */
  audio: svg(`
    <path d="M3 6.6 V9.4 M5.8 4.4 V11.6 M8.5 2.6 V13.4 M11.2 5.2 V10.8 M14 7 V9"/>`),

  /* Adjustment layer: a disc half-filled — the "affects what's below"
     mark, and the one icon in the set with a solid mass. */
  fx: svg(`
    <circle cx="8" cy="8" r="5.6"/>
    <path d="M8 2.4 A5.6 5.6 0 0 0 8 13.6 Z" fill="currentColor" stroke="none"/>`),

  /* Text: a letterform, drawn heavier because it IS the icon. */
  text: svg('<path d="M3.6 4.2 H12.4 M8 4.2 V12.4" stroke-width="1.9"/>'),

  /* Shape: two outlines overlapping — a family, not one particular shape. */
  shape: svg(`
    <rect x="2" y="2.2" width="8" height="8" rx="1.4"/>
    <circle cx="10.4" cy="10.2" r="3.6"/>`),

  /* Import: something arriving into a tray. */
  import: svg(`
    <path d="M8 2.4 V9.6 M5.2 6.9 L8 9.7 L10.8 6.9"/>
    <path d="M2.6 10.9 V13 H13.4 V10.9"/>`),
};

/** The icon for a clip, given the asset behind it (may be null/offline). */
export function clipIcon(clip, asset) {
  if (clip.kind === 'fx') return LAYER_ICONS.fx;
  if (clip.kind === 'audio') return LAYER_ICONS.audio;
  if (Array.isArray(clip.shapes) && clip.shapes.length) return LAYER_ICONS.shape;
  if (asset?.kind === 'audio') return LAYER_ICONS.audio;
  return asset?.kind === 'image' || asset?.kind === 'gif'
    ? LAYER_ICONS.image
    : LAYER_ICONS.video;
}

/**
 * A shape preset's icon, drawn from the preset's own path function — so
 * the picker shows the actual geometry rather than an approximate glyph,
 * and a new preset gets a correct icon for free.
 * @param {(c, x, y, w, h) => void} path  the preset's canvas path builder
 */
export function shapeIconCanvas(path, px = 15) {
  const c = document.createElement('canvas');
  const dpr = devicePixelRatio || 1;
  c.width = Math.round(px * dpr);
  c.height = Math.round(px * dpr);
  c.style.width = `${px}px`;
  c.style.height = `${px}px`;
  c.className = 'shape-ico';
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.beginPath();
  const pad = 1.6;
  path(ctx, pad, pad, px - pad * 2, px - pad * 2);
  ctx.fillStyle = 'currentColor';
  // Canvas can't resolve currentColor, so the caller's colour is applied
  // by the CSS filter on .shape-ico; draw in white and let opacity carry.
  ctx.fillStyle = '#dfe4ec';
  ctx.fill();
  return c;
}
