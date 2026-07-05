import { captureRegion } from './platform';

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Normalize a drag (start/end in client coords, any direction) into a rect.
export function rectFromDrag(
  start: { x: number; y: number },
  end: { x: number; y: number }
): Rect {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  return {
    x,
    y,
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function pngToDataUrl(png: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000; // avoid call-stack limits on large captures
  for (let i = 0; i < png.length; i += CHUNK) {
    binary += String.fromCharCode(...png.subarray(i, i + CHUNK));
  }
  return `data:image/png;base64,${btoa(binary)}`;
}

// Capture the region and commit it as a `clip` predicate on the unified query.
// The query pipeline resolves it via the embedding backend exactly like a
// `similar:` predicate, so the search shows up as a chip (with the clip as its
// thumbnail), composes with other predicates, and is removable/restorable like
// any other filter. Throws on failure (caller toasts).
export async function runRegionSearch(
  rect: Rect,
  authToken: string | null,
  dispatch: (event: any) => void
): Promise<void> {
  if (!captureRegion) {
    throw new Error('Region capture is only available in the desktop app.');
  }
  if (!authToken) {
    throw new Error('Visual search requires logging in to the local media server.');
  }
  const png = await captureRegion(rect);
  if (!png || png.length === 0) {
    throw new Error('Capture failed.');
  }
  dispatch({
    type: 'ADD_PREDICATE',
    data: {
      predicate: { type: 'clip', value: pngToDataUrl(png), exclude: false },
    },
  });
}
