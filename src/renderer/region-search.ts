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

// Capture the region, POST it to the server's image-search endpoint, and
// dispatch the results into the machine. Throws on failure (caller toasts).
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
  const res = await fetch('http://localhost:8090/api/media/search/image', {
    method: 'POST',
    headers: {
      'Content-Type': 'image/png',
      Authorization: `Bearer ${authToken}`,
    },
    body: png,
  });
  if (!res.ok) {
    throw new Error(`Region search failed (HTTP ${res.status}). Is the media server running?`);
  }
  const items = await res.json();
  dispatch({ type: 'REGION_SEARCH_RESULTS', data: { items: items || [] } });
}
