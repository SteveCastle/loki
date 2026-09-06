import type { ScaleModeOption } from '../../../settings';

// Touch handling for the detail view: one finger pans the media, two fingers
// pinch-zoom it around the point between them.
//
// Why not native touch scrolling for the pan: Safari treats any multi-finger
// drag as a pan of the nearest scroller and cancels the pointer stream the
// moment it starts, so a pinch never gets to run alongside it. The container
// has `touch-action: none` and this module does both gestures itself. Panning
// writes scrollLeft/scrollTop directly — the same scroll model the mouse-drag
// pan and the remembered pan fraction in detail.tsx already use, so nothing
// else in the view needs to know touch exists.
//
// Zoom is the existing numeric scale mode ("Zoom" in the settings list: the
// media's height as a percentage of the container). During the gesture the
// media element is resized imperatively — a React render per frame would
// mean a settings write per frame, and in Electron each of those is a
// synchronous IPC. When the last finger lifts, the reached size is committed
// once, through the caller, as a scale mode: a percentage when zoomed in, or
// `fit` when pinched down to (or below) the size that shows the whole thing.

type Point = { x: number; y: number };

// Pinching further out than fit is allowed while the fingers are down so the
// gesture doesn't feel clamped; the commit snaps it back to fit.
const MIN_ZOOM_OF_FIT = 0.5;
// Zoom ceiling: the smaller of 4× the container height and 2× the media's
// native height (never below fit). Past 2× native the picture is only blur,
// and an image or video laid out at many thousands of pixels a side is a
// huge composited layer — iPad Safari runs out of GPU memory and stalls.
const MAX_HEIGHT_OF_CONTAINER = 4;
const MAX_ZOOM_OF_NATIVE = 2;
// Sizes within this of fit commit as `fit` rather than as a percentage that
// happens to equal it — so the next item fits too, instead of inheriting an
// aspect-specific number.
const FIT_SNAP = 1.03;

// Fling: velocity in px/ms decays by this factor every 16ms; stops below the
// floor. Tuned to feel like a native scroller, not to match one exactly.
const FLING_DECAY_PER_FRAME = 0.95;
const FLING_MIN_VELOCITY = 0.02;
const FLING_START_VELOCITY = 0.15;

type Media = HTMLImageElement | HTMLVideoElement;

function isVisualMedia(el: Element | null): el is Media {
  return el instanceof HTMLImageElement || el instanceof HTMLVideoElement;
}

function nativeSize(media: Media): { w: number; h: number } {
  return media instanceof HTMLVideoElement
    ? { w: media.videoWidth, h: media.videoHeight }
    : { w: media.naturalWidth, h: media.naturalHeight };
}

// The rectangle the pixels actually occupy, in the container's scroll
// coordinate space. In `fit` mode the element box is the whole container
// with the picture letterboxed inside it by object-fit, so the box is not the
// picture; every other mode gives the element an aspect-correct box.
function visualRect(media: Media, aspect: number) {
  const boxW = media.offsetWidth;
  const boxH = media.offsetHeight;
  const h = Math.min(boxH, boxW * aspect);
  const w = h / aspect;
  return {
    left: media.offsetLeft + (boxW - w) / 2,
    top: media.offsetTop + (boxH - h) / 2,
    w,
    h,
  };
}

function fitHeight(container: HTMLElement, aspect: number): number {
  return Math.min(container.clientHeight, container.clientWidth * aspect);
}

function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function mid(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export function attachTouchGestures(
  container: HTMLElement,
  getMedia: () => Element | null,
  commitScaleMode: (mode: ScaleModeOption) => void
): () => void {
  const pointers = new Map<number, Point>();
  let mode: 'idle' | 'pan' | 'pinch' = 'idle';

  // Pan state.
  let last: Point = { x: 0, y: 0 };
  let lastT = 0;
  let velocity: Point = { x: 0, y: 0 };
  let flingFrame: number | null = null;

  // Pinch state.
  let pinch: {
    media: Media;
    aspect: number;
    nativeH: number;
    startDist: number;
    startH: number;
    prevMid: Point;
  } | null = null;
  // Set once a pinch has resized the media; the commit only happens then, so
  // a two-finger tap (the context palette press) never rewrites the setting.
  let zoomDirty = false;

  const stopFling = () => {
    if (flingFrame !== null) {
      cancelAnimationFrame(flingFrame);
      flingFrame = null;
    }
  };

  const startFling = () => {
    if (
      Math.abs(velocity.x) < FLING_START_VELOCITY &&
      Math.abs(velocity.y) < FLING_START_VELOCITY
    ) {
      return;
    }
    let prev = performance.now();
    const step = (now: number) => {
      const dt = now - prev;
      prev = now;
      const beforeX = container.scrollLeft;
      const beforeY = container.scrollTop;
      container.scrollLeft -= velocity.x * dt;
      container.scrollTop -= velocity.y * dt;
      // Hitting an edge kills that axis so the fling doesn't "wait" at the
      // boundary until it decays.
      if (container.scrollLeft === beforeX) velocity.x = 0;
      if (container.scrollTop === beforeY) velocity.y = 0;
      const decay = Math.pow(FLING_DECAY_PER_FRAME, dt / 16);
      velocity = { x: velocity.x * decay, y: velocity.y * decay };
      if (
        Math.abs(velocity.x) > FLING_MIN_VELOCITY ||
        Math.abs(velocity.y) > FLING_MIN_VELOCITY
      ) {
        flingFrame = requestAnimationFrame(step);
      } else {
        flingFrame = null;
      }
    };
    flingFrame = requestAnimationFrame(step);
  };

  const beginPan = (p: Point) => {
    mode = 'pan';
    last = p;
    lastT = performance.now();
    velocity = { x: 0, y: 0 };
  };

  const beginPinch = (a: Point, b: Point) => {
    const media = getMedia();
    if (!isVisualMedia(media)) {
      mode = 'idle';
      return;
    }
    const { w, h } = nativeSize(media);
    if (!w || !h) {
      // Not loaded yet; nothing sensible to scale.
      mode = 'idle';
      return;
    }
    const aspect = h / w;
    const startH = visualRect(media, aspect).h;
    const startDist = Math.max(dist(a, b), 1);
    pinch = {
      media,
      aspect,
      nativeH: h,
      startDist,
      startH,
      prevMid: mid(a, b),
    };
    mode = 'pinch';
  };

  const applyPinch = (a: Point, b: Point) => {
    if (!pinch) return;
    const { media, aspect, nativeH, startDist, startH } = pinch;
    const fitH = fitHeight(container, aspect);
    const minH = fitH * MIN_ZOOM_OF_FIT;
    const maxH = Math.max(
      fitH,
      Math.min(
        container.clientHeight * MAX_HEIGHT_OF_CONTAINER,
        nativeH * MAX_ZOOM_OF_NATIVE
      )
    );
    const newH = Math.min(maxH, Math.max(minH, (startH * dist(a, b)) / startDist));
    const newW = newH / aspect;

    // Where, as a fraction of the picture, the point between the fingers was
    // before this frame's resize — measured against the previous midpoint so
    // that moving both fingers together also pans.
    const rect = container.getBoundingClientRect();
    const before = visualRect(media, aspect);
    const fx = pinch.prevMid.x - rect.left;
    const fy = pinch.prevMid.y - rect.top;
    const relX = (container.scrollLeft + fx - before.left) / before.w;
    const relY = (container.scrollTop + fy - before.top) / before.h;

    media.style.width = `${newW}px`;
    media.style.height = `${newH}px`;
    zoomDirty = true;

    // Keep that same point of the picture under the (current) midpoint.
    const now = mid(a, b);
    const after = visualRect(media, aspect);
    container.scrollLeft = after.left + relX * after.w - (now.x - rect.left);
    container.scrollTop = after.top + relY * after.h - (now.y - rect.top);
    pinch.prevMid = now;
  };

  const commitPinch = () => {
    if (!pinch) return;
    const { media, aspect } = pinch;
    pinch = null;
    if (!zoomDirty) return;
    zoomDirty = false;
    const h = visualRect(media, aspect).h;
    // The inline size set during the gesture is replaced by exactly what the
    // committed scale mode renders as, so React's re-render changes nothing
    // visible. Width goes back to auto: a leftover pixel width would fight
    // the percentage height and letterbox the picture.
    media.style.width = '';
    if (h <= fitHeight(container, aspect) * FIT_SNAP) {
      media.style.height = '';
      commitScaleMode('fit');
      return;
    }
    const pct = Math.max(1, Math.round((h / container.clientHeight) * 100));
    media.style.height = `${pct}%`;
    commitScaleMode(pct);
  };

  const isTouchLike = (e: PointerEvent) =>
    e.pointerType === 'touch' || e.pointerType === 'pen';

  const onPointerDown = (e: PointerEvent) => {
    if (!isTouchLike(e)) return;
    stopFling();
    if (e.isPrimary) {
      // New sequence: forget any finger whose pointerup never arrived (its
      // target was re-rendered away mid-touch). A ghost finger would make
      // the next single-finger drag a pinch against a stale point.
      pointers.clear();
      pinch = null;
      mode = 'idle';
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      beginPan({ x: e.clientX, y: e.clientY });
    } else if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      beginPinch(a, b);
    } else {
      // Three or more: ignore until they settle back down.
      mode = 'idle';
      pinch = null;
    }
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!pointers.has(e.pointerId)) return;
    const p = { x: e.clientX, y: e.clientY };
    pointers.set(e.pointerId, p);
    if (mode === 'pan' && pointers.size === 1) {
      const now = performance.now();
      const dt = Math.max(now - lastT, 1);
      const dx = p.x - last.x;
      const dy = p.y - last.y;
      container.scrollLeft -= dx;
      container.scrollTop -= dy;
      // Smoothed so a jittery last sample doesn't decide the fling.
      velocity = {
        x: velocity.x * 0.6 + (dx / dt) * 0.4,
        y: velocity.y * 0.6 + (dy / dt) * 0.4,
      };
      last = p;
      lastT = now;
    } else if (mode === 'pinch' && pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      applyPinch(a, b);
    }
  };

  const onPointerEnd = (e: PointerEvent) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (mode === 'pinch') {
      commitPinch();
      if (pointers.size === 1) {
        // Lifting one finger of a pinch continues as a pan with the other,
        // with no fling from the pinch.
        const [p] = Array.from(pointers.values());
        beginPan(p);
      } else {
        mode = 'idle';
      }
      return;
    }
    if (mode === 'pan' && pointers.size === 0) {
      mode = 'idle';
      // A cancelled pointer (the browser took it) has no meaningful release
      // velocity.
      if (e.type === 'pointerup' && performance.now() - lastT < 80) {
        startFling();
      }
      return;
    }
    if (pointers.size === 1) {
      const [p] = Array.from(pointers.values());
      beginPan(p);
    } else if (pointers.size === 0) {
      mode = 'idle';
    }
  };

  // Safari's page-zoom gesture events, in case touch-action alone isn't
  // honoured; the pinch above is the only zoom this surface wants.
  const blockGesture = (e: Event) => e.preventDefault();

  container.addEventListener('pointerdown', onPointerDown);
  container.addEventListener('pointermove', onPointerMove);
  container.addEventListener('pointerup', onPointerEnd);
  container.addEventListener('pointercancel', onPointerEnd);
  container.addEventListener('gesturestart', blockGesture, { passive: false });

  return () => {
    stopFling();
    container.removeEventListener('pointerdown', onPointerDown);
    container.removeEventListener('pointermove', onPointerMove);
    container.removeEventListener('pointerup', onPointerEnd);
    container.removeEventListener('pointercancel', onPointerEnd);
    container.removeEventListener('gesturestart', blockGesture);
  };
}
