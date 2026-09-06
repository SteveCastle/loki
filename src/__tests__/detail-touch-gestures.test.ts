import { attachTouchGestures } from '../renderer/components/detail/touch-gestures';

// Detail-view touch: one finger pans by writing scrollLeft/scrollTop (the
// same scroll model the mouse pan uses), two fingers pinch-zoom the media
// imperatively and commit the reached size as a scale mode once, on release.

class FakePointerEvent extends MouseEvent {
  pointerId: number;
  pointerType: string;
  isPrimary: boolean;
  constructor(type: string, init: any = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'touch';
    // Like a browser: the first finger down is primary, later ones aren't.
    this.isPrimary = init.isPrimary ?? (this.pointerId === 1);
  }
}

const CONTAINER = { w: 1000, h: 800 };
// A 2:1 landscape picture: fit height in this container is 500.
const NATURAL = { w: 2000, h: 1000 };

function px(v: string): number {
  return v.endsWith('px') ? parseFloat(v) : NaN;
}

// jsdom has no layout, so give the container and media the geometry the
// browser would compute: the media's box follows its inline size (or, in
// `fit`, fills the container), centred when smaller than the container.
function makeScene(initial: 'fit' | 'zoomed') {
  const container = document.createElement('div');
  Object.defineProperty(container, 'clientWidth', { get: () => CONTAINER.w });
  Object.defineProperty(container, 'clientHeight', { get: () => CONTAINER.h });
  container.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: CONTAINER.w, height: CONTAINER.h } as DOMRect);
  let scrollLeft = 0;
  let scrollTop = 0;
  Object.defineProperty(container, 'scrollLeft', {
    get: () => scrollLeft,
    set: (v: number) => {
      scrollLeft = Math.max(0, v);
    },
  });
  Object.defineProperty(container, 'scrollTop', {
    get: () => scrollTop,
    set: (v: number) => {
      scrollTop = Math.max(0, v);
    },
  });

  const media = document.createElement('img');
  Object.defineProperty(media, 'naturalWidth', { get: () => NATURAL.w });
  Object.defineProperty(media, 'naturalHeight', { get: () => NATURAL.h });
  if (initial === 'zoomed') {
    media.style.height = '200%';
  }
  const boxW = () => {
    const w = px(media.style.width);
    if (!Number.isNaN(w)) return w;
    const h = px(media.style.height);
    if (!Number.isNaN(h)) return (h * NATURAL.w) / NATURAL.h;
    if (media.style.height.endsWith('%')) {
      return (((parseFloat(media.style.height) / 100) * CONTAINER.h) * NATURAL.w) / NATURAL.h;
    }
    return CONTAINER.w;
  };
  const boxH = () => {
    const h = px(media.style.height);
    if (!Number.isNaN(h)) return h;
    if (media.style.height.endsWith('%')) {
      return (parseFloat(media.style.height) / 100) * CONTAINER.h;
    }
    return CONTAINER.h;
  };
  Object.defineProperty(media, 'offsetWidth', { get: boxW });
  Object.defineProperty(media, 'offsetHeight', { get: boxH });
  Object.defineProperty(media, 'offsetLeft', {
    get: () => Math.max(0, (CONTAINER.w - boxW()) / 2),
  });
  Object.defineProperty(media, 'offsetTop', {
    get: () => Math.max(0, (CONTAINER.h - boxH()) / 2),
  });
  container.appendChild(media);
  document.body.appendChild(container);

  const commit = jest.fn();
  const detach = attachTouchGestures(container, () => media, commit);
  return { container, media, commit, detach };
}

function pointer(
  target: Element,
  type: string,
  opts: { id?: number; x: number; y: number; pointerType?: string }
) {
  target.dispatchEvent(
    new FakePointerEvent(type, {
      bubbles: true,
      cancelable: true,
      clientX: opts.x,
      clientY: opts.y,
      pointerId: opts.id ?? 1,
      pointerType: opts.pointerType ?? 'touch',
    })
  );
}

describe('detail touch gestures', () => {
  beforeAll(() => {
    (window as any).PointerEvent = FakePointerEvent;
  });
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('one finger pans by writing the scroll position', () => {
    const { container, media, commit } = makeScene('zoomed');
    container.scrollLeft = 300;
    container.scrollTop = 200;
    pointer(container, 'pointerdown', { x: 500, y: 400 });
    pointer(container, 'pointermove', { x: 480, y: 410 });
    expect(container.scrollLeft).toBe(320);
    expect(container.scrollTop).toBe(190);
    pointer(container, 'pointerup', { x: 480, y: 410 });
    // Pan never touches the media size or the setting.
    expect(media.style.width).toBe('');
    expect(commit).not.toHaveBeenCalled();
  });

  it('ignores mouse pointers (mouse drag-pan is handled elsewhere)', () => {
    const { container } = makeScene('zoomed');
    container.scrollLeft = 300;
    pointer(container, 'pointerdown', { x: 500, y: 400, pointerType: 'mouse' });
    pointer(container, 'pointermove', { x: 400, y: 400, pointerType: 'mouse' });
    expect(container.scrollLeft).toBe(300);
  });

  it('pinching out from fit scales the picture and commits a percentage', () => {
    const { container, media, commit } = makeScene('fit');
    // Fingers 100px apart, centred in the container.
    pointer(container, 'pointerdown', { id: 1, x: 450, y: 400 });
    pointer(container, 'pointerdown', { id: 2, x: 550, y: 400 });
    // Spread to 200px apart: 2x. Fit height is 500 -> 1000px tall, 2000 wide.
    pointer(container, 'pointermove', { id: 1, x: 400, y: 400 });
    pointer(container, 'pointermove', { id: 2, x: 600, y: 400 });
    expect(px(media.style.height)).toBeCloseTo(1000);
    expect(px(media.style.width)).toBeCloseTo(2000);
    // The point under the fingers (the picture's centre) stays put: the
    // picture is now 2000x1000 at origin, so its centre sits at (1000, 500)
    // in scroll space, under the fingers at (500, 400).
    expect(container.scrollLeft).toBeCloseTo(500);
    expect(container.scrollTop).toBeCloseTo(100);
    expect(commit).not.toHaveBeenCalled();

    pointer(container, 'pointerup', { id: 2, x: 600, y: 400 });
    // 1000px of an 800px container = 125%. Inline width is released so the
    // committed percentage height alone sizes the picture.
    expect(commit).toHaveBeenCalledWith(125);
    expect(media.style.width).toBe('');
    expect(media.style.height).toBe('125%');
  });

  it('caps the zoom at twice the native height', () => {
    const { container, media } = makeScene('fit');
    pointer(container, 'pointerdown', { id: 1, x: 490, y: 400 });
    pointer(container, 'pointerdown', { id: 2, x: 510, y: 400 });
    // 20px apart -> 900px apart: 45x. Native height is 1000.
    pointer(container, 'pointermove', { id: 1, x: 50, y: 400 });
    pointer(container, 'pointermove', { id: 2, x: 950, y: 400 });
    expect(px(media.style.height)).toBeCloseTo(2000);
  });

  it('pinching down to fit (or below) commits `fit`', () => {
    const { container, media, commit } = makeScene('zoomed');
    pointer(container, 'pointerdown', { id: 1, x: 400, y: 400 });
    pointer(container, 'pointerdown', { id: 2, x: 600, y: 400 });
    // Squeeze to a quarter: 1600px tall -> 400, under the 500px fit height.
    pointer(container, 'pointermove', { id: 1, x: 475, y: 400 });
    pointer(container, 'pointermove', { id: 2, x: 525, y: 400 });
    expect(px(media.style.height)).toBeCloseTo(400);
    pointer(container, 'pointerup', { id: 1, x: 475, y: 400 });
    pointer(container, 'pointerup', { id: 2, x: 525, y: 400 });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledWith('fit');
    expect(media.style.width).toBe('');
    expect(media.style.height).toBe('');
  });

  it('a two-finger touch without movement commits nothing', () => {
    // That is the context-palette press; it must not rewrite the setting.
    const { container, media, commit } = makeScene('zoomed');
    pointer(container, 'pointerdown', { id: 1, x: 400, y: 400 });
    pointer(container, 'pointerdown', { id: 2, x: 600, y: 400 });
    pointer(container, 'pointerup', { id: 1, x: 400, y: 400 });
    pointer(container, 'pointerup', { id: 2, x: 600, y: 400 });
    expect(commit).not.toHaveBeenCalled();
    expect(media.style.height).toBe('200%');
  });

  it('lifting one pinch finger continues as a pan with the other', () => {
    const { container } = makeScene('zoomed');
    pointer(container, 'pointerdown', { id: 1, x: 400, y: 400 });
    pointer(container, 'pointerdown', { id: 2, x: 600, y: 400 });
    pointer(container, 'pointerup', { id: 2, x: 600, y: 400 });
    container.scrollLeft = 300;
    pointer(container, 'pointermove', { id: 1, x: 350, y: 400 });
    expect(container.scrollLeft).toBe(350);
  });

  it('detaches cleanly', () => {
    const { container, detach } = makeScene('zoomed');
    detach();
    container.scrollLeft = 300;
    pointer(container, 'pointerdown', { x: 500, y: 400 });
    pointer(container, 'pointermove', { x: 400, y: 400 });
    expect(container.scrollLeft).toBe(300);
  });
});
