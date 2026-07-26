import { render, fireEvent } from '@testing-library/react';

// Regression test for the detail-view scroll-wheel cursor control.
//
// Bug: the wheel-listener effect depended on `containerRef.current`. On the
// initial mount the library hasn't loaded yet, so `item` is null and Detail
// early-returns `null` — the ref'd container isn't in the DOM. Ref assignment
// doesn't trigger a re-render, and React reads `containerRef.current` as null at
// render time, so when the container later mounted the effect never re-ran and
// the wheel listener was never attached. Toggling control mode forced an
// unrelated re-render that happened to re-run the effect (the user's workaround).
//
// The fix keys the effect on `item?.path` (which flips null -> value when the
// container mounts), so the listener attaches as soon as the media is available.

// Mutable holders read lazily inside the mock factories (assigned before render).
let mockState: any;
let mockLibrary: any[] = [];
const mockSend = jest.fn();

jest.mock('@xstate/react', () => ({
  // Ignore the equality fn; just run the selector against our fake state.
  useSelector: (_service: any, selector: any) => selector(mockState),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({
      libraryService: { send: (...args: any[]) => mockSend(...args) },
    }),
  };
});

// `filter` produces the working list the item selector cycles through.
jest.mock('renderer/filter', () => ({
  __esModule: true,
  default: () => mockLibrary,
}));

jest.mock('react-scroll-ondrag', () => ({
  __esModule: true,
  default: () => ({ events: {} }),
}));

jest.mock('renderer/hooks/useMediaDimensions', () => ({
  __esModule: true,
  default: () => ({ orientation: 'landscape' }),
}));

jest.mock('renderer/hooks/useTagDrop', () => ({
  __esModule: true,
  default: () => ({ drop: () => undefined }),
}));

jest.mock('../file-types', () => ({
  __esModule: true,
  FileTypes: {
    Video: 'video',
    Image: 'image',
    Audio: 'audio',
    Document: 'document',
    Archive: 'archive',
    Other: 'other',
  },
  getFileType: () => 'image',
}));

// Stub the media viewers / child UI so the test doesn't pull their heavy deps.
jest.mock('../renderer/components/media-viewers/image', () => ({
  __esModule: true,
  Image: () => null,
}));
jest.mock('../renderer/components/media-viewers/animated-gif', () => ({
  __esModule: true,
  AnimatedGif: () => null,
}));
jest.mock('../renderer/components/media-viewers/video', () => ({
  __esModule: true,
  Video: () => null,
}));
jest.mock('../renderer/components/media-viewers/audio', () => ({
  __esModule: true,
  Audio: () => null,
}));
jest.mock('../renderer/components/controls/video-controls', () => ({
  __esModule: true,
  default: () => null,
}));
jest.mock('../renderer/components/metadata/tags', () => ({
  __esModule: true,
  default: () => null,
}));
jest.mock('../renderer/components/elo/BattleMode', () => ({
  __esModule: true,
  default: () => null,
}));

import { Detail } from '../renderer/components/detail/detail';

const baseSettings = {
  controlMode: 'mouse',
  comicMode: false,
  battleMode: false,
  showControls: true,
  showTags: 'none',
  showFileInfo: 'none',
  filters: {},
  sortBy: 'name',
  scaleMode: 'cover',
  detailImageCache: false,
};

function makeState(controlMode: string) {
  return {
    context: {
      libraryLoadId: 'id',
      textFilter: '',
      library: [],
      cursor: 0,
      settings: { ...baseSettings, controlMode },
    },
  };
}

beforeAll(() => {
  // jsdom has no ResizeObserver; Detail instantiates one once the container mounts.
  (global as any).ResizeObserver = class {
    observe() {
      /* no-op */
    }
    unobserve() {
      /* no-op */
    }
    disconnect() {
      /* no-op */
    }
  };
});

beforeEach(() => {
  mockSend.mockClear();
  mockLibrary = [];
});

describe('Detail scroll-wheel cursor control', () => {
  it('attaches the wheel->cursor listener once the container mounts after the library loads', () => {
    mockState = makeState('mouse');

    // Initial mount: library not loaded -> item null -> container not rendered.
    mockLibrary = [];
    const { rerender, container } = render(<Detail />);
    expect(container.querySelector('.Detail')).toBeNull();

    // Library loads -> item present -> the ref'd container mounts.
    mockLibrary = [{ path: '/a.jpg', timeStamp: 0 }];
    rerender(<Detail />);
    const detail = container.querySelector('.Detail') as HTMLElement;
    expect(detail).not.toBeNull();

    // Scrolling down must move the cursor forward — proves the native wheel
    // listener attached even though the container mounted after the first render.
    fireEvent.wheel(detail, { deltaY: 10 });
    expect(mockSend).toHaveBeenCalledWith('INCREMENT_CURSOR');
  });

  it('does not move the cursor on wheel in touchpad mode', () => {
    mockState = makeState('touchpad');
    mockLibrary = [{ path: '/a.jpg', timeStamp: 0 }];

    const { container } = render(<Detail />);
    const detail = container.querySelector('.Detail') as HTMLElement;
    expect(detail).not.toBeNull();

    fireEvent.wheel(detail, { deltaY: 10 });
    expect(mockSend).not.toHaveBeenCalledWith('INCREMENT_CURSOR');
  });
});

describe('Detail touchpad click cursor control', () => {
  // Touchpad clicks advance the cursor only after a short window so that a
  // very fast double-click can be told apart and toggle the list view
  // instead of stepping twice (see handleClick in detail.tsx).
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  function renderTouchpadDetail() {
    mockState = makeState('touchpad');
    mockLibrary = [{ path: '/a.jpg', timeStamp: 0 }];
    const { container } = render(<Detail />);
    return container.querySelector('.Detail') as HTMLElement;
  }

  it('advances the cursor after the double-click window on a single click', () => {
    const detail = renderTouchpadDetail();

    // jsdom rects are all zeros, so any clientX lands on the right half.
    fireEvent.click(detail, { clientX: 100 });
    expect(mockSend).not.toHaveBeenCalledWith('INCREMENT_CURSOR');

    jest.advanceTimersByTime(250);
    expect(mockSend).toHaveBeenCalledWith('INCREMENT_CURSOR');
  });

  it('toggles the list view instead of stepping twice on a fast double-click', () => {
    const detail = renderTouchpadDetail();
    const toggled = jest.fn();
    window.addEventListener('loki-toggle-list-detail', toggled);

    fireEvent.click(detail, { clientX: 100 });
    jest.advanceTimersByTime(50);
    fireEvent.click(detail, { clientX: 100 });
    jest.advanceTimersByTime(500);

    expect(toggled).toHaveBeenCalledTimes(1);
    expect(mockSend).not.toHaveBeenCalledWith('INCREMENT_CURSOR');
    expect(mockSend).not.toHaveBeenCalledWith('DECREMENT_CURSOR');
    window.removeEventListener('loki-toggle-list-detail', toggled);
  });

  it('treats two clicks slower than the window as two cursor steps', () => {
    const detail = renderTouchpadDetail();
    const toggled = jest.fn();
    window.addEventListener('loki-toggle-list-detail', toggled);

    fireEvent.click(detail, { clientX: 100 });
    jest.advanceTimersByTime(300);
    fireEvent.click(detail, { clientX: 100 });
    jest.advanceTimersByTime(300);

    expect(toggled).not.toHaveBeenCalled();
    expect(
      mockSend.mock.calls.filter(([type]) => type === 'INCREMENT_CURSOR')
    ).toHaveLength(2);
    window.removeEventListener('loki-toggle-list-detail', toggled);
  });
});
