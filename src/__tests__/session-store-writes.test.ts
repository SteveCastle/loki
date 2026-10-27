// The session store hands item arrays (the library, the FS base snapshot) to
// the main process by structured clone — seconds of renderer time at a million
// items. These pin the two guards that keep that from happening needlessly:
// an unchanged array is never re-sent, and an unload flush sends only keys the
// main process doesn't already have.
const mockSet = jest.fn(() => Promise.resolve());
const mockSetMany = jest.fn((_updates: Record<string, unknown>) => Promise.resolve());
const mockFlush = jest.fn(() => Promise.resolve());

jest.mock('../renderer/platform', () => ({
  sessionStore: {
    set: (...args: unknown[]) => (mockSet as any)(...args),
    setMany: (u: Record<string, unknown>) => mockSetMany(u),
    flush: () => mockFlush(),
    clear: () => Promise.resolve(),
    clearKeys: () => Promise.resolve(),
    getAll: () => Promise.resolve(null),
  },
}));

// eslint-disable-next-line import/first
import {
  flushSession,
  setSessionValue,
  setSessionValues,
} from '../renderer/hooks/useSessionStore';

describe('session store writes', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockSet.mockClear();
    mockSetMany.mockClear();
    mockFlush.mockClear();
  });
  afterEach(() => jest.useRealTimers());

  it('does not re-send a library array the main process already has', () => {
    const library = [{ path: '/a.jpg', mtimeMs: 1 }];
    setSessionValues({ library: { library, initialFile: '/a.jpg' } });
    jest.runAllTimers();
    expect(mockSetMany).toHaveBeenCalledTimes(1);

    setSessionValues({
      library: { library, initialFile: '/a.jpg' },
      cursor: { cursor: 3 },
    });
    jest.runAllTimers();
    expect(mockSetMany).toHaveBeenCalledTimes(2);
    expect(Object.keys(mockSetMany.mock.calls[1][0])).toEqual(['cursor']);

    setSessionValue('library', { library, initialFile: '/a.jpg' });
    jest.runAllTimers();
    expect(mockSet).not.toHaveBeenCalled();
  });

  it('cancels a queued older value when the sent one comes back', () => {
    const a = [{ path: '/a.jpg', mtimeMs: 1 }];
    const b = [{ path: '/b.jpg', mtimeMs: 1 }];
    setSessionValue('previous', { previousLibrary: a, previousCursor: 0 });
    jest.runAllTimers();
    setSessionValue('previous', { previousLibrary: b, previousCursor: 0 });
    setSessionValue('previous', { previousLibrary: a, previousCursor: 0 });
    jest.runAllTimers();
    expect(mockSet).toHaveBeenCalledTimes(1);
  });

  it('flushes only unsent keys', async () => {
    const library = [{ path: '/c.jpg', mtimeMs: 1 }];
    setSessionValues({ library: { library, initialFile: '/c.jpg' } });
    jest.runAllTimers();
    mockSetMany.mockClear();

    setSessionValue('cursor', { cursor: 7 });
    await flushSession();
    expect(mockSetMany).toHaveBeenCalledTimes(1);
    expect(Object.keys(mockSetMany.mock.calls[0][0])).toEqual(['cursor']);
    expect(mockFlush).toHaveBeenCalled();

    mockSetMany.mockClear();
    await flushSession();
    expect(mockSetMany).not.toHaveBeenCalled();
  });
});
