/**
 * Navigating to a specific item must land on it. With the library filtered to
 * one media type (e.g. Images), jumping to a video used to silently land on
 * some other item. Explicit navigation now clears a type filter that would
 * hide the target; the settings menu's own cursor reset (sent right after the
 * user picks a filter) must not.
 */

const mockGetSessionValue = jest.fn(() => null);

jest.mock('../renderer/hooks/useSessionStore', () => ({
  initSessionStore: jest.fn().mockResolvedValue({}),
  getSessionValue: mockGetSessionValue,
  setSessionValue: jest.fn(),
  setSessionValues: jest.fn(),
  clearSessionKeys: jest.fn(),
  flushSession: jest.fn(),
  hasPersistedLibrary: jest.fn(() => false),
  hasPersistedTextFilter: jest.fn(() => false),
  hasPersistedTags: jest.fn(() => false),
}));

const mockInvoke = jest.fn();
const mockSend = jest.fn();

jest.mock('../renderer/platform', () => ({
  invoke: mockInvoke,
  send: mockSend,
  logEvent: jest.fn(),
  on: jest.fn(() => () => undefined),
  isElectron: true,
  mediaServerBase: 'http://localhost:10111',
  capabilities: {
    fileSystemAccess: true,
    clipboard: true,
    windowControls: true,
    autoUpdate: true,
    shutdown: true,
  },
  appArgs: {
    filePath: '/test/folder',
    // 'web' skips the post-load-db fetch to the media server's /config
    dbPath: 'web',
    appUserData: '/test/userData',
  },
  store: {
    get: (_k: string, d: any) => d,
    set: jest.fn(),
    getMany: (pairs: [string, any][]) =>
      Object.fromEntries(pairs.map(([k, def]) => [k, def])),
  },
  sessionStore: {
    get: jest.fn(),
    set: jest.fn(),
    getAll: jest.fn(),
    setMany: jest.fn(),
    clear: jest.fn(),
    clearKeys: jest.fn(),
    flush: jest.fn(),
  },
  loadMediaFromDB: jest.fn(),
  loadMediaByDescriptionSearch: jest.fn(),
  loadMediaByQuery: jest.fn(),
  fetchMediaPreview: jest.fn(),
  fetchTagPreview: jest.fn(),
  fetchTagCount: jest.fn(),
  listThumbnails: jest.fn(),
  regenerateThumbnail: jest.fn(),
  loadDuplicatesByPath: jest.fn(),
  mergeDuplicatesByPath: jest.fn(),
  getGifMetadata: jest.fn(),
  mediaUrl: (p: string) => p,
  hlsUrl: null,
  transcript: { loadTranscript: jest.fn(), modifyTranscript: jest.fn() },
}));

import { interpret } from 'xstate';
import { libraryMachine } from '../renderer/state';

// The scan never settles while true, so the machine parks in `loadingFromFS`
// and the streaming-phase context stays observable.
let holdLoad = false;

const inState = (value: any, name: string): boolean => {
  const lib = value?.library;
  if (lib === name) return true;
  if (lib && typeof lib === 'object' && name in lib) return true;
  return false;
};

const waitForState = (service: any, name: string) =>
  new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error(`timed out waiting for ${name}`)),
      4000
    );
    if (inState(service.state.value, name)) {
      clearTimeout(timeout);
      resolve();
      return;
    }
    const sub = service.subscribe((state: any) => {
      if (inState(state.value, name)) {
        clearTimeout(timeout);
        sub.unsubscribe();
        resolve();
      }
    });
  });

const filters = (service: any) => service.state.context.settings.filters;

describe('navigating past a media-type filter', () => {
  let service: any;

  beforeEach(() => {
    holdLoad = false;
    mockInvoke.mockReset();
    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        if (holdLoad) return new Promise(() => undefined);
        return {
          library: [
            { path: '/test/folder/a.jpg', mtimeMs: 0 },
            { path: '/test/folder/b.mp4', mtimeMs: 0 },
          ],
          cursor: 0,
        };
      }
      return null;
    });
  });

  afterEach(() => {
    service?.stop();
    service = undefined;
  });

  const boot = async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadedFromFS');
    // The user filters the library to images only.
    service.send({ type: 'CHANGE_SETTING', data: { filters: 'static' } });
    expect(filters(service)).toBe('static');
  };

  it('clears the filter when opening a file it hides', async () => {
    await boot();
    const toastsBefore = service.state.context.toasts.length;

    holdLoad = true;
    service.send({ type: 'SET_FILE', path: '/test/other/clip.mp4' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(filters(service)).toBe('all');
    expect(service.state.context.toasts).toHaveLength(toastsBefore + 1);
  });

  it('keeps the filter when the opened file passes it', async () => {
    await boot();
    holdLoad = true;
    service.send({ type: 'SET_FILE', path: '/test/other/photo.jpg' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(filters(service)).toBe('static');
  });

  it('keeps the filter when opening a folder', async () => {
    await boot();
    holdLoad = true;
    service.send({ type: 'SET_FILE', path: '/test/other' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(filters(service)).toBe('static');
  });

  it('reveals a hidden item on a jump-to-media cursor reset', async () => {
    await boot();
    service.send({
      type: 'RESET_CURSOR',
      currentItem: { path: '/test/folder/b.mp4' },
      reveal: true,
    } as any);

    expect(filters(service)).toBe('all');
    const { cursor, library } = service.state.context;
    expect(cursor).toBe(1);
    expect(library[cursor].path).toBe('/test/folder/b.mp4');
  });

  it("leaves the settings menu's cursor reset alone", async () => {
    await boot();
    service.send({
      type: 'RESET_CURSOR',
      currentItem: { path: '/test/folder/b.mp4' },
    } as any);

    expect(filters(service)).toBe('static');
  });
});
