/**
 * Regression test: opening a FOLDER must not put the folder itself in the
 * library.
 *
 * Bug: `loadingFromFS` unconditionally seeded the library with
 * `[{ path: initialFile }]` and pinned that path. That seed is right when the
 * user opens a single media file (the viewer can paint it before the scan
 * finishes), but when `initialFile` is a directory (or an archive) it put a
 * blank, unrenderable item at cursor 0 — and because the pinned directory path
 * never appears in the scan results, the cursor stayed parked on it for the
 * whole stream.
 *
 * Expected: a container load starts EMPTY and the cursor lands on the first
 * file streamed in; a media-file load still seeds and pins that file.
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

jest.mock('../renderer/platform', () => ({
  invoke: mockInvoke,
  send: jest.fn(),
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

describe('directories are never library items', () => {
  let service: any;

  beforeEach(() => {
    holdLoad = false;
    mockInvoke.mockReset();
    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        if (holdLoad) return new Promise(() => undefined);
        return {
          library: [{ path: '/test/folder/a.jpg', mtimeMs: 0 }],
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

  it('starts a folder load with an empty library and no pin', async () => {
    holdLoad = true;
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadingFromFS');

    // The folder itself is not an item.
    expect(service.state.context.library).toEqual([]);
    expect(service.state.context.pinnedPath).toBeNull();
    expect(service.state.context.initialFile).toBe('/test/folder');
  });

  it('lands the cursor on the first file streamed in from the folder', async () => {
    holdLoad = true;
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadingFromFS');

    service.send({
      type: 'LOAD_FILES_BATCH',
      data: {
        files: [
          { path: '/test/folder/a.jpg', mtimeMs: 0 },
          { path: '/test/folder/b.jpg', mtimeMs: 0 },
        ],
      },
    } as any);

    const { library, cursor } = service.state.context;
    expect(library.map((i: any) => i.path)).toEqual([
      '/test/folder/a.jpg',
      '/test/folder/b.jpg',
    ]);
    expect(cursor).toBe(0);
    expect(library[cursor].path).toBe('/test/folder/a.jpg');
  });

  it('still seeds and pins the file when a single media file is opened', async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadedFromFS');

    holdLoad = true;
    service.send({ type: 'SET_FILE', path: '/test/other/photo.jpg' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(
      service.state.context.library.map((i: any) => i.path)
    ).toEqual(['/test/other/photo.jpg']);
    expect(service.state.context.pinnedPath).toBe('/test/other/photo.jpg');
  });

  it('treats an archive like a container, not an item', async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadedFromFS');

    holdLoad = true;
    service.send({ type: 'SET_FILE', path: '/test/books/vol1.cbz' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(service.state.context.library).toEqual([]);
    expect(service.state.context.pinnedPath).toBeNull();
  });
});
