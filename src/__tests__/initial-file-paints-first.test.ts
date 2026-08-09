/**
 * Regression test: the file the user opened must be renderable on the FIRST
 * render, before any I/O.
 *
 * Opening media from the filesystem is the app's critical path. The library
 * used to start empty and only gain the opened file when the machine entered
 * `loadingFromFS` — which happens after the `load-db` IPC round trip (a
 * dynamic import of the sqlite driver, opening the DB, running migrations, and
 * registering ~50 IPC handlers). None of that is needed to point an <img> at a
 * path, so the viewer sat on a blank panel through work it didn't depend on.
 *
 * Expected: the machine's INITIAL context already holds the opened media file,
 * so the detail viewer mounts its media element on the very first render. A
 * directory or archive still starts empty — it is a container, not an item
 * (see directory-not-a-library-item.test.ts).
 */

jest.mock('../renderer/hooks/useSessionStore', () => ({
  initSessionStore: jest.fn().mockResolvedValue({}),
  getSessionValue: jest.fn(() => null),
  setSessionValue: jest.fn(),
  setSessionValues: jest.fn(),
  clearSessionKeys: jest.fn(),
  flushSession: jest.fn(),
  hasPersistedLibrary: jest.fn(() => false),
  hasPersistedTextFilter: jest.fn(() => false),
  hasPersistedTags: jest.fn(() => false),
}));

// Set per-test before the module under test is (re)imported: the initial
// context is computed at machine-definition time, so each case needs a fresh
// module registry.
let mockFilePath = '';

jest.mock('../renderer/platform', () => ({
  invoke: jest.fn(() => new Promise(() => undefined)),
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
  get appArgs() {
    return { filePath: mockFilePath, dbPath: 'web', appUserData: '/test' };
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
  logEvent: jest.fn(),
  transcript: { loadTranscript: jest.fn(), modifyTranscript: jest.fn() },
}));

function initialLibraryFor(filePath: string): { path: string }[] {
  mockFilePath = filePath;
  let machine: any;
  jest.isolateModules(() => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
    machine = require('../renderer/state').libraryMachine;
  });
  return machine.context.library;
}

describe('initial context seeds the opened media file', () => {
  it('holds the opened image before any state transition', () => {
    expect(initialLibraryFor('/media/photos/cat.jpg')).toEqual([
      { path: '/media/photos/cat.jpg', mtimeMs: 0 },
    ]);
  });

  it('holds the opened video too', () => {
    expect(initialLibraryFor('C:\\clips\\take3.mp4')).toEqual([
      { path: 'C:\\clips\\take3.mp4', mtimeMs: 0 },
    ]);
  });

  it('starts empty for a directory — a container is not a library item', () => {
    expect(initialLibraryFor('/media/photos')).toEqual([]);
  });

  it('starts empty for an archive', () => {
    expect(initialLibraryFor('/media/books/vol1.cbz')).toEqual([]);
  });

  it('starts empty when nothing was opened', () => {
    expect(initialLibraryFor('')).toEqual([]);
  });
});
