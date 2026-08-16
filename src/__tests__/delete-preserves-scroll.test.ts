/**
 * Regression test: deleting an item must not send the list back to the top.
 *
 * The list resets scroll whenever `libraryLoadId` changes, UNLESS the machine
 * hands it `preserveScrollFromLoadId` equal to the OUTGOING loadId (see the
 * effect in components/list/list.tsx). Removing a single item is an in-place
 * edit of the view the user is already looking at, so DELETE_FILE — and its
 * siblings FORGET_FILE / REMOVE_MERGED_FILES — must mint that token.
 *
 * The "outgoing" part is load-bearing: within one xstate `assign` object every
 * property assigner is called with the pre-update context, so
 * `preserveScrollFromLoadId` captures the id the list is still holding, not
 * the freshly minted one it is about to see.
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

const mockInvoke = jest.fn();

jest.mock('../renderer/platform', () => ({
  invoke: mockInvoke,
  send: jest.fn(),
  on: jest.fn(() => () => {}),
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
// eslint-disable-next-line import/first
import { libraryMachine } from '../renderer/state';

const PATHS = ['/test/folder/a.jpg', '/test/folder/b.jpg', '/test/folder/c.jpg'];

const inLoadedFromFS = (value: any): boolean => {
  const lib = value?.library;
  if (lib === 'loadedFromFS') return true;
  if (lib && typeof lib === 'object' && 'loadedFromFS' in lib) return true;
  return false;
};

async function startInLoadedFromFS() {
  const seeded = libraryMachine.withContext({
    ...libraryMachine.context,
    canWrite: true,
  });
  const service = interpret(seeded).start();

  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('timed out waiting for loadedFromFS')),
      4000
    );
    const done = () => {
      clearTimeout(timeout);
      resolve();
    };
    if (inLoadedFromFS(service.state.value)) {
      done();
      return;
    }
    const sub = service.subscribe((state) => {
      if (inLoadedFromFS(state.value)) {
        sub.unsubscribe();
        done();
      }
    });
  });

  return service;
}

describe('in-place removals preserve list scroll', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        return {
          library: PATHS.map((path) => ({ path, mtimeMs: 0 })),
          cursor: 1,
        };
      }
      return null;
    });
  });

  it.each([
    ['DELETE_FILE', { path: PATHS[1] }],
    ['FORGET_FILE', { path: PATHS[1] }],
    ['REMOVE_MERGED_FILES', { paths: [PATHS[1]] }],
  ])('%s hands the list the outgoing loadId', async (type, data) => {
    const service = await startInLoadedFromFS();
    const before = service.state.context;
    expect(before.library.map((i: any) => i.path)).toEqual(PATHS);

    service.send({ type, data } as any);

    const after = service.state.context;
    // The item really left the library, and the list is told to re-render.
    expect(after.library.map((i: any) => i.path)).toEqual([
      PATHS[0],
      PATHS[2],
    ]);
    expect(after.libraryLoadId).not.toBe(before.libraryLoadId);

    // …but the change is an in-place edit: the token must name the loadId the
    // list is still holding, so its "did the library change under me?" check
    // matches and it keeps the user's scroll position.
    expect(after.preserveScrollFromLoadId).toBe(before.libraryLoadId);
    expect(after.preserveScrollFromLoadId).not.toBe(after.libraryLoadId);

    service.stop();
  });
});
