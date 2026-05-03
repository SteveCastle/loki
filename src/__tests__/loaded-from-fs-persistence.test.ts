/**
 * Regression test for the FS-mode persistence asymmetry.
 *
 * Bug: when the state machine enters `loadedFromFS`, the in-memory invariant
 * forces `dbQuery.tags=[]` and `textFilter=''` — but unlike the analogous
 * `loadedFromDB` and `loadedFromSearch` entries, it never mirrors that
 * cleared state to the session store. As a result, on-disk session.query
 * could retain stale tags while session.library held an FS-loaded folder.
 * On next boot, `loadingFromPersisted` would route the FS library into
 * `loadedFromDB` because it saw non-empty persisted tags.
 *
 * This test enters `loadedFromFS` from a context with stale tags and
 * verifies that session.query is written with the cleared values.
 */

const mockSetSessionValue = jest.fn();
const mockSetSessionValues = jest.fn();
const mockClearSessionKeys = jest.fn();
const mockGetSessionValue = jest.fn(() => null);

jest.mock('../renderer/hooks/useSessionStore', () => ({
  initSessionStore: jest.fn().mockResolvedValue({}),
  getSessionValue: mockGetSessionValue,
  setSessionValue: mockSetSessionValue,
  setSessionValues: mockSetSessionValues,
  clearSessionKeys: mockClearSessionKeys,
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
  capabilities: {
    fileSystemAccess: true,
    clipboard: true,
    windowControls: true,
    autoUpdate: true,
    shutdown: true,
  },
  appArgs: {
    filePath: '/test/folder',
    // 'web' skips the post-load-db fetch to localhost:8090/config
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

describe('loadedFromFS persistence invariant', () => {
  beforeEach(() => {
    mockSetSessionValue.mockClear();
    mockSetSessionValues.mockClear();
    mockClearSessionKeys.mockClear();
    mockInvoke.mockReset();

    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        return {
          library: [{ path: '/test/folder/a.jpg', mtimeMs: 0 }],
          cursor: 0,
        };
      }
      return null;
    });
  });

  it('writes session.query with cleared dbQuery and textFilter when entering loadedFromFS', async () => {
    // Seed the machine context with stale tags/filter that the FS-mode
    // invariant must clear and persist on entry.
    const seeded = libraryMachine.withContext({
      ...libraryMachine.context,
      dbQuery: { tags: ['stale-tag'] },
      textFilter: 'stale-filter',
    });

    const inLoadedFromFS = (value: any): boolean => {
      const lib = value?.library;
      if (lib === 'loadedFromFS') return true;
      if (lib && typeof lib === 'object' && 'loadedFromFS' in lib) return true;
      return false;
    };

    const service = interpret(seeded).start();

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error('timed out waiting for loadedFromFS')),
        4000
      );
      if (inLoadedFromFS(service.state.value)) {
        clearTimeout(timeout);
        resolve();
        return;
      }
      const sub = service.subscribe((state) => {
        if (inLoadedFromFS(state.value)) {
          clearTimeout(timeout);
          sub.unsubscribe();
          resolve();
        }
      });
    });

    service.stop();

    const queryWrites = mockSetSessionValue.mock.calls.filter(
      ([key]) => key === 'query'
    );

    // The FS-mode invariant entry must persist the cleared query state.
    // Before the fix this is 0; after the fix it is >= 1.
    expect(queryWrites.length).toBeGreaterThan(0);

    const lastQueryWrite = queryWrites[queryWrites.length - 1][1];
    expect(lastQueryWrite.dbQuery.tags).toEqual([]);
    expect(lastQueryWrite.textFilter).toBe('');
  });
});
