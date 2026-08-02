/**
 * Machine-level tests for the session filter-state history:
 *
 *  - every successfully-loaded DB query is recorded in context.queryHistory
 *    (newest first, deduped, capped);
 *  - the FS base snapshot is captured when the first query leaves the FS
 *    view, is restored from memory when the query empties out (CLEAR_QUERY /
 *    removing the last predicate), and is NOT consumed by the restore — the
 *    base stays available for the next round trip;
 *  - APPLY_QUERY_STATE re-runs a recorded query instead of restoring any
 *    cached library.
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
  hasPersistedTags: jest.fn(() => false),
  hasPersistedQuery: jest.fn(() => false),
}));

const mockInvoke = jest.fn();
const mockLoadMediaByQuery = jest.fn();

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
  loadMediaByQuery: mockLoadMediaByQuery,
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

const FS_LIBRARY = [
  { path: '/test/folder/a.jpg', mtimeMs: 0 },
  { path: '/test/folder/b.jpg', mtimeMs: 0 },
];

const stateName = (value: any): string | null => {
  const lib = value?.library;
  if (typeof lib === 'string') return lib;
  if (lib && typeof lib === 'object') return Object.keys(lib)[0] ?? null;
  return null;
};

function waitFor(
  service: any,
  predicate: (state: any) => boolean,
  label: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error(`timed out waiting for ${label}`)),
      4000
    );
    if (predicate(service.state)) {
      clearTimeout(timeout);
      resolve();
      return;
    }
    const sub = service.subscribe((state: any) => {
      if (predicate(state)) {
        clearTimeout(timeout);
        sub.unsubscribe();
        resolve();
      }
    });
  });
}

const inState = (name: string) => (state: any) =>
  stateName(state.value) === name;

const tagPredicate = (label: string) => ({
  type: 'tag',
  value: label,
  exclude: false,
  join: 'AND',
});

describe('session filter-state history', () => {
  beforeEach(() => {
    mockSetSessionValue.mockClear();
    mockSetSessionValues.mockClear();
    mockClearSessionKeys.mockClear();
    mockInvoke.mockReset();
    mockLoadMediaByQuery.mockReset();

    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        return { library: FS_LIBRARY, cursor: 0 };
      }
      return null;
    });
    // Each query "returns" one item per predicate so counts are observable.
    mockLoadMediaByQuery.mockImplementation(async (predicates: any[]) => ({
      library: predicates.map((p, i) => ({
        path: `/db/${p.value}-${i}.jpg`,
        mtimeMs: 0,
      })),
      cursor: 0,
    }));
  });

  async function startInFs() {
    const service = interpret(libraryMachine).start();
    await waitFor(service, inState('loadedFromFS'), 'loadedFromFS');
    // The store-mock default filteringMode is EXCLUSIVE (a second predicate
    // would REPLACE the query); these tests stack predicates, so use AND.
    service.send({ type: 'CHANGE_SETTING', data: { filteringMode: 'AND' } });
    return service;
  }

  it('records each loaded query, restores the FS base on clear, and keeps the base available', async () => {
    const service = await startInFs();

    // First query: cats. The FS view must be snapshotted as the base.
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('cats') },
    });
    await waitFor(
      service,
      (s) =>
        inState('loadedFromDB')(s) && s.context.queryHistory.length === 1,
      'first query recorded'
    );
    expect(service.state.context.previousLibrary).toEqual(FS_LIBRARY);
    expect(service.state.context.queryHistory[0].count).toBe(1);
    expect(
      service.state.context.queryHistory[0].query.predicates[0].value
    ).toBe('cats');

    // Second query: cats AND dogs — a NEW state on top; base untouched.
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('dogs') },
    });
    await waitFor(
      service,
      (s) =>
        inState('loadedFromDB')(s) && s.context.queryHistory.length === 2,
      'second query recorded'
    );
    expect(
      service.state.context.queryHistory[0].query.predicates
    ).toHaveLength(2);
    expect(service.state.context.previousLibrary).toEqual(FS_LIBRARY);

    // Clear: back to the FS base FROM MEMORY (no load-files re-run), and the
    // base snapshot survives for the next round trip.
    const loadFilesCallsBefore = mockInvoke.mock.calls.filter(
      ([c]: [string]) => c === 'load-files'
    ).length;
    service.send({ type: 'CLEAR_QUERY' });
    await waitFor(service, inState('loadedFromFS'), 'restored to FS');
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    expect(service.state.context.query.predicates).toEqual([]);
    expect(service.state.context.previousLibrary).toEqual(FS_LIBRARY);
    expect(
      mockInvoke.mock.calls.filter(([c]: [string]) => c === 'load-files').length
    ).toBe(loadFilesCallsBefore);
    // History survives the trip back to base.
    expect(service.state.context.queryHistory).toHaveLength(2);

    service.stop();
  });

  it('removing the last predicate falls back to the FS base, never the previous query state', async () => {
    const service = await startInFs();

    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('cats') },
    });
    await waitFor(service, inState('loadedFromDB'), 'cats loaded');
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('dogs') },
    });
    await waitFor(
      service,
      (s) => inState('loadedFromDB')(s) && s.context.queryHistory.length === 2,
      'cats+dogs loaded'
    );

    // Remove dogs → still a query (cats) → re-runs.
    const dogsKey = service.state.context.query.predicates
      .filter((p: any) => p.value === 'dogs')
      .map((p: any) => `${p.exclude ? '-' : ''}${p.type}:${p.value}`)[0];
    service.send({ type: 'REMOVE_PREDICATE', data: { key: dogsKey } });
    await waitFor(
      service,
      (s) =>
        inState('loadedFromDB')(s) && s.context.query.predicates.length === 1,
      'back to cats'
    );

    // Remove cats (the last predicate) → FS base, not the cats+dogs state.
    const catsKey = 'tag:cats';
    service.send({ type: 'REMOVE_PREDICATE', data: { key: catsKey } });
    await waitFor(service, inState('loadedFromFS'), 'FS base restored');
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    expect(service.state.context.query.predicates).toEqual([]);

    service.stop();
  });

  it('APPLY_QUERY_STATE re-runs the recorded query (fresh load, not a cached library)', async () => {
    const service = await startInFs();

    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('cats') },
    });
    await waitFor(service, inState('loadedFromDB'), 'cats loaded');
    service.send({ type: 'CLEAR_QUERY' });
    await waitFor(service, inState('loadedFromFS'), 'back to base');

    const entry = service.state.context.queryHistory[0];
    const queriesBefore = mockLoadMediaByQuery.mock.calls.length;
    service.send({
      type: 'APPLY_QUERY_STATE',
      data: { predicates: entry.query.predicates },
    });
    await waitFor(service, inState('loadedFromDB'), 're-applied');
    expect(mockLoadMediaByQuery.mock.calls.length).toBe(queriesBefore + 1);
    expect(service.state.context.query.predicates[0].value).toBe('cats');
    // Re-running the same state dedupes: still one entry, moved to top.
    expect(service.state.context.queryHistory).toHaveLength(1);

    // An EMPTY applied state routes to the base restore instead.
    service.send({ type: 'APPLY_QUERY_STATE', data: { predicates: [] } });
    await waitFor(service, inState('loadedFromFS'), 'empty state → base');
    expect(service.state.context.library).toEqual(FS_LIBRARY);

    service.stop();
  });
});
