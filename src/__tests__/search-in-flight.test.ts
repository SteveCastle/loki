/**
 * Machine-level tests for a library search in flight (runningQuery /
 * loadingFromDB): the contract the progress UI and Cancel rely on.
 *
 *  - settledQuery tracks the query whose results are on screen;
 *  - CANCEL_QUERY stops the search and reverts the query to the settled one,
 *    keeping the library on screen (folder view or previous DB results);
 *  - a FAILED refinement keeps the previous DB results AND their chips (it
 *    used to drop to loadedFromFS, wiping the chips over stale DB results);
 *  - a cancelled/superseded rejection never toasts;
 *  - tag clicks during a search are applied latest-wins, not dropped;
 *  - toggling off the last TAG keeps other predicates (path etc.).
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
const mockCancelMediaQuery = jest.fn();

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
  cancelMediaQuery: mockCancelMediaQuery,
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

const pathPredicate = (value: string) => ({
  type: 'path',
  value,
  exclude: false,
  join: 'AND',
});

interface Deferred {
  resolve: () => void;
  reject: (e: any) => void;
}

const resultFor = (predicates: any[]) => ({
  library: predicates.map((p: any, i: number) => ({
    path: `/db/${p.value}-${i}.jpg`,
    mtimeMs: 0,
  })),
  cursor: 0,
});

describe('search in flight', () => {
  // Queries resolve only when the test says so (or immediately when `auto`).
  let pending: Deferred[];
  let auto: boolean;

  beforeEach(() => {
    pending = [];
    auto = true;
    mockSetSessionValue.mockClear();
    mockSetSessionValues.mockClear();
    mockClearSessionKeys.mockClear();
    mockCancelMediaQuery.mockReset();
    mockInvoke.mockReset();
    mockLoadMediaByQuery.mockReset();
    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') return { library: FS_LIBRARY, cursor: 0 };
      return null;
    });
    mockLoadMediaByQuery.mockImplementation((predicates: any[]) => {
      if (auto) return Promise.resolve(resultFor(predicates));
      return new Promise((resolve, reject) => {
        pending.push({ resolve: () => resolve(resultFor(predicates)), reject });
      });
    });
  });

  async function startInFs() {
    const service = interpret(libraryMachine).start();
    await waitFor(service, inState('loadedFromFS'), 'loadedFromFS');
    service.send({ type: 'CHANGE_SETTING', data: { filteringMode: 'AND' } });
    return service;
  }

  async function startWithCats() {
    const service = await startInFs();
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: tagPredicate('cats') },
    });
    await waitFor(service, inState('loadedFromDB'), 'cats loaded');
    return service;
  }

  const values = (q: any) => q.predicates.map((p: any) => p.value);

  it('tracks the settled query and stamps when a search starts', async () => {
    const service = await startInFs();
    expect(service.state.context.settledQuery.predicates).toEqual([]);

    auto = false;
    const before = Date.now();
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    expect(inState('runningQuery')(service.state)).toBe(true);
    expect(service.state.context.queryStartedAt).toBeGreaterThanOrEqual(before);
    // Asked for, but not on screen yet.
    expect(values(service.state.context.query)).toEqual(['zaush']);
    expect(service.state.context.settledQuery.predicates).toEqual([]);

    pending[0].resolve();
    await waitFor(service, inState('loadedFromDB'), 'path loaded');
    expect(values(service.state.context.settledQuery)).toEqual(['zaush']);
    service.stop();
  });

  it('CANCEL_QUERY from a folder view returns to it, untouched', async () => {
    const service = await startInFs();
    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    service.send('CANCEL_QUERY');
    expect(inState('loadedFromFS')(service.state)).toBe(true);
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    expect(service.state.context.query.predicates).toEqual([]);
    expect(mockCancelMediaQuery).toHaveBeenCalledTimes(1);
    // The abandoned search landing later changes nothing.
    pending[0].resolve();
    await new Promise((r) => setTimeout(r, 20));
    expect(inState('loadedFromFS')(service.state)).toBe(true);
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    service.stop();
  });

  it('CANCEL_QUERY over DB results keeps them and restores their chips', async () => {
    const service = await startWithCats();
    const catsLibrary = service.state.context.library;
    const loadId = service.state.context.libraryLoadId;

    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    expect(values(service.state.context.query)).toEqual(['cats', 'zaush']);
    service.send('CANCEL_QUERY');

    expect(inState('loadedFromDB')(service.state)).toBe(true);
    expect(values(service.state.context.query)).toEqual(['cats']);
    expect(service.state.context.dbQuery.tags).toEqual(['cats']);
    expect(service.state.context.library).toBe(catsLibrary);
    // Same library, same load: the list keeps its scroll.
    expect(service.state.context.libraryLoadId).toBe(loadId);
    expect(mockCancelMediaQuery).toHaveBeenCalled();
    service.stop();
  });

  it('a failed refinement keeps the previous results AND their chips', async () => {
    const service = await startWithCats();
    const catsLibrary = service.state.context.library;
    const toastsBefore = service.state.context.toasts.length;

    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    pending[0].reject(new Error('boom'));
    await waitFor(service, (s) => !inState('runningQuery')(s), 'settled');

    expect(inState('loadedFromDB')(service.state)).toBe(true);
    expect(service.state.context.currentStateType).toBe('db');
    expect(values(service.state.context.query)).toEqual(['cats']);
    expect(service.state.context.library).toBe(catsLibrary);
    expect(service.state.context.toasts.length).toBe(toastsBefore + 1);
    // Completion (incl. failure) is not an abandonment: nothing to cancel.
    expect(mockCancelMediaQuery).not.toHaveBeenCalled();
    service.stop();
  });

  it('a failed first search from a folder view stays in the folder view', async () => {
    const service = await startInFs();
    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    pending[0].reject(new Error('boom'));
    await waitFor(service, inState('loadedFromFS'), 'back to folder');
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    expect(service.state.context.query.predicates).toEqual([]);
    service.stop();
  });

  it('a cancelled/superseded rejection never toasts', async () => {
    const service = await startWithCats();
    const toastsBefore = service.state.context.toasts.length;
    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    const cancelled = new Error('Search cancelled');
    cancelled.name = 'QueryCancelledError';
    pending[0].reject(cancelled);
    await waitFor(service, inState('loadedFromDB'), 'reverted');
    expect(service.state.context.toasts.length).toBe(toastsBefore);
    service.stop();
  });

  it('a tag click during a search is applied latest-wins, not dropped', async () => {
    const service = await startInFs();
    auto = false;
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    service.send({ type: 'SET_QUERY_TAG', data: { tag: 'cats' } });
    expect(inState('runningQuery')(service.state)).toBe(true);
    expect(values(service.state.context.query)).toEqual(['zaush', 'cats']);
    expect(service.state.context.dbQuery.tags).toEqual(['cats']);
    // The superseded search was told to stop.
    expect(mockCancelMediaQuery).toHaveBeenCalledTimes(1);
    // The FS base is still the folder view.
    expect(service.state.context.previousLibrary).toEqual(FS_LIBRARY);

    // Only the latest request's result is applied.
    pending[0].resolve();
    pending[1].resolve();
    await waitFor(service, inState('loadedFromDB'), 'latest loaded');
    expect(service.state.context.library).toHaveLength(2);
    expect(values(service.state.context.settledQuery)).toEqual([
      'zaush',
      'cats',
    ]);
    service.stop();
  });

  it('toggling off the last tag keeps the other predicates', async () => {
    const service = await startInFs();
    service.send({
      type: 'ADD_PREDICATE',
      data: { predicate: pathPredicate('zaush') },
    });
    await waitFor(service, inState('loadedFromDB'), 'path loaded');
    service.send({ type: 'SET_QUERY_TAG', data: { tag: 'cats' } });
    await waitFor(
      service,
      (s) =>
        inState('loadedFromDB')(s) && s.context.query.predicates.length === 2,
      'path + cats loaded'
    );

    service.send({ type: 'SET_QUERY_TAG', data: { tag: 'cats' } });
    await waitFor(
      service,
      (s) =>
        inState('loadedFromDB')(s) && s.context.query.predicates.length === 1,
      'back to path only'
    );
    expect(values(service.state.context.query)).toEqual(['zaush']);
    expect(service.state.context.dbQuery.tags).toEqual([]);
    service.stop();
  });

  it('toggling off the only predicate returns to the folder base', async () => {
    const service = await startInFs();
    service.send({ type: 'SET_QUERY_TAG', data: { tag: 'cats' } });
    await waitFor(service, inState('loadedFromDB'), 'cats loaded');
    service.send({ type: 'SET_QUERY_TAG', data: { tag: 'cats' } });
    await waitFor(service, inState('loadedFromFS'), 'base restored');
    expect(service.state.context.library).toEqual(FS_LIBRARY);
    service.stop();
  });
});
