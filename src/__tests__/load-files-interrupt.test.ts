/**
 * A folder scan can take a long time, and loadingFromFS used to ignore every
 * navigation event while it ran. Opening another path, opening the file or
 * folder picker, or stopping outright must all interrupt the scan: main is
 * told to cancel it, the streamed-in partial library is kept in the user's
 * sort (not the temporary 'stream' sort), and late batches from the abandoned
 * scan are dropped.
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

const scanIdsSent = () =>
  mockInvoke.mock.calls
    .filter(([c]: [string]) => c === 'load-files')
    .map(([, args]: [string, any[]]) => args[3]?.scanId);

const cancelsSent = () =>
  mockSend.mock.calls
    .filter(([c]: [string]) => c === 'cancel-load-files')
    .map(([, args]: [string, any[]]) => args[0]);

const batch = (scanId: string | undefined, paths: string[]) =>
  ({
    type: 'LOAD_FILES_BATCH',
    data: { files: paths.map((p) => ({ path: p, mtimeMs: 0 })), scanId },
  } as any);

describe('interrupting a folder scan', () => {
  let service: any;

  beforeEach(() => {
    holdLoad = true;
    mockSend.mockReset();
    mockInvoke.mockReset();
    mockInvoke.mockImplementation(async (channel: string) => {
      if (channel === 'load-db') return { dbPath: '/test/db' };
      if (channel === 'load-files') {
        if (holdLoad) return new Promise(() => undefined);
        return { library: [], cursor: 0 };
      }
      if (channel === 'select-directory') return new Promise(() => undefined);
      return null;
    });
  });

  afterEach(() => {
    service?.stop();
    service = undefined;
  });

  it('opens another path mid-scan and cancels the old scan', async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadingFromFS');
    const [firstScan] = scanIdsSent();
    expect(firstScan).toBeTruthy();
    service.send(batch(firstScan, ['/test/folder/a.jpg']));

    service.send({ type: 'SET_FILE', path: '/test/other' } as any);
    await waitForState(service, 'loadingFromFS');

    expect(cancelsSent()).toEqual([firstScan]);
    const ids = scanIdsSent();
    expect(ids).toHaveLength(2);
    expect(ids[1]).not.toBe(firstScan);
    expect(service.state.context.initialFile).toBe('/test/other');
    // The user's sort survives the interruption (it would otherwise be
    // re-saved as 'stream' and the next load would end in stream order).
    expect(service.state.context.savedSortByDuringStreaming).not.toBe(
      'stream'
    );

    // A late batch from the abandoned scan is ignored; the new one lands.
    service.send(batch(firstScan, ['/test/folder/late.jpg']));
    expect(service.state.context.library).toEqual([]);
    service.send(batch(ids[1], ['/test/other/b.jpg']));
    expect(service.state.context.library.map((i: any) => i.path)).toEqual([
      '/test/other/b.jpg',
    ]);
  });

  it('opens the folder picker mid-scan', async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadingFromFS');
    const [scan] = scanIdsSent();

    service.send({ type: 'SELECT_DIRECTORY' } as any);
    await waitForState(service, 'selectingDirectory');

    expect(cancelsSent()).toEqual([scan]);
    expect(service.state.context.streaming).toBe(false);
  });

  it('stops a scan and keeps what streamed in', async () => {
    service = interpret(libraryMachine).start();
    await waitForState(service, 'loadingFromFS');
    const [scan] = scanIdsSent();
    const sortBefore = service.state.context.savedSortByDuringStreaming;
    service.send(
      batch(scan, ['/test/folder/b.jpg', '/test/folder/a.jpg'])
    );

    service.send({ type: 'CANCEL_LOAD' } as any);
    await waitForState(service, 'loadedFromFS');

    const ctx = service.state.context;
    expect(cancelsSent()).toEqual([scan]);
    expect(ctx.library.map((i: any) => i.path)).toEqual([
      '/test/folder/b.jpg',
      '/test/folder/a.jpg',
    ]);
    expect(ctx.streaming).toBe(false);
    expect(ctx.settings.sortBy).toBe(sortBefore);
    expect(ctx.savedSortByDuringStreaming).toBeNull();
  });
});
