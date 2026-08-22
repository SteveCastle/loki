import { render, act, fireEvent, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TextEncoder } from 'util';

// jsdom has no TextEncoder; the palette base64-encodes job queries with it.
(global as any).TextEncoder = (global as any).TextEncoder || TextEncoder;
// ...nor AbortSignal.timeout, which every palette fetch passes as its signal.
// Without this the job POST throws before reaching fetch and is swallowed by
// its own try/catch, so the test would see "no request" and blame the feature.
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

// The context palette's scope toggle.
//
// Right-clicking a list item targets THAT file (and whatever was shift/ctrl-
// clicked into the selection), which is submitted as an explicit path list.
// Widening the scope to the library must instead submit the base64 query for
// the current view — the only form that can address media the client hasn't
// loaded, and how a job runs over a whole tag/filter/directory.

const mockSend = jest.fn();

const mockContext: any = {};

const FILE = 'C:/media/a.jpg';

function resetContext() {
  Object.assign(mockContext, {
    contextPalette: {
      display: true,
      position: { x: 10, y: 10 },
      target: { type: 'file', path: FILE },
      selection: [FILE],
      anchorIdx: 0,
    },
    commandPalette: { display: false, position: {} },
    currentStateType: 'db',
    dbQuery: { tags: ['cats'] },
    query: {
      predicates: [{ type: 'tag', value: 'cats', exclude: false, join: 'AND' }],
    },
    textFilter: '',
    initialFile: FILE,
    settings: {
      filteringMode: 'EXCLUSIVE',
      recursive: false,
      filters: 'all',
      sortBy: 'name',
    },
    library: [{ path: FILE }, { path: 'C:/media/b.jpg' }, { path: 'C:/media/c.jpg' }],
    // Distinct per test run so filter()'s single-entry memo never serves a
    // stale list from a previous test.
    libraryLoadId: `load-${Math.random()}`,
    streaming: false,
    authToken: 'token',
    canWrite: true,
  });
}

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({ context: mockContext }),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({
      libraryService: {
        send: mockSend,
        getSnapshot: () => ({ context: mockContext, matches: () => false }),
      },
    }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  capabilities: {},
  mediaServerBase: 'http://server',
  isElectron: false,
  invoke: jest.fn(),
  send: jest.fn(),
}));

jest.mock('../renderer/stream-bus', () => ({
  __esModule: true,
  subscribeStream: () => () => undefined,
  streamConnected: () => true, // server reads as available without a probe
}));

jest.mock('../renderer/first-paint', () => ({
  __esModule: true,
  // In prod the chip-selection hydration waits for first paint; tests hydrate
  // synchronously so the localStorage seeding in beforeEach is visible at mount.
  onIdleAfterFirstPaint: (cb: () => void) => {
    cb();
    return () => undefined;
  },
}));

jest.mock('@rehooks/component-size', () => ({
  __esModule: true,
  default: () => ({ width: 300, height: 200 }),
}));

jest.mock('../renderer/components/controls/login-widget', () => ({
  __esModule: true,
  default: () => null,
}));

jest.mock('../renderer/onboarding/api', () => ({
  __esModule: true,
  // Never settles: a resolution would setState after the test's last render
  // and trip React's not-wrapped-in-act warning; deps aren't under test here.
  fetchStatus: jest.fn(() => new Promise(() => undefined)),
  startModelDownload: jest.fn(),
  isDownloadableState: () => false,
  isDownloadingState: () => false,
}));

jest.mock('../renderer/onboarding/requirements', () => ({
  __esModule: true,
  TASK_REQUIREMENTS: {},
  depsApiBase: 'http://server/api/deps',
  fmtSize: (n: number) => String(n),
}));

import ContextPalette from '../renderer/components/controls/context-palette';

let client: QueryClient;

function renderPalette() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ContextPalette />
    </QueryClientProvider>
  );
}

// The input of the last POST /create the palette made.
function lastJobInput(): string {
  const calls = (global.fetch as jest.Mock).mock.calls.filter(([url]) =>
    String(url).endsWith('/create')
  );
  expect(calls.length).toBeGreaterThan(0);
  return JSON.parse(calls[calls.length - 1][1].body).input;
}

function decodeQuery64(input: string): string {
  const m = /--query64=(\S+)/.exec(input);
  expect(m).not.toBeNull();
  return atob(m![1]);
}

describe('context palette scope toggle', () => {
  beforeEach(() => {
    resetContext();
    mockSend.mockClear();
    // One chip pre-selected so Run is enabled without extra clicks.
    localStorage.setItem('loki.contextPalette.selectedTypes', '["Tags"]');
    // Only the job POST is under test; the palette's background polls (jobs
    // list, workflows) fail silently so they can't setState mid-assertion.
    global.fetch = jest.fn((url: string) =>
      String(url).endsWith('/create')
        ? Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
        : Promise.reject(new Error('offline'))
    ) as any;
  });

  it('runs on the explicit path list by default, and on the library query once widened', async () => {
    renderPalette();

    // Default scope: the right-clicked file only.
    expect(screen.getByText('1 file')).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByText('Run'));
    });
    const fileInput = lastJobInput();
    expect(fileInput).toBe(`autotag "${FILE}"`);
    expect(fileInput).not.toContain('--query64');

    // Widen to the whole current view.
    await act(async () => {
      fireEvent.click(screen.getByText('Entire library'));
    });

    // The header now describes the library context, not the file.
    expect(screen.getByText('3 items')).toBeTruthy();
    expect(screen.getByText('1 filter selected')).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByText('Run'));
    });
    const libraryInput = lastJobInput();
    expect(decodeQuery64(libraryInput)).toBe('tag:"cats"');
    // --query64 must stay the LAST token — the server takes the final token
    // as the job input.
    expect(libraryInput.trim().endsWith(libraryInput.split(' ').pop()!)).toBe(
      true
    );
    expect(libraryInput.split(' ').pop()).toMatch(/^--query64=/);
  });

  it('widens past a multi-item selection rather than combining with it', async () => {
    mockContext.contextPalette.selection = [FILE, 'C:/media/b.jpg'];
    renderPalette();

    expect(screen.getByText('2 files')).toBeTruthy();
    expect(screen.getByText('2 selected')).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByText('Entire library'));
    });

    await act(async () => {
      fireEvent.click(screen.getByText('Run'));
    });
    const input = lastJobInput();
    expect(decodeQuery64(input)).toBe('tag:"cats"');
    expect(input).not.toContain('C:/media/b.jpg');

    // The multi-item actions widen with the scope: Merge now targets the
    // whole filtered view (3 items), not the 2-file selection it set aside.
    expect(screen.getByText('Merge 3 items into first')).toBeTruthy();
  });

  it('falls back to a path query for a filesystem directory view', async () => {
    mockContext.currentStateType = 'fs';
    mockContext.dbQuery = { tags: [] };
    mockContext.query = { predicates: [] };
    renderPalette();

    await act(async () => {
      fireEvent.click(screen.getByText('Entire library'));
    });
    await act(async () => {
      fireEvent.click(screen.getByText('Run'));
    });

    // No representable predicates: the query matches the directory being
    // browsed, and Tags-alone in `missing` mode skips already-tagged media.
    expect(decodeQuery64(lastJobInput())).toBe(
      'pathdir:"C:/media" tagcount:<3'
    );
  });

  it('is not offered for tag targets, which are already library-wide queries', () => {
    mockContext.contextPalette.target = { type: 'tag', tag: 'cats' };
    mockContext.contextPalette.selection = [];
    renderPalette();

    expect(screen.queryByText('Entire library')).toBeNull();
  });

  it('is not offered when no job can be launched (signed out)', () => {
    mockContext.authToken = null;
    renderPalette();

    expect(screen.queryByText('Entire library')).toBeNull();
    // ...and the file context still renders normally.
    expect(screen.getByText('1 file')).toBeTruthy();
  });
});
