import { render, act, fireEvent, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TextEncoder } from 'util';

// jsdom has no TextEncoder; the palette base64-encodes job queries with it.
(global as any).TextEncoder = (global as any).TextEncoder || TextEncoder;
// ...nor AbortSignal.timeout, which every palette fetch passes as its signal.
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

// The context palette's Dedupe action.
//
// Dedupe launches the server's `dedupe` task over the current context: a
// discrete multi-selection goes as an explicit path list, anything wider
// (library view, tag, category) as the base64 query — the same target
// contract as the generate jobs. Deleting files is not undoable, so the
// button is two-click armed like Merge, and a lone file offers no dedupe at
// all (nothing to deduplicate against).

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
    library: [{ path: FILE }, { path: 'C:/media/b.jpg' }],
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

function createCalls(): any[][] {
  return (global.fetch as jest.Mock).mock.calls.filter(([url]) =>
    String(url).endsWith('/create')
  );
}

function lastJobInput(): string {
  const calls = createCalls();
  expect(calls.length).toBeGreaterThan(0);
  return JSON.parse(calls[calls.length - 1][1].body).input;
}

describe('context palette dedupe action', () => {
  beforeEach(() => {
    resetContext();
    mockSend.mockClear();
    global.fetch = jest.fn((url: string) =>
      String(url).endsWith('/create')
        ? Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
        : Promise.reject(new Error('offline'))
    ) as any;
  });

  it('is hidden for a lone file — nothing to deduplicate against', () => {
    renderPalette();
    expect(screen.queryByText(/^Dedupe/)).toBeNull();
  });

  it('submits the multi-selection as a path list, after an arm-confirm', async () => {
    mockContext.contextPalette.selection = [FILE, 'C:/media/b.jpg'];
    renderPalette();

    // First click only arms — deleting files must never be one accidental
    // click away.
    await act(async () => {
      fireEvent.click(screen.getByText('Dedupe 2 selected files'));
    });
    expect(createCalls().length).toBe(0);

    await act(async () => {
      fireEvent.click(screen.getByText('Confirm — deletes duplicate files'));
    });
    expect(lastJobInput()).toBe(`dedupe "${FILE}\nC:/media/b.jpg"`);
  });

  it('submits the library scope as a query, with --query64 as the last token', async () => {
    renderPalette();

    await act(async () => {
      fireEvent.click(screen.getByText('Entire library'));
    });
    await act(async () => {
      fireEvent.click(screen.getByText('Dedupe duplicates'));
    });
    await act(async () => {
      fireEvent.click(screen.getByText('Confirm — deletes duplicate files'));
    });

    const input = lastJobInput();
    const m = /^dedupe --query64=(\S+)$/.exec(input);
    expect(m).not.toBeNull();
    expect(atob(m![1])).toBe('tag:"cats"');
  });

  it('offers query dedupe for a tag target', async () => {
    mockContext.contextPalette.target = { type: 'tag', tag: 'cats' };
    mockContext.contextPalette.selection = [];
    renderPalette();

    await act(async () => {
      fireEvent.click(screen.getByText('Dedupe duplicates'));
    });
    await act(async () => {
      fireEvent.click(screen.getByText('Confirm — deletes duplicate files'));
    });
    const input = lastJobInput();
    const m = /^dedupe --query64=(\S+)$/.exec(input);
    expect(m).not.toBeNull();
    expect(atob(m![1])).toBe('tag:"cats"');
  });
});
