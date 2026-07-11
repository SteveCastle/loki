import { render, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TextEncoder } from 'util';

// jsdom has no TextEncoder; the palette base64-encodes job queries with it.
(global as any).TextEncoder = (global as any).TextEncoder || TextEncoder;

// Regression test: while a directory load STREAMS items in, every
// LOAD_FILES_BATCH mints a new libraryLoadId for the same logical load (list
// views re-key on it). The context palette's close-on-library-change effect
// used to treat each of those as a library switch, so the right-click menu
// collapsed the instant the next batch arrived. It must stay open across
// streaming batches (and across stream completion), close on a genuine
// library switch, and close when a NEW load starts mid-stream (initialFile
// changes).

const mockSend = jest.fn();

const mockContext: any = {};

// resetContext restores the palette-open, mid-stream baseline each test.
function resetContext() {
  Object.assign(mockContext, {
    contextPalette: {
      display: true,
      position: { x: 10, y: 10 },
      target: { type: 'library' },
    },
    commandPalette: { display: false, position: {} },
    currentStateType: 'fs',
    dbQuery: { tags: [] },
    query: { predicates: [] },
    textFilter: '',
    initialFile: 'C:/media/a.jpg',
    settings: {
      filteringMode: 'EXCLUSIVE',
      recursive: false,
      filters: 'all',
      sortBy: 'name',
    },
    library: [],
    libraryLoadId: 'load-1',
    streaming: true,
    authToken: null,
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
}));

jest.mock('../renderer/stream-bus', () => ({
  __esModule: true,
  subscribeStream: () => () => undefined,
  streamConnected: () => true, // health check short-circuits, no fetch loop
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
  // and trip React's not-wrapped-in-act warning; the deps panel isn't under
  // test here.
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

const paletteTree = () => (
  <QueryClientProvider client={client}>
    <ContextPalette />
  </QueryClientProvider>
);

function renderPalette() {
  client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(paletteTree());
}

// Mutates the mocked machine context and re-renders so the palette's
// selectors and effects see the change, like a real service transition.
function transition(
  rerender: (ui: React.ReactElement) => void,
  changes: Record<string, unknown>
) {
  Object.assign(mockContext, changes);
  act(() => {
    rerender(paletteTree());
  });
}

const hideCalls = () =>
  mockSend.mock.calls.filter(([e]) => e === 'HIDE_CONTEXT_PALETTE').length;

describe('context palette during streaming directory loads', () => {
  beforeEach(() => {
    resetContext();
    mockSend.mockClear();
    // Every palette fetch path try/catches; a network error exercises the
    // "server unavailable" branches without needing response shapes.
    global.fetch = jest.fn(() => Promise.reject(new Error('offline'))) as any;
  });

  it('stays open across streaming batches and stream completion, then closes on a real switch', () => {
    const { container, rerender } = renderPalette();
    expect(container.querySelector('.ContextPalette')).not.toBeNull();

    // Two streamed-in batches, each minting a new libraryLoadId.
    transition(rerender, { libraryLoadId: 'load-2' });
    transition(rerender, { libraryLoadId: 'load-3' });
    expect(hideCalls()).toBe(0);

    // Stream completes: one more loadId change, streaming flips off.
    transition(rerender, { libraryLoadId: 'load-4', streaming: false });
    expect(hideCalls()).toBe(0);

    // A later loadId change NOT born from a stream (query re-run, reload,
    // shuffle) is a genuine library switch → the palette closes.
    transition(rerender, { libraryLoadId: 'load-5' });
    expect(hideCalls()).toBe(1);
  });

  it('closes when a NEW load starts mid-stream (initialFile changes)', () => {
    const { rerender } = renderPalette();

    transition(rerender, { libraryLoadId: 'load-2' }); // same-load batch
    expect(hideCalls()).toBe(0);

    // User opens a different directory while the old one is still streaming:
    // loadingFromFS re-enters with a new initialFile, streaming stays true.
    transition(rerender, {
      libraryLoadId: 'load-3',
      initialFile: 'C:/other/b.jpg',
    });
    expect(hideCalls()).toBe(1);
  });

  it('renders nothing for view-only public visitors (canWrite=false)', () => {
    resetContext();
    mockContext.canWrite = false;
    const { container } = renderPalette();
    expect(container.querySelector('.ContextPalette')).toBeNull();
  });
});
