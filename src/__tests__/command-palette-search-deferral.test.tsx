import { act, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// The command palette is the most frequently opened surface in the app, so the
// commit that opens it must stay cheap. Everything data-bound behind the search
// input (the categories fetch, the shared tag index, the suggestion sections
// and their per-category count IPCs) lives in CommandPaletteResults, which is
// mounted only AFTER the palette has painted — or immediately on the first
// keystroke, so a fast typist never loses the results surface.
//
// These tests lock that contract in: the search surface's first commit must not
// fire a single IPC.

const mockInvoke = jest.fn(async (..._args: any[]) => [] as unknown[]);

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({
      context: {
        query: { predicates: [], mode: 'AND' },
        settings: { filteringMode: 'AND', applyTagPreview: false },
        initSessionId: 'session-123',
        // Filter-state history dropdown inputs (useFilterHistory).
        queryHistory: [],
        currentStateType: 'fs',
        library: [],
        previousLibrary: [],
        initialFile: '/test/folder',
        previousInitialFile: '',
      },
    }),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({ libraryService: {} }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  invoke: (...args: any[]) => mockInvoke(...args),
  mediaUrl: (p: string) => p,
  sessionStore: {
    get: jest.fn(async () => []),
    set: jest.fn(async () => undefined),
  },
}));

// The real service builds a Worker (or a Fuse index over the whole library);
// neither belongs in this test.
jest.mock('../renderer/search/tag-search-service', () => ({
  __esModule: true,
  indexTags: jest.fn(),
  searchTags: (_q: string, _limit: number, cb: (items: unknown[]) => void) =>
    cb([]),
  // The results subtree reports whether the scope's index exists so a cold
  // index can be told apart from a warm one in the open trace.
  isWarm: () => true,
}));

import CommandPaletteSearch from '../renderer/components/controls/command-palette-search';

const libraryService = { send: jest.fn() } as any;

function channels() {
  return mockInvoke.mock.calls.map((call) => call[0]);
}

function renderSearch() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <CommandPaletteSearch libraryService={libraryService} />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  mockInvoke.mockClear();
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe('command palette search deferral', () => {
  it('renders the input immediately without firing any IPC', async () => {
    renderSearch();

    // Flush microtasks only — no timers, so the post-paint deferral has not
    // fired yet. This is the frame the palette becomes visible in.
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByPlaceholderText('Search & filter')).toBeTruthy();
    expect(channels()).toHaveLength(0);
  });

  it('mounts the search engine after paint', async () => {
    renderSearch();

    await act(async () => {
      jest.advanceTimersByTime(100);
      await Promise.resolve();
    });

    expect(channels()).toContain('load-categories');
  });

  it('mounts the search engine on the first keystroke, without waiting', async () => {
    renderSearch();

    await act(async () => {
      await Promise.resolve();
    });
    expect(channels()).toHaveLength(0);

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText('Search & filter'), {
        target: { value: 'cat' },
      });
      await Promise.resolve();
    });

    expect(channels()).toContain('load-categories');
  });
});
