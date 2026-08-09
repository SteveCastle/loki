import { act, render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Two things gate the startup all-tags fetch, and this covers both.
//
// 1. The handler-registration race. useWarmTagSearch used to fire its
//    `load-all-tags` IPC fetch with `enabled: true` the moment it mounted. The
//    hook mounts during app boot, before the main process finishes `load-db`
//    and registers the taxonomy IPC handlers, so the fetch failed with "No
//    handler registered for 'load-all-tags'" on essentially every launch. It is
//    now gated on `initSessionId`, which the state machine only assigns once it
//    reaches its post-DB `init` state.
//
// 2. The startup critical path. Even with handlers registered, this fetch is
//    the whole tag table (~190K rows: ~0.5s of SQLite plus a structured clone
//    across IPC and again into the search worker). Running it while the user is
//    waiting to see the file they opened blocked the main process — which also
//    serves gsm:// media bytes — and starved the shared SQLite connection. It
//    now additionally waits for the first media paint (see first-paint.ts).
//
// So: DB not ready => no fetch. DB ready but nothing painted yet => no fetch.
// Both => fetch.

let mockInitSessionId = '';
const mockInvoke = jest.fn(async (..._args: any[]) => [] as unknown[]);
const mockIndexTags = jest.fn();

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({ context: { initSessionId: mockInitSessionId } }),
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
}));

jest.mock('../renderer/search/tag-search-service', () => ({
  __esModule: true,
  indexTags: (...args: any[]) => mockIndexTags(...args),
}));

import {
  notifyFirstMediaPainted,
  resetFirstPaintForTests,
} from '../renderer/first-paint';
import { useWarmTagSearch } from '../renderer/hooks/useWarmTagSearch';

function Harness() {
  useWarmTagSearch();
  return null;
}

function renderHarness() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  mockInvoke.mockClear();
  mockIndexTags.mockClear();
  resetFirstPaintForTests();
});

describe('useWarmTagSearch startup gating', () => {
  it('does not fetch all-tags before the DB is ready (empty initSessionId)', async () => {
    mockInitSessionId = '';
    renderHarness();

    // Paint, then give React Query a tick to (not) schedule the disabled query.
    await act(async () => {
      notifyFirstMediaPainted();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(mockInvoke).not.toHaveBeenCalled();
  });

  it('does not fetch all-tags while the opened media is still loading', async () => {
    mockInitSessionId = 'session-123';
    renderHarness();

    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(mockInvoke).not.toHaveBeenCalled();
  });

  it('fetches tags once the DB is ready and the media has painted', async () => {
    mockInitSessionId = 'session-123';
    renderHarness();
    // The paint releases the gate via an idle callback, so let that timer land
    // inside act() rather than leaving React to warn about the late update.
    await act(async () => {
      notifyFirstMediaPainted();
      await new Promise((r) => setTimeout(r, 0));
    });

    // The CURATED scope, not the whole table. What gets warmed at startup is
    // what the command palette searches, and the palette skips the autotagger's
    // "Suggested" bucket — ~183K of ~189K tags. Warming the full set here would
    // put all of that back on the startup path for a surface that never shows
    // it; the taxonomy sidebar fetches the complete set lazily on focus instead.
    await waitFor(() =>
      expect(mockInvoke).toHaveBeenCalledWith('load-all-tags', [['Suggested']])
    );
    // Categories ride the same idle window: it is the last thing the palette
    // waits on to be fully ready, and asking at open time queues it behind the
    // still-busy main process (~194ms measured on a first open).
    await waitFor(() =>
      expect(mockInvoke).toHaveBeenCalledWith('load-categories', [])
    );
  });
});
