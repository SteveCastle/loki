import { render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Regression test for the startup handler-registration race.
//
// Bug: useWarmTagSearch fired its `load-all-tags` IPC fetch with `enabled: true`
// the moment it mounted. The hook mounts during app boot, before the main
// process finishes `load-db` and registers the taxonomy IPC handlers, so the
// fetch failed with "No handler registered for 'load-all-tags'" on essentially
// every launch (visible as renderer:invoke:load-all-tags failures in
// app-log.jsonl). React Query retried after the DB became ready, so it was
// benign — but it logged an error every time.
//
// The fix gates the fetch on `initSessionId`, which the state machine only
// assigns once it reaches its post-DB `init` state. Empty id (pre-DB) => no
// fetch; real id (post-DB, handlers registered) => fetch.

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
});

describe('useWarmTagSearch startup gating', () => {
  it('does not fetch all-tags before the DB is ready (empty initSessionId)', async () => {
    mockInitSessionId = '';
    renderHarness();

    // Give React Query a tick to (not) schedule the disabled query.
    await new Promise((r) => setTimeout(r, 0));
    expect(mockInvoke).not.toHaveBeenCalled();
  });

  it('fetches all-tags once the DB is ready (initSessionId set)', async () => {
    mockInitSessionId = 'session-123';
    renderHarness();

    await waitFor(() => expect(mockInvoke).toHaveBeenCalledTimes(1));
    expect(mockInvoke).toHaveBeenCalledWith('load-all-tags', []);
  });
});
