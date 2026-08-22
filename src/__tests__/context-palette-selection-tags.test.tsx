import { render, act, fireEvent, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TextEncoder } from 'util';

// jsdom has no TextEncoder; the palette base64-encodes job queries with it.
(global as any).TextEncoder = (global as any).TextEncoder || TextEncoder;
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

// Bulk tag editing for a multi-item selection (the SelectionTags block under
// the Merge action):
//  - shows the tags shared by EVERY selected item,
//  - removes a shared tag from all items with a two-click confirm,
//  - adds a searched tag to all items with one create-assignment call.

const mockSend = jest.fn();

const mockContext: any = {};

const FILE_A = 'C:/media/a.jpg';
const FILE_B = 'C:/media/b.jpg';

function resetContext() {
  Object.assign(mockContext, {
    contextPalette: {
      display: true,
      position: { x: 10, y: 10 },
      target: { type: 'file', path: FILE_A },
      selection: [FILE_A, FILE_B],
      anchorIdx: 0,
    },
    commandPalette: { display: false, position: {} },
    currentStateType: 'db',
    dbQuery: { tags: [] },
    query: { predicates: [] },
    textFilter: '',
    initialFile: FILE_A,
    settings: {
      filteringMode: 'EXCLUSIVE',
      recursive: false,
      filters: 'all',
      sortBy: 'name',
      applyTagPreview: true,
    },
    library: [{ path: FILE_A }, { path: FILE_B }],
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

// Per-item tag fixtures: 'cats' is on both files, the others on only one —
// so the shared strip must show exactly 'cats'.
const TAGS_BY_PATH: Record<string, Array<{ tag_label: string; category_label?: string; time_stamp: number }>> = {
  [FILE_A]: [
    { tag_label: 'cats', category_label: 'Animals', time_stamp: 0 },
    { tag_label: 'dogs', category_label: 'Animals', time_stamp: 0 },
  ],
  [FILE_B]: [
    { tag_label: 'cats', category_label: 'Animals', time_stamp: 0 },
    { tag_label: 'birds', category_label: 'Animals', time_stamp: 0 },
  ],
};

const mockInvoke = jest.fn(async (channel: string, args: any[]) => {
  if (channel === 'load-tags-by-media-path') {
    const path = args[0]?.path ?? args[0];
    return { path, tags: TAGS_BY_PATH[path] ?? [] };
  }
  return undefined;
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  capabilities: {},
  mediaServerBase: 'http://server',
  mediaServerConfigured: false,
  isElectron: false,
  invoke: (channel: string, args: any[]) => mockInvoke(channel, args),
  send: jest.fn(),
}));

jest.mock('../renderer/stream-bus', () => ({
  __esModule: true,
  subscribeStream: () => () => undefined,
  streamConnected: () => true,
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

// Deterministic type-ahead: skip the debounced worker search. 'cats' is in the
// results on purpose — it is already shared, so the component must filter it
// out of the suggestions.
jest.mock('../renderer/hooks/useTagSearch', () => ({
  __esModule: true,
  useTagSearch: (text: string, enabled: boolean) => ({
    results:
      enabled && text
        ? [
            { label: 'space', category: 'Theme', weight: 1 },
            { label: 'cats', category: 'Animals', weight: 1 },
          ]
        : [],
  }),
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

describe('context palette selection tags', () => {
  beforeEach(() => {
    resetContext();
    mockSend.mockClear();
    mockInvoke.mockClear();
    localStorage.setItem('loki.contextPalette.selectedTypes', '[]');
    global.fetch = jest.fn(() => Promise.reject(new Error('offline'))) as any;
  });

  it('shows only the tags shared by every selected item', async () => {
    renderPalette();
    expect(await screen.findByText('cats')).toBeTruthy();
    expect(screen.queryByText('dogs')).toBeNull();
    expect(screen.queryByText('birds')).toBeNull();
  });

  it('removes a shared tag from all items with a two-click confirm', async () => {
    renderPalette();
    await screen.findByText('cats');

    const removeBtn = screen.getByLabelText('Remove cats from all items');
    // First click arms only — nothing deleted yet.
    fireEvent.click(removeBtn);
    expect(
      mockInvoke.mock.calls.filter(([ch]) => ch === 'delete-assignment')
    ).toHaveLength(0);

    // Second click deletes the label from every selected path, all
    // occurrences (time_stamp 0).
    await act(async () => {
      fireEvent.click(screen.getByLabelText('Confirm removing cats'));
    });
    const deletes = mockInvoke.mock.calls.filter(
      ([ch]) => ch === 'delete-assignment'
    );
    expect(deletes).toHaveLength(2);
    expect(deletes[0][1]).toEqual([
      FILE_A,
      { tag_label: 'cats', time_stamp: 0 },
    ]);
    expect(deletes[1][1]).toEqual([
      FILE_B,
      { tag_label: 'cats', time_stamp: 0 },
    ]);
    expect(mockSend).toHaveBeenCalledWith({ type: 'DELETED_ASSIGNMENT' });
  });

  it('adds a searched tag to all selected items in one call, hiding already-shared suggestions', async () => {
    renderPalette();
    await screen.findByText('cats');

    const input = screen.getByLabelText('Add tag to all selected items');
    fireEvent.change(input, { target: { value: 'sp' } });

    // 'space' is offered; 'cats' (already on every item) is filtered out of
    // the suggestion list — its only rendering is the shared chip.
    const row = await screen.findByText('space');
    expect(screen.getAllByText('cats')).toHaveLength(1);

    await act(async () => {
      fireEvent.click(row);
    });
    const creates = mockInvoke.mock.calls.filter(
      ([ch]) => ch === 'create-assignment'
    );
    expect(creates).toHaveLength(1);
    expect(creates[0][1]).toEqual([
      [FILE_A, FILE_B],
      'space',
      'Theme',
      null,
      true, // applyTagPreview from settings
    ]);
  });

  it('Enter commits the highlighted suggestion', async () => {
    renderPalette();
    await screen.findByText('cats');

    const input = screen.getByLabelText('Add tag to all selected items');
    fireEvent.change(input, { target: { value: 'sp' } });
    await screen.findByText('space');

    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' });
    });
    const creates = mockInvoke.mock.calls.filter(
      ([ch]) => ch === 'create-assignment'
    );
    expect(creates).toHaveLength(1);
    expect(creates[0][1][1]).toBe('space');
  });
});
