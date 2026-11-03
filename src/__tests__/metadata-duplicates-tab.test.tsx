import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// The metadata Duplicates tab answers "does this item have copies?" from the
// vector-similarity duplicate groups. An unrecorded preview (id 0) is
// recorded on the first action, and a merge only ever merges the members the
// card showed — recording can join a bigger group the user hasn't seen.

const mockContext: any = {
  authToken: null,
  canWrite: true,
  initSessionId: 's1',
  settings: { filteringMode: 'AND' },
  query: { predicates: [] },
};
const mockSend = jest.fn();
const mockApiGet = jest.fn();
const mockApiSend = jest.fn();
const mockReportMerge = jest.fn();

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({ context: mockContext }),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({
      libraryService: { send: (...a: any[]) => mockSend(...a) },
    }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  isElectron: false,
  mediaServerConfigured: true,
}));

jest.mock('../renderer/stream-bus', () => ({
  __esModule: true,
  subscribeStream: () => () => {},
}));

jest.mock('../renderer/components/taxonomy/duplicates-grid', () => ({
  __esModule: true,
  MERGE_TIMEOUT_MS: 1,
  apiGet: (...a: any[]) => mockApiGet(...a),
  apiSend: (...a: any[]) => mockApiSend(...a),
  reportMerge: (...a: any[]) => mockReportMerge(...a),
  GroupCard: ({ group, onMerge, onForget, currentPath }: any) => (
    <div>
      <span>card {group.id}</span>
      <span>current {currentPath}</span>
      <span>{onForget ? 'can forget' : 'no forget'}</span>
      <button onClick={() => onMerge(group, group.members[0].path)}>
        merge
      </button>
    </div>
  ),
}));
jest.mock('../renderer/components/taxonomy/people-grid.css', () => ({}));
jest.mock('../renderer/components/taxonomy/duplicates-grid.css', () => ({}));
jest.mock('../renderer/components/metadata/duplicates.css', () => ({}));

import Duplicates from '../renderer/components/metadata/duplicates';

const A = 'C:/media/a.jpg';
const B = 'C:/media/b.jpg';
const C = 'C:/media/c.jpg';

function renderTab() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <Duplicates basePath={A} />
    </QueryClientProvider>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
});

test('a preview is recorded on merge, and only the shown members merge', async () => {
  mockApiGet.mockResolvedValue({
    recorded: false,
    noEmbedding: false,
    group: {
      id: 0,
      anchorPath: A,
      status: 'pending',
      activeCount: 3,
      members: [
        { path: A, excluded: false },
        { path: B, excluded: false },
        { path: C, excluded: true },
      ],
    },
  });
  mockApiSend.mockImplementation(async (_t: any, _m: string, url: string) =>
    url === '/api/duplicates/check/record'
      ? { id: 7 }
      : { merge: { target: A, deleted: [B], failed: [] } }
  );

  renderTab();
  expect(await screen.findByText('card 0')).toBeTruthy();
  expect(screen.getByText('2 possible duplicates')).toBeTruthy();
  expect(screen.getByText(`current ${A}`)).toBeTruthy();
  expect(screen.getByText('no forget')).toBeTruthy();
  expect(mockApiGet.mock.calls[0][1]).toBe(
    `/api/duplicates/check?path=${encodeURIComponent(A)}&threshold=100`
  );

  fireEvent.click(screen.getByText('merge'));
  await waitFor(() => expect(mockReportMerge).toHaveBeenCalled());
  expect(mockApiSend.mock.calls[0].slice(1)).toEqual([
    'POST',
    '/api/duplicates/check/record',
    { path: A, threshold: 100 },
  ]);
  expect(mockApiSend.mock.calls[1].slice(1, 4)).toEqual([
    'POST',
    '/api/duplicates/7/merge',
    { keep: A, sources: [B] },
  ]);
});

test('a recorded group is acted on directly', async () => {
  mockApiGet.mockResolvedValue({
    recorded: true,
    noEmbedding: false,
    group: {
      id: 12,
      anchorPath: B,
      status: 'pending',
      activeCount: 2,
      members: [
        { path: A, excluded: false },
        { path: B, excluded: false },
      ],
    },
  });
  mockApiSend.mockResolvedValue({
    merge: { target: A, deleted: [B], failed: [] },
  });

  renderTab();
  expect(await screen.findByText('In duplicate group #12')).toBeTruthy();
  expect(screen.getByText('can forget')).toBeTruthy();
  fireEvent.click(screen.getByText('merge'));
  await waitFor(() => expect(mockReportMerge).toHaveBeenCalled());
  expect(mockApiSend).toHaveBeenCalledTimes(1);
  expect(mockApiSend.mock.calls[0][2]).toBe('/api/duplicates/12/merge');
});

test('no embedding offers to embed the item', async () => {
  mockApiGet.mockResolvedValue({
    recorded: false,
    noEmbedding: true,
    group: null,
  });
  mockApiSend.mockResolvedValue({});
  renderTab();
  fireEvent.click(await screen.findByText('Embed this item'));
  await waitFor(() =>
    expect(mockApiSend).toHaveBeenCalledWith(null, 'POST', '/create', {
      input: `embed "${A}"`,
    })
  );
});

test('no candidates says so', async () => {
  mockApiGet.mockResolvedValue({
    recorded: false,
    noEmbedding: false,
    group: null,
  });
  renderTab();
  expect(await screen.findByText('No duplicates found')).toBeTruthy();
});
