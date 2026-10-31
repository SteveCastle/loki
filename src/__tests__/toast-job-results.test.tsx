import { render, act, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Job output never moves the user's view: new files are offered on the job's
// toast (opening in a new window), and only an overwrite of the file on
// screen reloads it (media-changes.ts has the contract).

let streamHandler: ((type: string, event: { data: string }) => void) | null =
  null;

const mockContext: any = {
  library: [{ path: 'C:\\media\\current.png' }, { path: 'C:\\media\\other.png' }],
  libraryLoadId: 'load-1',
  textFilter: '',
  cursor: 0,
  settings: { filters: 'all', sortBy: 'name' },
  authToken: 'tok',
  toasts: [],
};

const mockSend = jest.fn();
const mockOpenInNewWindow = jest.fn();

jest.mock('@xstate/react', () => ({
  useSelector: (_service: any, selector: any) =>
    selector({ context: mockContext }),
}));

jest.mock('../renderer/filter', () => ({
  __esModule: true,
  default: (_id: string, _text: string, library: any[]) => library,
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({
      libraryService: {
        send: (...args: any[]) => mockSend(...args),
        getSnapshot: () => ({ context: mockContext, matches: () => false }),
      },
    }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  send: jest.fn(),
  mediaServerBase: 'http://server',
  openInNewWindow: (...args: any[]) => mockOpenInNewWindow(...args),
}));

jest.mock('../renderer/stream-bus', () => ({
  __esModule: true,
  subscribeStream: (cb: any) => {
    streamHandler = cb;
    return () => {
      streamHandler = null;
    };
  },
}));

import { ToastSystem } from '../renderer/components/controls/toast-system';

const job = {
  id: 'job-1',
  command: 'retouch',
  arguments: [],
  input: 'C:\\media\\current.png',
  state: 'in_progress',
  created_at: '2026-10-31T00:00:00Z',
};

function emitJob(type: string, j: any) {
  act(() => {
    streamHandler?.(type, { data: JSON.stringify({ job: j }) });
  });
}

function emitMedia(type: string, payload: any) {
  act(() => {
    streamHandler?.(type, { data: JSON.stringify({ msg: JSON.stringify(payload) }) });
  });
}

function renderToasts() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ToastSystem />
    </QueryClientProvider>
  );
}

describe('job results', () => {
  beforeEach(() => {
    streamHandler = null;
    mockSend.mockClear();
    mockOpenInNewWindow.mockClear();
  });

  it('lists new files on the job toast and opens them in a new window, never moving the cursor', () => {
    const { container } = renderToasts();
    emitJob('create', job);
    emitMedia('media-created', { jobId: 'job-1', paths: ['C:\\media\\current_retouch.png'] });
    emitJob('update', { ...job, state: 'completed' });

    const link = container.querySelector('.job-toast .toast-result') as HTMLElement;
    expect(link).not.toBeNull();
    expect(link.textContent).toContain('current_retouch.png');
    fireEvent.click(link);
    expect(mockOpenInNewWindow).toHaveBeenCalledWith('C:\\media\\current_retouch.png');

    // No navigation, no library edit, no separate toast.
    const sent = mockSend.mock.calls.map((c) => c[0]?.type ?? c[0]);
    expect(sent).not.toContain('RESET_CURSOR');
    expect(sent).not.toContain('SET_FILE');
    expect(sent).not.toContain('ADD_TOAST');
  });

  it('falls back to a toast of its own when the job toast is not showing', () => {
    renderToasts();
    emitMedia('media-created', { jobId: 'unknown-job', paths: ['C:\\media\\a.png', 'C:\\media\\b.png'] });
    expect(mockSend).toHaveBeenCalledWith({
      type: 'ADD_TOAST',
      data: expect.objectContaining({
        type: 'success',
        title: '2 new files ready',
        files: ['C:\\media\\a.png', 'C:\\media\\b.png'],
      }),
    });
  });

  it('reloads the file on screen when a job overwrites it, without announcing it', () => {
    const reloads: string[][] = [];
    const onUpdated = (e: Event) => reloads.push((e as CustomEvent).detail.paths);
    window.addEventListener('loki-media-updated', onUpdated);
    try {
      renderToasts();
      emitMedia('media-updated', { jobId: 'gone', paths: ['c:/media/current.png'] });
      expect(reloads).toEqual([['c:/media/current.png']]);
      expect(mockSend).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('loki-media-updated', onUpdated);
    }
  });
});
