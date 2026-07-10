import { render, act, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Regression test: the × (clear) button on a job toast must JUST hide the
// notification — never cancel or remove the underlying job — and the toast must
// stay hidden even when more SSE updates for that job arrive.
//
// Bug: handleClearJob POSTed /job/{id}/cancel (active jobs) or /job/{id}/remove
// (finished), so clearing killed the job; and for active jobs the resulting
// `cancelled` state wasn't auto-removed, so the toast lingered too.

let streamHandler: ((type: string, event: { data: string }) => void) | null =
  null;

const mockContext: any = {
  library: [],
  authToken: 'tok',
  libraryLoadId: 'load-1',
  toasts: [],
};

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
        send: jest.fn(),
        getSnapshot: () => ({
          context: { libraryLoadId: 'load-1' },
          matches: () => false,
        }),
      },
    }),
  };
});

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  send: jest.fn(),
  mediaServerBase: 'http://server',
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

const runningJob = {
  id: 'job-1',
  command: 'embed',
  arguments: [],
  input: '',
  state: 'in_progress',
  created_at: '2026-07-06T00:00:00Z',
};

function emit(type: string, job: any) {
  act(() => {
    streamHandler?.(type, { data: JSON.stringify({ job }) });
  });
}

function renderToasts() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ToastSystem />
    </QueryClientProvider>
  );
}

describe('job toast clear button', () => {
  beforeEach(() => {
    streamHandler = null;
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true } as Response)
    ) as any;
  });

  it('hides the toast without touching the job, and keeps it hidden on later updates', () => {
    const { container } = renderToasts();

    // A running job arrives over the stream → its toast renders.
    emit('create', runningJob);
    expect(container.querySelector('.job-toast')).not.toBeNull();

    // Click the × close button.
    const close = container.querySelector('.toast-close') as HTMLElement;
    expect(close).not.toBeNull();
    fireEvent.click(close);

    // The job must NOT be cancelled/removed on the server.
    expect(global.fetch).not.toHaveBeenCalled();
    // The toast is gone.
    expect(container.querySelector('.job-toast')).toBeNull();

    // A subsequent progress/state update for the same job must NOT resurrect it.
    emit('update', { ...runningJob, progress_done: 3, progress_total: 10 });
    expect(container.querySelector('.job-toast')).toBeNull();
  });
});
