import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

import type { DepStatus } from '../renderer/onboarding/api';

jest.mock('../renderer/onboarding/api', () => {
  const actual = jest.requireActual('../renderer/onboarding/api');
  return {
    ...actual,
    fetchStatus: jest.fn(),
    startModelDownload: jest.fn(async () => undefined),
    cancelModelDownload: jest.fn(async () => undefined),
  };
});

import * as api from '../renderer/onboarding/api';
import {
  EngineSetupDeferred,
  engineOfJob,
  ensureEngines,
  getEngineSetup,
  planEngineSetup,
} from '../renderer/components/transform/engine-setup';
import EngineSetupDialog from '../renderer/components/transform/engine-setup-dialog';
import { submitJobs } from '../renderer/components/transform/store';

const GB = 1024 ** 3;
const dep = (id: string, state: DepStatus['state'], size: number, extra: Partial<DepStatus> = {}): DepStatus => ({
  id,
  name: id,
  category: id.startsWith('loki-') ? 'tool' : 'model',
  state,
  size_bytes: size,
  ...extra,
});
const missing = () => [
  dep('loki-retouch', 'missing', 45e6),
  dep('qwen-image-2.1', 'missing', 17 * GB),
  dep('loki-reshoot', 'missing', 55e6),
  dep('minimax-h3-ref2va', 'missing', 42 * GB),
];

const fetchStatus = api.fetchStatus as jest.Mock;
const startModelDownload = api.startModelDownload as jest.Mock;
const cancelModelDownload = api.cancelModelDownload as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
});

describe('planEngineSetup', () => {
  it('lists the executable and the model when neither is installed', () => {
    const need = planEngineSetup(missing(), 'retouch');
    expect(need?.items.map((d) => d.id)).toEqual(['loki-retouch', 'qwen-image-2.1']);
    expect(need?.bytes).toBe(45e6 + 17 * GB);
  });

  it('omits what is already installed and returns null when nothing is missing', () => {
    const s = missing().map((d) => (d.id === 'loki-retouch' ? { ...d, state: 'installed' as const } : d));
    expect(planEngineSetup(s, 'retouch')?.items.map((d) => d.id)).toEqual(['qwen-image-2.1']);
    const all = s.map((d) => ({ ...d, state: 'installed' as const }));
    expect(planEngineSetup(all, 'retouch')).toBeNull();
  });

  it('does not gate an executable that is on the server PATH, nor an unknown engine', () => {
    const s = missing().map((d) =>
      d.id === 'loki-reshoot' ? { ...d, state: 'installed' as const, detail: { source: 'path' } } : d
    );
    expect(planEngineSetup(s, 'reshoot')).toBeNull();
    expect(planEngineSetup([], 'reshoot')).toBeNull();
  });
});

describe('engineOfJob', () => {
  it('maps task ids, including the legacy 4kify alias', () => {
    expect(engineOfJob({ input: 'retouch "C:\\a.png"' })).toBe('retouch');
    expect(engineOfJob({ input: '4kify "C:\\a.png"' })).toBe('retouch');
    expect(engineOfJob({ input: 'reshoot "C:\\a.png"' })).toBe('reshoot');
    expect(engineOfJob({ input: 'ffmpeg "C:\\a.mp4"' })).toBeNull();
  });
});

describe('ensureEngines + dialog', () => {
  it('lets jobs for other tasks, and installed engines, straight through without asking', async () => {
    fetchStatus.mockResolvedValue(missing().map((d) => ({ ...d, state: 'installed' })));
    await expect(ensureEngines([{ input: 'retouch "x.png"' }], '')).resolves.toBe('ready');
    await expect(ensureEngines([{ input: 'ffmpeg "x.mp4"' }], '')).resolves.toBe('ready');
    expect(getEngineSetup()).toBeNull();
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });

  it('does not block when the deps API is unreachable', async () => {
    fetchStatus.mockRejectedValue(new Error('down'));
    await expect(ensureEngines([{ input: 'reshoot "x.png"' }], '')).resolves.toBe('ready');
  });

  it('asks first, downloads only after consent, and resolves ready when installed', async () => {
    render(<EngineSetupDialog />);
    fetchStatus.mockResolvedValueOnce(missing());
    let result: string | undefined;
    const p = ensureEngines([{ input: 'retouch "x.png"' }], 'http://srv').then((r) => {
      result = r;
    });

    await screen.findByText('One-time download');
    expect(screen.getByText(/runs entirely on this computer/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Download 17\.04 GB and continue/ })).toBeInTheDocument();
    expect(startModelDownload).not.toHaveBeenCalled(); // nothing downloads before consent
    expect(result).toBeUndefined();

    // After consent the server reports progress, then completion.
    fetchStatus
      .mockResolvedValueOnce([
        dep('loki-retouch', 'installed', 45e6),
        dep('qwen-image-2.1', 'downloading', 17 * GB, { detail: { bytes_done: 8.5 * GB } }),
      ])
      .mockResolvedValue([
        dep('loki-retouch', 'installed', 45e6),
        dep('qwen-image-2.1', 'installed', 17 * GB),
      ]);
    fireEvent.click(screen.getByRole('button', { name: /Download .* and continue/ }));
    await waitFor(() => expect(startModelDownload).toHaveBeenCalledTimes(2));
    expect(startModelDownload).toHaveBeenCalledWith('loki-retouch', 'http://srv');
    expect(startModelDownload).toHaveBeenCalledWith('qwen-image-2.1', 'http://srv');
    await screen.findByRole('progressbar');

    await act(async () => {
      await p;
    });
    expect(result).toBe('ready');
    await waitFor(() => expect(screen.queryByText('One-time download')).not.toBeInTheDocument());
  });

  it('"Not now" queues nothing and downloads nothing', async () => {
    render(<EngineSetupDialog />);
    fetchStatus.mockResolvedValueOnce(missing());
    const p = ensureEngines([{ input: 'reshoot "x.png"' }], '');
    fireEvent.click(await screen.findByRole('button', { name: 'Not now' }));
    await expect(p).resolves.toBe('cancelled');
    expect(startModelDownload).not.toHaveBeenCalled();
    expect(getEngineSetup()).toBeNull();
  });

  it('"Continue in background" leaves the download running; "Cancel download" stops it', async () => {
    render(<EngineSetupDialog />);
    const slow = [dep('loki-reshoot', 'installed', 55e6), dep('minimax-h3-ref2va', 'downloading', 42 * GB, { detail: { bytes_done: GB } })];

    fetchStatus.mockResolvedValueOnce(missing());
    fetchStatus.mockResolvedValue(slow);
    let p = ensureEngines([{ input: 'reshoot "x.png"' }], '');
    fireEvent.click(await screen.findByRole('button', { name: /Download .* and continue/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Continue in background' }));
    await expect(p).resolves.toBe('background');
    expect(cancelModelDownload).not.toHaveBeenCalled();

    fetchStatus.mockResolvedValueOnce(missing());
    p = ensureEngines([{ input: 'reshoot "x.png"' }], '');
    fireEvent.click(await screen.findByRole('button', { name: /Download .* and continue/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel download' }));
    await expect(p).resolves.toBe('cancelled');
    expect(cancelModelDownload).toHaveBeenCalledWith('loki-reshoot', '');
    expect(cancelModelDownload).toHaveBeenCalledWith('minimax-h3-ref2va', '');
  });

  it('shows a failure with a retry', async () => {
    render(<EngineSetupDialog />);
    fetchStatus.mockResolvedValueOnce(missing());
    const p = ensureEngines([{ input: 'retouch "x.png"' }], '');
    fetchStatus.mockResolvedValue([
      dep('loki-retouch', 'failed', 45e6, { error: 'HTTP 404 from github.com' }),
      dep('qwen-image-2.1', 'missing', 17 * GB),
    ]);
    fireEvent.click(await screen.findByRole('button', { name: /Download .* and continue/ }));
    expect(await screen.findByText('HTTP 404 from github.com')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await expect(p).resolves.toBe('cancelled');
  });
});

describe('submitJobs gate', () => {
  it('queues nothing and throws EngineSetupDeferred when the download is declined', async () => {
    render(<EngineSetupDialog />);
    const fetchMock = jest.fn();
    (global as any).fetch = fetchMock;
    fetchStatus.mockResolvedValueOnce(missing());
    const p = submitJobs([{ input: 'retouch "x.png"', fields: {}, label: 'x' }], {
      mediaServerBase: 'http://srv',
      authToken: null,
    });
    const assertion = expect(p).rejects.toBeInstanceOf(EngineSetupDeferred);
    fireEvent.click(await screen.findByRole('button', { name: 'Not now' }));
    await assertion;
    expect(fetchMock).not.toHaveBeenCalled(); // no /create
  });
});
