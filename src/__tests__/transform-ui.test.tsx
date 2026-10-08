import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';

// jsdom has no AbortSignal.timeout (the job POST passes one as its signal).
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

import TransformSection from '../renderer/components/transform/transform-section';
import { TransformStudioView } from '../renderer/components/transform/studio-view';
import { getStudioRequest, closeTransformStudio } from '../renderer/components/transform/store';
import { settingsFor } from '../renderer/components/transform/intents';

const IMG = 'C:/media/photo.jpg';
const IMG2 = 'C:/media/other.png';
const VID = 'C:/media/clip.mp4';

describe('TransformSection (palette chips)', () => {
  let fetchMock: jest.Mock;
  beforeEach(() => {
    window.localStorage.clear();
    closeTransformStudio();
    fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job-1' }) });
    (global as any).fetch = fetchMock;
  });

  const setup = (paths: string[]) => {
    const onDone = jest.fn();
    const notify = jest.fn();
    render(<TransformSection paths={paths} authToken="tok" getVideoTime={() => undefined} onDone={onDone} notify={notify} />);
    return { onDone, notify };
  };

  it('offers the image presets for an image and hides combine/audio-only choices', () => {
    setup([IMG]);
    expect(screen.getByRole('button', { name: /Restore/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Upscale 2×/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Wallpaper/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Edit…/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Bring to life/ })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Combine/ })).toBeNull();
  });

  it('renders nothing for a selection no intent can use', () => {
    const { container } = render(
      <TransformSection paths={['C:/media/notes.txt']} authToken="tok" getVideoTime={() => undefined} onDone={() => {}} notify={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('a preset chip queues a retouch job straight away and closes the palette', async () => {
    const { onDone, notify } = setup([IMG, IMG2]);
    fireEvent.click(screen.getByRole('button', { name: /Restore/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/create$/);
    const body = JSON.parse(init.body);
    expect(body.input).toBe('retouch "C:/media/photo.jpg\nC:/media/other.png"');
    expect(body.fields).toMatchObject({ preset: 'restore', steps: '25' });
    expect(init.headers.Authorization).toBe('Bearer tok');
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(notify).toHaveBeenCalledWith('success', 'Queued', expect.any(String));
    expect(getStudioRequest()).toBeNull();
  });

  it('remembers the last upscale factor in the chip label', () => {
    window.localStorage.setItem('loki.transform.settings.v1', JSON.stringify({ upscale: { sizeMode: 'scale', scale: 4 } }));
    setup([IMG]);
    expect(screen.getByRole('button', { name: /Upscale 4×/ })).toBeTruthy();
  });

  it('shift-click opens the studio at the shape phase instead of running', () => {
    const { onDone } = setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Restore/ }), { shiftKey: true });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(getStudioRequest()).toMatchObject({ paths: [IMG], intent: 'restore', phase: 'shape' });
    expect(onDone).toHaveBeenCalled();
  });

  it('prompt-driven chips open the studio, and "Studio" opens the chooser', () => {
    setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Edit…/ }));
    expect(getStudioRequest()).toMatchObject({ intent: 'edit', phase: 'shape' });
    closeTransformStudio();
    fireEvent.click(screen.getByRole('button', { name: /Studio/ }));
    expect(getStudioRequest()).toMatchObject({ phase: 'choose' });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('reports a failed submission without opening anything', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500, json: async () => ({}) });
    const { notify } = setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Restore/ }));
    await waitFor(() => expect(notify).toHaveBeenCalledWith('error', 'Failed to Create Job', expect.any(String)));
  });

  it('on a single video, chips act on the current frame', () => {
    const onDone = jest.fn();
    render(<TransformSection paths={[VID]} authToken="tok" getVideoTime={() => 12.5} onDone={onDone} notify={() => {}} />);
    expect(screen.getByRole('button', { name: /Edit frame…/ })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Bring to life/ })).toBeNull();
  });
});

describe('TransformStudioView', () => {
  const baseProps = (over: Partial<React.ComponentProps<typeof TransformStudioView>> = {}) => ({
    paths: [IMG],
    intent: null as any,
    phase: 'choose' as const,
    settings: settingsFor('restore'),
    source: { width: 1000, height: 500 },
    queueAhead: 0,
    status: 'idle' as const,
    thumbUrl: (p: string) => p,
    onSettings: jest.fn(),
    onIntent: jest.fn(),
    onPhase: jest.fn(),
    onRun: jest.fn(),
    onClose: jest.fn(),
    onRemovePath: jest.fn(),
    onAgain: jest.fn(),
    ...over,
  });

  it('choose: unavailable intents are disabled with a reason; picking one reports it', () => {
    const props = baseProps();
    render(<TransformStudioView {...props} />);
    const combine = screen.getByText('Combine').closest('button') as HTMLButtonElement;
    expect(combine.disabled).toBe(true);
    expect(screen.getByText('Select 2 or more images')).toBeTruthy();
    fireEvent.click(screen.getByText('Upscale').closest('button') as HTMLButtonElement);
    expect(props.onIntent).toHaveBeenCalledWith('upscale');
  });

  it('shape (edit): needs text before it can continue, then reviews', () => {
    const empty = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    const { rerender } = render(<TransformStudioView {...empty} />);
    const review = screen.getByText(/Review →/).closest('button') as HTMLButtonElement;
    expect(review.disabled).toBe(true);
    expect(screen.getByText('Describe what should change')).toBeTruthy();
    const filled = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: 'make it night' }, onPhase: empty.onPhase });
    rerender(<TransformStudioView {...filled} />);
    const ok = screen.getByText(/Review →/).closest('button') as HTMLButtonElement;
    expect(ok.disabled).toBe(false);
    fireEvent.click(ok);
    expect(empty.onPhase).toHaveBeenCalledWith('review');
  });

  it('shape (edit): idea chips write into the prompt', () => {
    const props = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    render(<TransformStudioView {...props} />);
    fireEvent.click(screen.getByText('+ Night'));
    expect(props.onSettings).toHaveBeenCalledWith({ prompt: expect.stringMatching(/night/i) });
  });

  it('shape (upscale): scale options map to settings and show the resulting size', () => {
    const props = baseProps({ intent: 'upscale', phase: 'shape', settings: settingsFor('upscale', { scale: 2 }) });
    render(<TransformStudioView {...props} />);
    expect(screen.getAllByText('2000×1000').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('radio', { name: /^4×/ }));
    expect(props.onSettings).toHaveBeenCalledWith({ sizeMode: 'scale', scale: 4 });
    // 'Same' makes no sense for an upscale
    expect(screen.queryByRole('radio', { name: /^Same/ })).toBeNull();
  });

  it('shape (alive): duration under the trained range warns, 5 s does not', () => {
    const ok = baseProps({ intent: 'alive', phase: 'shape', settings: { ...settingsFor('alive'), describe: 'x', duration: 5 } });
    const { rerender } = render(<TransformStudioView {...ok} />);
    expect(screen.queryByText(/Shorter than the model was trained on/)).toBeNull();
    rerender(<TransformStudioView {...baseProps({ intent: 'alive', phase: 'shape', settings: { ...settingsFor('alive'), describe: 'x', duration: 2 } })} />);
    expect(screen.getByText(/Shorter than the model was trained on/)).toBeTruthy();
  });

  it('review: summarises, warns about the GPU queue and runs', () => {
    const props = baseProps({
      intent: 'upscale',
      phase: 'review',
      paths: [IMG, IMG2],
      queueAhead: 2,
      settings: settingsFor('upscale', { scale: 2 }),
    });
    render(<TransformStudioView {...props} />);
    expect(screen.getByText(/2 GPU jobs already queued ahead/)).toBeTruthy();
    expect(screen.getByText(/Built-in “upscale” prompt/)).toBeTruthy();
    expect(screen.getByText(/photo_up\.png/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Queue it/).closest('button') as HTMLButtonElement);
    expect(props.onRun).toHaveBeenCalled();
  });

  it('done: offers to run again or close', () => {
    const props = baseProps({ intent: 'restore', phase: 'review', status: 'done', queuedCount: 3 });
    render(<TransformStudioView {...props} />);
    expect(screen.getByText('3 jobs queued')).toBeTruthy();
    fireEvent.click(screen.getByText('Tweak and run again'));
    expect(props.onAgain).toHaveBeenCalled();
    fireEvent.click(screen.getByText('Done'));
    expect(props.onClose).toHaveBeenCalled();
  });

  it('stepper only lets you jump to phases that are ready', () => {
    const props = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    render(<TransformStudioView {...props} />);
    const review = screen.getByText('Review', { selector: '.ts-step-label' }).closest('button') as HTMLButtonElement;
    expect(review.disabled).toBe(true); // no prompt yet
    const choose = screen.getByText('Choose', { selector: '.ts-step-label' }).closest('button') as HTMLButtonElement;
    expect(choose.disabled).toBe(false);
    act(() => {
      fireEvent.click(choose);
    });
    expect(props.onPhase).toHaveBeenCalledWith('choose');
  });
});
