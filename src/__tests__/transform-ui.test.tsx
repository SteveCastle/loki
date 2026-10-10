import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';

// jsdom has no AbortSignal.timeout (the job POST passes one as its signal).
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

import TransformSection from '../renderer/components/transform/transform-section';
import { TransformFlowView } from '../renderer/components/transform/flow-view';
import { getFlowRequest, closeTransformFlow } from '../renderer/components/transform/store';
import { settingsFor, type IntentId, type TransformSettings } from '../renderer/components/transform/intents';
import { insertToken } from '../renderer/components/transform/flow-view';

const IMG = 'C:/media/photo.jpg';
const IMG2 = 'C:/media/other.png';
const VID = 'C:/media/clip.mp4';

describe('TransformSection (palette chips)', () => {
  let fetchMock: jest.Mock;
  beforeEach(() => {
    window.localStorage.clear();
    closeTransformFlow();
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
    // (the engine status check comes first; nothing is missing here, so no confirmation)
    const created = () => fetchMock.mock.calls.filter((c) => /\/create$/.test(String(c[0])));
    await waitFor(() => expect(created()).toHaveLength(1));
    const [url, init] = created()[0];
    expect(url).toMatch(/\/create$/);
    const body = JSON.parse(init.body);
    expect(body.input).toBe('retouch "C:/media/photo.jpg\nC:/media/other.png"');
    expect(body.fields).toMatchObject({ preset: 'restore', steps: '25' });
    expect(init.headers.Authorization).toBe('Bearer tok');
    await waitFor(() => expect(onDone).toHaveBeenCalled());
    expect(getFlowRequest()).toBeNull();
  });

  it('remembers the last upscale factor in the chip label', () => {
    window.localStorage.setItem('loki.transform.settings.v1', JSON.stringify({ upscale: { sizeMode: 'scale', scale: 4 } }));
    setup([IMG]);
    expect(screen.getByRole('button', { name: /Upscale 4×/ })).toBeTruthy();
  });

  it('shift-click opens the full options at the shape phase instead of running', () => {
    const { onDone } = setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Restore/ }), { shiftKey: true });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(getFlowRequest()).toMatchObject({ paths: [IMG], intent: 'restore', phase: 'shape' });
    expect(onDone).toHaveBeenCalled();
  });

  it('prompt-driven chips open the flow, and "Customize" opens the chooser', () => {
    setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Edit…/ }));
    expect(getFlowRequest()).toMatchObject({ intent: 'edit', phase: 'shape' });
    closeTransformFlow();
    fireEvent.click(screen.getByRole('button', { name: /Customize/ }));
    expect(getFlowRequest()).toMatchObject({ phase: 'choose' });
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

describe('TransformFlowView', () => {
  const baseProps = (over: Partial<React.ComponentProps<typeof TransformFlowView>> = {}) => ({
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
    const props = baseProps({ paths: [IMG2] });
    render(<TransformFlowView {...props} />);
    const direct = screen.getByText('Bring to life').closest('button') as HTMLButtonElement;
    expect(direct.disabled).toBe(false);
    expect(screen.queryByText('Combine')).toBeNull();
    fireEvent.click(screen.getByText('Upscale').closest('button') as HTMLButtonElement);
    expect(props.onIntent).toHaveBeenCalledWith('upscale');
  });

  it('shape (edit): needs text before it can continue, then reviews', () => {
    const empty = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    const { rerender } = render(<TransformFlowView {...empty} />);
    const review = screen.getByText(/Review →/).closest('button') as HTMLButtonElement;
    expect(review.disabled).toBe(true);
    expect(screen.getByText('Describe what should change')).toBeTruthy();
    const filled = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: 'make it night' }, onPhase: empty.onPhase });
    rerender(<TransformFlowView {...filled} />);
    const ok = screen.getByText(/Review →/).closest('button') as HTMLButtonElement;
    expect(ok.disabled).toBe(false);
    fireEvent.click(ok);
    expect(empty.onPhase).toHaveBeenCalledWith('review');
  });

  it('shape (edit): idea chips write into the prompt', () => {
    const props = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    render(<TransformFlowView {...props} />);
    fireEvent.click(screen.getByText('+ Night'));
    expect(props.onSettings).toHaveBeenCalledWith({ prompt: expect.stringMatching(/night/i) });
  });

  it('shape (upscale): scale options map to settings and show the resulting size', () => {
    const props = baseProps({ intent: 'upscale', phase: 'shape', settings: settingsFor('upscale', { scale: 2 }) });
    render(<TransformFlowView {...props} />);
    expect(screen.getAllByText('2000×1000').length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('radio', { name: /^4×/ }));
    expect(props.onSettings).toHaveBeenCalledWith({ sizeMode: 'scale', scale: 4 });
    // 'Same' makes no sense for an upscale
    expect(screen.queryByRole('radio', { name: /^Same/ })).toBeNull();
  });

  it('shape (alive): duration under the trained range warns, 5 s does not', () => {
    const ok = baseProps({ intent: 'alive', phase: 'shape', settings: { ...settingsFor('alive'), describe: 'x', duration: 5 } });
    const { rerender } = render(<TransformFlowView {...ok} />);
    expect(screen.queryByText(/Shorter than the model was trained on/)).toBeNull();
    rerender(<TransformFlowView {...baseProps({ intent: 'alive', phase: 'shape', settings: { ...settingsFor('alive'), describe: 'x', duration: 2 } })} />);
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
    render(<TransformFlowView {...props} />);
    expect(screen.getByText(/2 GPU jobs already queued ahead/)).toBeTruthy();
    expect(screen.getByText(/built-in “upscale” prompt/)).toBeTruthy();
    expect(screen.getByText(/photo_up\.png/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Queue it/).closest('button') as HTMLButtonElement);
    expect(props.onRun).toHaveBeenCalled();
  });

  it('done: offers to run again or close', () => {
    const props = baseProps({ intent: 'restore', phase: 'review', status: 'done', queuedCount: 3 });
    render(<TransformFlowView {...props} />);
    expect(screen.getByText('3 jobs queued')).toBeTruthy();
    fireEvent.click(screen.getByText('Tweak and run again'));
    expect(props.onAgain).toHaveBeenCalled();
    fireEvent.click(screen.getByText('Done'));
    expect(props.onClose).toHaveBeenCalled();
  });

  it('stepper only lets you jump to phases that are ready', () => {
    const props = baseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } });
    render(<TransformFlowView {...props} />);
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

describe('TransformFlowView: typing and prompt tokens', () => {
  const noop = jest.fn();
  // A real stateful parent, so controlled inputs behave as in the app.
  function Harness({ intent, paths, initial }: { intent: any; paths: string[]; initial: Partial<ReturnType<typeof settingsFor>> }) {
    const [settings, setSettings] = React.useState({ ...settingsFor(intent), ...initial });
    return (
      <TransformFlowView
        paths={paths}
        intent={intent}
        phase="shape"
        settings={settings}
        source={{ width: 1000, height: 500 }}
        queueAhead={0}
        status="idle"
        thumbUrl={(p: string) => p}
        onSettings={(patch: Partial<TransformSettings>) => setSettings((cur) => ({ ...cur, ...patch }))}
        onIntent={noop}
        onPhase={noop}
        onRun={noop}
        onClose={noop}
        onRemovePath={noop}
        onAgain={noop}
      />
    );
  }

  it('custom size: digits typed one at a time stay in the fields', () => {
    render(<Harness intent="upscale" paths={[IMG]} initial={{ sizeMode: 'custom', customSize: '' }} />);
    const w = screen.getByPlaceholderText('width') as HTMLInputElement;
    const h = screen.getByPlaceholderText('height') as HTMLInputElement;
    fireEvent.change(w, { target: { value: '1' } });
    expect(w.value).toBe('1'); // a lone "1" is not a size yet, but must not be wiped
    fireEvent.change(w, { target: { value: '1920' } });
    fireEvent.change(h, { target: { value: '1' } });
    expect(h.value).toBe('1');
    fireEvent.change(h, { target: { value: '1080' } });
    expect(w.value).toBe('1920');
    expect(h.value).toBe('1080');
  });

  it('typing in a field does not reach the app hotkeys', () => {
    render(<Harness intent="upscale" paths={[IMG]} initial={{ sizeMode: 'custom', customSize: '' }} />);
    const seen = jest.fn();
    window.addEventListener('keydown', seen);
    fireEvent.keyDown(screen.getByPlaceholderText('width'), { key: '5', code: 'Digit5' });
    window.removeEventListener('keydown', seen);
    expect(seen).not.toHaveBeenCalled();
  });

  it('clicking a reference thumbnail puts its tag into the prompt, spaced from the words around it', () => {
    render(<Harness intent="edit" paths={[IMG, IMG2]} initial={{ prompt: 'put the jacket of' }} />);
    const ta = screen.getByRole('textbox') as HTMLTextAreaElement;
    expect(ta.value).toBe('put the jacket of');
    fireEvent.click(screen.getByTitle(/click to put <image2> in the prompt/));
    expect(ta.value).toBe('put the jacket of <image2> ');
    fireEvent.click(screen.getByTitle(/click to put <image1> in the prompt/));
    expect(ta.value).toBe('put the jacket of <image2> <image1> ');
  });

  it('insertToken splices at the caret or over a selection', () => {
    expect(insertToken('', '<Picture 1>', 0, 0)).toEqual({ text: '<Picture 1> ', caret: 12 });
    expect(insertToken('a b', '<Video 1>', 1, 1).text).toBe('a <Video 1> b');
    expect(insertToken('make X glow', '<Picture 2>', 5, 6).text).toBe('make <Picture 2> glow');
  });
});

describe('TransformFlowView: references and the full prompt', () => {
  const noop = jest.fn();
  function Harness({
    intent,
    paths,
    phase = 'shape',
    initial,
    preview,
    onMakeFirst,
  }: {
    intent: IntentId;
    paths: string[];
    phase?: 'shape' | 'review';
    initial: Partial<TransformSettings>;
    preview?: React.ComponentProps<typeof TransformFlowView>['preview'];
    onMakeFirst?: (p: string) => void;
  }) {
    const [settings, setSettings] = React.useState({ ...settingsFor(intent), ...initial });
    return (
      <>
        <TransformFlowView
          paths={paths}
          intent={intent}
          phase={phase}
          settings={settings}
          source={{ width: 1000, height: 500 }}
          queueAhead={0}
          status="idle"
          thumbUrl={(p: string) => p}
          onSettings={(patch: Partial<TransformSettings>) => setSettings((cur) => ({ ...cur, ...patch }))}
          onIntent={noop}
          onPhase={noop}
          onRun={noop}
          onClose={noop}
          onRemovePath={noop}
          onMakeFirst={onMakeFirst}
          onAgain={noop}
          preview={preview}
        />
        <output data-testid="manual">{settings.manualPrompt === null ? '<null>' : settings.manualPrompt}</output>
      </>
    );
  }

  it('shape (edit, 2 images): token chips name each image and insert their token; the second can become <image1>', () => {
    const onMakeFirst = jest.fn();
    render(<Harness intent="edit" paths={[IMG, IMG2]} initial={{ prompt: 'use the sky of' }} onMakeFirst={onMakeFirst} />);
    expect(screen.getByText(/Refer to the 2 images by their tokens/)).toBeTruthy();
    const ta = screen.getByRole('textbox') as HTMLTextAreaElement;
    fireEvent.click(screen.getByTitle('Insert <image2> (other.png) into the prompt'));
    expect(ta.value).toBe('use the sky of <image2> ');
    expect(screen.getByText(/is the image being edited/)).toBeTruthy();
    fireEvent.click(screen.getByText('make <image1>'));
    expect(onMakeFirst).toHaveBeenCalledWith(IMG2);
  });

  it('shape: a token that names no selected file blocks the review', () => {
    render(<Harness intent="edit" paths={[IMG, IMG2]} initial={{ prompt: 'put <image3> on <image1>' }} />);
    expect(screen.getAllByText(/<image3> matches none of the selected files/).length).toBeGreaterThan(0);
    expect((screen.getByText(/Review →/).closest('button') as HTMLButtonElement).disabled).toBe(true);
  });

  const upscalePreview = {
    status: 'ok' as const,
    data: {
      engine: 'loki-retouch',
      prompt: 'Enlarge <image1> faithfully.',
      modelInput: '<|im_start|>system\n...<image1><|vision_start|><|image_pad|><|vision_end|>Enlarge <image1> faithfully.',
      references: [{ token: '<image1>', path: IMG, kind: 'image' }],
      runs: 1,
      fromEngine: true,
    },
  };

  it('review: shows the prompt the engine expands, and the raw model input on request', () => {
    render(<Harness intent="upscale" phase="review" paths={[IMG]} initial={{}} preview={upscalePreview} />);
    expect(screen.getByText('Enlarge <image1> faithfully.')).toBeTruthy();
    expect(screen.getByText(/as loki-retouch expands it/)).toBeTruthy();
    fireEvent.click(screen.getByText('Raw model input'));
    expect(screen.getByText(/<\|vision_start\|>/)).toBeTruthy();
  });

  it('review: "Edit full prompt" starts from the expanded prompt, tokens insert into it, reset discards it', () => {
    render(<Harness intent="upscale" phase="review" paths={[IMG]} initial={{}} preview={upscalePreview} />);
    fireEvent.click(screen.getByText('Edit full prompt'));
    const ta = screen.getByLabelText('Full prompt') as HTMLTextAreaElement;
    expect(ta.value).toBe('Enlarge <image1> faithfully.');
    fireEvent.change(ta, { target: { value: 'Sharpen' } });
    fireEvent.click(screen.getByTitle('Insert <image1> (photo.jpg) into the prompt'));
    expect(screen.getByTestId('manual').textContent).toBe('Sharpen <image1> ');
    fireEvent.change(ta, { target: { value: '' } });
    expect((screen.getByText(/Queue it/).closest('button') as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByText('Reset to generated'));
    expect(screen.getByTestId('manual').textContent).toBe('<null>');
    expect(screen.getByText('Enlarge <image1> faithfully.')).toBeTruthy();
  });

  it('review (edit): the prompt is verbatim, so editing it edits the prompt itself', () => {
    render(<Harness intent="edit" phase="review" paths={[IMG]} initial={{ prompt: 'make it night' }} />);
    expect(screen.getByText(/your words, sent as written/)).toBeTruthy();
    fireEvent.click(screen.getByText('Edit full prompt'));
    const ta = screen.getByLabelText('Full prompt') as HTMLTextAreaElement;
    expect(ta.value).toBe('make it night');
    fireEvent.change(ta, { target: { value: 'make it dawn' } });
    expect(ta.value).toBe('make it dawn');
    expect(screen.getByTestId('manual').textContent).toBe('<null>');
  });
});
