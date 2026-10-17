import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';

// jsdom has no AbortSignal.timeout (the job POST passes one as its signal).
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

import TransformSection from '../renderer/components/transform/transform-section';
import { TransformFlowView } from '../renderer/components/transform/flow-view';
import {
  getFlowRequest,
  closeTransformFlow,
  loadLastRun,
  loadTemplates,
  resetLastRunCache,
  resetTemplateCache,
  rememberTemplateRun,
  saveLastRun,
  saveTemplate,
  savePins,
  loadRemembered,
  templateSettings,
} from '../renderer/components/transform/store';
import {
  buildJobs,
  NO_TARGET,
  referenceTokens,
  remapTokens,
  runSummary,
  settingsFor,
  type IntentId,
  type TransformSettings,
} from '../renderer/components/transform/intents';
import { validateShape } from '../renderer/components/transform/flow-view';
import { insertToken, suggestTemplateName } from '../renderer/components/transform/flow-view';

const IMG = 'C:/media/photo.jpg';
const IMG2 = 'C:/media/other.png';
const VID = 'C:/media/clip.mp4';

describe('TransformSection (palette chips)', () => {
  let fetchMock: jest.Mock;
  beforeEach(() => {
    window.localStorage.clear();
    resetLastRunCache();
    resetTemplateCache();
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
    expect(screen.getByRole('button', { name: /Animate/ })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Combine/ })).toBeNull();
  });

  it("a preset's saved task (saved from a result) gets a chip that opens it with its settings", () => {
    saveTemplate('upscale', 'Big and gentle', '', { ...settingsFor('upscale'), scale: 4, manualPrompt: 'upscale gently' });
    setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /☆ Big and gentle/ }));
    expect(getFlowRequest()).toMatchObject({ intent: 'upscale', phase: 'shape', settings: expect.objectContaining({ scale: 4, manualPrompt: 'upscale gently' }) });
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
    // the recipe rides along so the result can be saved as a task again (recipe.ts)
    expect(JSON.parse(body.fields.meta)).toMatchObject({ kind: 'loki-transform', intent: 'restore', paths: [IMG, IMG2] });
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

  it('prompt-driven chips open the flow, and "All options" opens the chooser', () => {
    setup([IMG]);
    fireEvent.click(screen.getByRole('button', { name: /Edit…/ }));
    expect(getFlowRequest()).toMatchObject({ intent: 'edit', phase: 'shape' });
    closeTransformFlow();
    fireEvent.click(screen.getByRole('button', { name: /All options/ }));
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
    expect(screen.queryByRole('button', { name: /Animate/ })).toBeNull();
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
    const direct = screen.getByText('Animate').closest('button') as HTMLButtonElement;
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
    expect(screen.getByText(/Upscale preset prompt/)).toBeTruthy();
    expect(screen.getByText(/photo_up\.png/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Queue it/).closest('button') as HTMLButtonElement);
    expect(props.onRun).toHaveBeenCalled();
  });

  it('done: offers to run again or close', () => {
    const props = baseProps({ intent: 'restore', phase: 'review', status: 'done', queuedCount: 3 });
    render(<TransformFlowView {...props} />);
    expect(screen.getByText('3 jobs queued')).toBeTruthy();
    fireEvent.click(screen.getByText('Adjust and run again'));
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
    fireEvent.click(screen.getByTitle(/click to insert <image2> into the prompt/));
    expect(ta.value).toBe('put the jacket of <image2> ');
    fireEvent.click(screen.getByTitle(/click to insert <image1> into the prompt/));
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

  it('review: "Write my own" starts from the expanded prompt, tokens insert into it, reset discards it', () => {
    render(<Harness intent="upscale" phase="review" paths={[IMG]} initial={{}} preview={upscalePreview} />);
    fireEvent.click(screen.getByText('Write my own'));
    const ta = screen.getByLabelText('Full prompt') as HTMLTextAreaElement;
    expect(ta.value).toBe('Enlarge <image1> faithfully.');
    fireEvent.change(ta, { target: { value: 'Sharpen' } });
    fireEvent.click(screen.getByTitle('Insert <image1> (photo.jpg) into the prompt'));
    expect(screen.getByTestId('manual').textContent).toBe('Sharpen <image1> ');
    fireEvent.change(ta, { target: { value: '' } });
    expect((screen.getByText(/Queue it/).closest('button') as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByText('Use the preset prompt'));
    expect(screen.getByTestId('manual').textContent).toBe('<null>');
    expect(screen.getByText('Enlarge <image1> faithfully.')).toBeTruthy();
  });

  it('review (edit): the prompt is verbatim, so editing it edits the prompt itself', () => {
    render(<Harness intent="edit" phase="review" paths={[IMG]} initial={{ prompt: 'make it night' }} />);
    expect(screen.getByText(/your words, sent as written/)).toBeTruthy();
    fireEvent.click(screen.getByText('Edit prompt'));
    const ta = screen.getByLabelText('Full prompt') as HTMLTextAreaElement;
    expect(ta.value).toBe('make it night');
    fireEvent.change(ta, { target: { value: 'make it dawn' } });
    expect(ta.value).toBe('make it dawn');
    expect(screen.getByTestId('manual').textContent).toBe('<null>');
  });
});

describe('TransformFlowView: saved prompts', () => {
  const noop = jest.fn();
  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
  });

  function Harness({ intent, initial }: { intent: any; initial: string }) {
    const [settings, setSettings] = React.useState({ ...settingsFor(intent), prompt: initial });
    return (
      <TransformFlowView
        paths={[IMG]}
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

  it('saves the current prompt under a name, then loads it back from the list', () => {
    const { unmount } = render(<Harness intent="edit" initial="" />);
    // Always visible next to the prompt, usable once there is something to save.
    expect((screen.getByText(/Save prompt/) as HTMLButtonElement).disabled).toBe(true);
    const ta = document.querySelector('textarea.ts-text') as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: 'turn it into a watercolor painting with soft edges and paper texture' } });
    fireEvent.click(screen.getByText(/Save prompt/));
    const name = screen.getByLabelText('Saved prompt name') as HTMLInputElement;
    expect(name.value).toBe('turn it into a watercolor…');
    fireEvent.change(name, { target: { value: 'Watercolor' } });
    fireEvent.keyDown(name, { key: 'Enter' });
    expect(screen.queryByLabelText('Saved prompt name')).toBeNull();
    // Already saved: the button says so while the prompt matches.
    expect(screen.queryByText(/Save prompt/)).toBeNull();
    expect(screen.getByText('★ Saved')).toBeTruthy();
    expect(JSON.parse(window.localStorage.getItem('loki.transform.templates.v1') || '{}').edit).toEqual([
      expect.objectContaining({ name: 'Watercolor', prompt: 'turn it into a watercolor painting with soft edges and paper texture' }),
    ]);
    unmount();

    render(<Harness intent="edit" initial="" />);
    fireEvent.click(screen.getByText('☆ Watercolor'));
    expect((document.querySelector('textarea.ts-text') as HTMLTextAreaElement).value).toBe(
      'turn it into a watercolor painting with soft edges and paper texture'
    );
  });

  it('keeps saved prompts per intent, replaces by name, and deletes', () => {
    saveTemplate('edit', 'Night', 'make it night');
    saveTemplate('edit', 'night', 'make it night, with stars');
    saveTemplate('direct', 'Dolly', 'slow dolly in on <Picture 1>');
    expect(loadTemplates('edit')).toEqual([expect.objectContaining({ name: 'night', prompt: 'make it night, with stars' })]);
    render(<Harness intent="edit" initial="" />);
    expect(screen.getByText('☆ night')).toBeTruthy();
    expect(screen.queryByText('☆ Dolly')).toBeNull();
    fireEvent.click(screen.getByLabelText('Delete saved task night'));
    expect(screen.queryByText('☆ night')).toBeNull();
    expect(loadTemplates('edit')).toEqual([]);
  });

  it('Esc in the name field cancels naming only', () => {
    render(<Harness intent="edit" initial="make it night" />);
    fireEvent.click(screen.getByText(/Save prompt/));
    fireEvent.keyDown(screen.getByLabelText('Saved prompt name'), { key: 'Escape' });
    expect(screen.queryByLabelText('Saved prompt name')).toBeNull();
    expect(screen.getByText(/Save prompt/)).toBeTruthy();
  });

  it('suggestTemplateName trims long prompts at a word', () => {
    expect(suggestTemplateName('  make   it night ')).toBe('make it night');
    expect(suggestTemplateName('replace the sky with a dramatic sunset over the ocean')).toBe('replace the sky with a dramatic…');
  });
});

describe('Transform: presets, custom and saved prompts are told apart', () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
    resetLastRunCache();
    closeTransformFlow();
  });

  const chooseProps = (over: Partial<React.ComponentProps<typeof TransformFlowView>> = {}) => ({
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
    onTemplate: jest.fn(),
    onPhase: jest.fn(),
    onRun: jest.fn(),
    onClose: jest.fn(),
    onRemovePath: jest.fn(),
    onAgain: jest.fn(),
    ...over,
  });

  it('choose: every card says whether it is a preset or custom', () => {
    render(<TransformFlowView {...chooseProps()} />);
    const card = (title: string) => screen.getByText(title, { exact: false, selector: 'strong' });
    expect(card('Restore').textContent).toMatch(/Preset/);
    expect(card('Animate').textContent).toMatch(/Preset/);
    expect(card('Edit').textContent).toMatch(/Custom/);
    expect(card('Direct').textContent).toMatch(/Custom/);
    expect(screen.queryByText('Saved tasks')).toBeNull();
  });

  it('choose: saved prompts get their own section and start their intent with the prompt', () => {
    saveTemplate('edit', 'Watercolor', 'turn it into a watercolor painting');
    const props = chooseProps();
    render(<TransformFlowView {...props} />);
    expect(screen.getByText('Saved tasks')).toBeTruthy();
    fireEvent.click(screen.getByText(/☆ Watercolor/).closest('button') as HTMLButtonElement);
    expect(props.onTemplate).toHaveBeenCalledWith('edit', expect.objectContaining({ prompt: 'turn it into a watercolor painting' }));
  });

  it('shape: chip rows are labelled by what they do', () => {
    saveTemplate('edit', 'Watercolor', 'turn it into a watercolor painting');
    render(<TransformFlowView {...chooseProps({ intent: 'edit', phase: 'shape', settings: { ...settingsFor('edit'), prompt: '' } })} />);
    expect(screen.getByRole('group', { name: 'Saved' })).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Ideas' })).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Add' })).toBeTruthy();
  });

  it('shape (preset): "Add to the preset" is up front, not hidden under More options', () => {
    render(<TransformFlowView {...chooseProps({ intent: 'restore', phase: 'shape' })} />);
    expect(screen.getByText('Add to the preset')).toBeTruthy();
  });

  it('palette: saved prompts are chips that open the flow with the prompt filled in', () => {
    saveTemplate('edit', 'Watercolor', 'turn it into a watercolor painting');
    const onDone = jest.fn();
    render(<TransformSection paths={[IMG]} authToken="tok" getVideoTime={() => undefined} onDone={onDone} notify={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /☆ Watercolor/ }));
    expect(getFlowRequest()).toMatchObject({ intent: 'edit', phase: 'shape', settings: expect.objectContaining({ prompt: 'turn it into a watercolor painting' }) });
    expect(onDone).toHaveBeenCalled();
  });
});

describe('Transform: the last run', () => {
  let fetchMock: jest.Mock;
  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
    resetLastRunCache();
    closeTransformFlow();
    fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job-1' }) });
    (global as any).fetch = fetchMock;
  });

  const editRun = (): TransformSettings => ({ ...settingsFor('edit'), prompt: 'make it night', quality: 'fine', variations: 2 });

  it('runSummary says what the run did and how', () => {
    expect(runSummary('edit', editRun())).toBe('“make it night” · fine · 2 variations');
    expect(runSummary('upscale', settingsFor('upscale', { scale: 4 }))).toBe('4× · balanced');
    expect(runSummary('wallpaper', settingsFor('wallpaper', { wallpaperTarget: 'phone' }))).toBe('phone · balanced');
    expect(runSummary('restore', { ...settingsFor('restore'), manualPrompt: 'clean it up' })).toMatch(/^your prompt “clean it up”/);
  });

  it('a one-click preset becomes the last run', async () => {
    render(<TransformSection paths={[IMG]} authToken="tok" getVideoTime={() => undefined} onDone={() => {}} notify={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /^Restore$/ }));
    await waitFor(() => expect(loadLastRun()).toMatchObject({ intent: 'restore' }));
  });

  it('palette ↻ re-runs the last run, prompt and settings included, on the new selection', async () => {
    saveLastRun('edit', editRun());
    const onDone = jest.fn();
    render(<TransformSection paths={[IMG2]} authToken="tok" getVideoTime={() => undefined} onDone={onDone} notify={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /^Edit again$/ }));
    const created = () => fetchMock.mock.calls.filter((c) => /\/create$/.test(String(c[0])));
    await waitFor(() => expect(created()).toHaveLength(2));
    const body = JSON.parse(created()[0][1].body);
    expect(body.input).toContain('C:/media/other.png');
    expect(body.fields).toMatchObject({ prompt: 'make it night' });
    await waitFor(() => expect(onDone).toHaveBeenCalled());
  });

  it('palette ↻ with Shift loads the last run into the flow to adjust', () => {
    saveLastRun('edit', editRun());
    render(<TransformSection paths={[IMG]} authToken="tok" getVideoTime={() => undefined} onDone={() => {}} notify={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /^Edit again$/ }), { shiftKey: true });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(getFlowRequest()).toMatchObject({ intent: 'edit', phase: 'shape', settings: expect.objectContaining({ prompt: 'make it night', quality: 'fine' }) });
  });

  it('choose: the last run is a card that loads it', () => {
    saveLastRun('edit', editRun());
    const onLoadRun = jest.fn();
    render(
      <TransformFlowView
        paths={[IMG]}
        intent={null}
        phase="choose"
        settings={settingsFor('restore')}
        source={null}
        queueAhead={0}
        status="idle"
        thumbUrl={(p: string) => p}
        onSettings={jest.fn()}
        onIntent={jest.fn()}
        onLoadRun={onLoadRun}
        onPhase={jest.fn()}
        onRun={jest.fn()}
        onClose={jest.fn()}
        onRemovePath={jest.fn()}
        onAgain={jest.fn()}
      />
    );
    expect(screen.getByText('Last run')).toBeTruthy();
    fireEvent.click(screen.getByText(/“make it night” · fine/).closest('button') as HTMLButtonElement);
    expect(onLoadRun).toHaveBeenCalledWith(expect.objectContaining({ intent: 'edit', settings: expect.objectContaining({ prompt: 'make it night' }) }));
  });

  it('done: offers to save the prompt that just ran', () => {
    render(
      <TransformFlowView
        paths={[IMG]}
        intent="edit"
        phase="review"
        settings={editRun()}
        source={null}
        queueAhead={0}
        status="done"
        queuedCount={2}
        thumbUrl={(p: string) => p}
        onSettings={jest.fn()}
        onIntent={jest.fn()}
        onPhase={jest.fn()}
        onRun={jest.fn()}
        onClose={jest.fn()}
        onRemovePath={jest.fn()}
        onAgain={jest.fn()}
      />
    );
    fireEvent.click(screen.getByText(/Save prompt/));
    const name = screen.getByLabelText('Saved prompt name') as HTMLInputElement;
    fireEvent.keyDown(name, { key: 'Enter' });
    expect(loadTemplates('edit')).toEqual([expect.objectContaining({ name: 'make it night', prompt: 'make it night' })]);
  });
});

describe('Transform: saved prompts keep their settings', () => {
  let fetchMock: jest.Mock;
  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
    resetLastRunCache();
    closeTransformFlow();
    fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job-1' }) });
    (global as any).fetch = fetchMock;
  });

  const sized = (): TransformSettings => ({ ...settingsFor('edit'), prompt: 'make it night', sizeMode: 'custom', customSize: '1920x1080', quality: 'fine' });

  it('saving a prompt keeps its custom size and quality, and bringing it back restores them', () => {
    const t = saveTemplate('edit', 'Night', 'make it night', sized()) as any;
    expect(t.settings).toMatchObject({ sizeMode: 'custom', customSize: '1920x1080', quality: 'fine' });
    expect(t.settings).not.toHaveProperty('prompt');
    expect(templateSettings('edit', t)).toMatchObject({ prompt: 'make it night', sizeMode: 'custom', customSize: '1920x1080', quality: 'fine' });
  });

  it('running a saved prompt refreshes the settings saved with it (older saves included)', () => {
    saveTemplate('edit', 'Night', 'make it night');
    saveTemplate('edit', 'Other', 'something else');
    rememberTemplateRun('edit', sized());
    const [night, other] = loadTemplates('edit');
    expect(night.settings).toMatchObject({ sizeMode: 'custom', customSize: '1920x1080' });
    expect(other.settings).toBeUndefined();
  });

  it('the palette chip opens the saved prompt with its saved size', () => {
    saveTemplate('edit', 'Night', 'make it night', sized());
    render(<TransformSection paths={[IMG]} authToken="tok" getVideoTime={() => undefined} onDone={() => {}} notify={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /☆ Night/ }));
    expect(getFlowRequest()).toMatchObject({
      intent: 'edit',
      settings: expect.objectContaining({ prompt: 'make it night', sizeMode: 'custom', customSize: '1920x1080' }),
    });
  });

  it('set up: picking a saved chip restores its size too', () => {
    saveTemplate('edit', 'Night', 'make it night', sized());
    const onSettings = jest.fn();
    render(
      <TransformFlowView
        paths={[IMG]}
        intent="edit"
        phase="shape"
        settings={{ ...settingsFor('edit'), prompt: '' }}
        source={null}
        queueAhead={0}
        status="idle"
        thumbUrl={(p: string) => p}
        onSettings={onSettings}
        onIntent={jest.fn()}
        onPhase={jest.fn()}
        onRun={jest.fn()}
        onClose={jest.fn()}
        onRemovePath={jest.fn()}
        onAgain={jest.fn()}
      />
    );
    fireEvent.click(screen.getByText('☆ Night'));
    expect(onSettings).toHaveBeenCalledWith(expect.objectContaining({ prompt: 'make it night', sizeMode: 'custom', customSize: '1920x1080' }));
  });
});

describe('Transform: pinned references', () => {
  const R1 = 'C:/refs/style.png';
  const R2 = 'C:/refs/face.jpg';
  const editWith = (pinned: string[], prompt = '<image1> in the style of <image2>'): TransformSettings => ({ ...settingsFor('edit'), prompt, pinned });
  const inputsOf = (input: string) => (input.match(/"([^"]*)"/) as RegExpMatchArray)[1].split(String.fromCharCode(10));

  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
    resetLastRunCache();
    closeTransformFlow();
  });

  it('edit: each selected image is <image1> of its own run, the pins follow it', () => {
    const built = buildJobs('edit', editWith([R1, R2]), [IMG, IMG2], { random: () => 0.5 });
    expect(built.error).toBeUndefined();
    expect(built.jobs).toHaveLength(2);
    expect(inputsOf(built.jobs[0].input)).toEqual([IMG, R1, R2]);
    expect(inputsOf(built.jobs[1].input)).toEqual([IMG2, R1, R2]);
    expect(built.jobs[0].fields).toMatchObject({ combine: '1' });
    expect(referenceTokens('edit', [IMG, IMG2], false, [R1, R2])).toEqual({ [IMG]: '<image1>', [IMG2]: '<image1>', [R1]: '<image2>', [R2]: '<image3>' });
  });

  it('edit: a single selected image plus pins is one run', () => {
    const built = buildJobs('edit', editWith([R1]), [IMG], { random: () => 0.5 });
    expect(built.jobs).toHaveLength(1);
    expect(inputsOf(built.jobs[0].input)).toEqual([IMG, R1]);
  });

  it('edit: selecting only a pinned image asks for the image to edit', () => {
    expect(buildJobs('edit', editWith([R1]), [R1]).error).toBe(NO_TARGET);
    expect(validateShape('edit', editWith([R1]), [R1])).toBe(NO_TARGET);
  });

  it('direct: the selected item comes first, then the pins', () => {
    const s = { ...settingsFor('direct'), prompt: '<Picture 1> walks into <Picture 2>', pinned: [R1] };
    const built = buildJobs('direct', s, [IMG], { random: () => 0.5 });
    expect(built.jobs).toHaveLength(1);
    expect(inputsOf(built.jobs[0].input)).toEqual([IMG, R1]);
    expect(referenceTokens('direct', [IMG], false, [R1])).toEqual({ [IMG]: '<Picture 1>', [R1]: '<Picture 2>' });
    expect(referenceTokens('direct', [VID], false, [R1])).toEqual({ [VID]: '<Video 1>', [R1]: '<Picture 1>' });
  });

  it('presets ignore pins', () => {
    const built = buildJobs('restore', { ...settingsFor('restore'), pinned: [R1] }, [IMG]);
    expect(built.jobs).toHaveLength(1);
    expect(inputsOf(built.jobs[0].input)).toEqual([IMG]);
  });

  it('pins are remembered per intent and come back with the intent', () => {
    savePins('edit', [R1]);
    expect(loadRemembered('edit').pinned).toEqual([R1]);
    expect(loadRemembered('direct').pinned).toEqual([]);
    expect(loadRemembered('restore').pinned).toBeUndefined();
  });

  it('a saved prompt keeps its pins', () => {
    const t = saveTemplate('edit', 'Styled', '<image1> in the style of <image2>', editWith([R1])) as any;
    expect(templateSettings('edit', t).pinned).toEqual([R1]);
  });

  it('pinning a selected image keeps the prompt pointing at the same files', () => {
    // [A, B, C] as one run: C is <image3>. Pin B: A and C are each <image1> of their run, B is <image2>.
    const C = 'C:/media/third.png';
    expect(remapTokens('<image3> wearing <image2>', 'edit', [IMG, IMG2, C], [IMG, IMG2, C], [], [IMG2])).toBe('<image1> wearing <image2>');
  });

  it('set up: thumbnails offer the pin, and pinned references show with the selection', () => {
    const onPin = jest.fn();
    render(
      <TransformFlowView
        paths={[IMG]}
        intent="edit"
        phase="shape"
        settings={editWith([R1])}
        source={null}
        queueAhead={0}
        status="idle"
        thumbUrl={(p: string) => p}
        onSettings={jest.fn()}
        onIntent={jest.fn()}
        onPhase={jest.fn()}
        onRun={jest.fn()}
        onClose={jest.fn()}
        onRemovePath={jest.fn()}
        onPin={onPin}
        onAgain={jest.fn()}
      />
    );
    expect(screen.getByText(/1 pinned/)).toBeTruthy();
    fireEvent.click(screen.getByLabelText('Unpin style.png'));
    expect(onPin).toHaveBeenCalledWith(R1, false);
    fireEvent.click(screen.getByLabelText('Pin photo.jpg as a reference'));
    expect(onPin).toHaveBeenCalledWith(IMG, true);
  });

  it('runSummary mentions the pins', () => {
    expect(runSummary('edit', editWith([R1, R2], 'x'))).toMatch(/2 pinned$/);
  });
});
