import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SavedPicker, { matchesSearch, savedEntries } from '../renderer/components/controls/saved-picker';
import { settingsFor } from '../renderer/components/transform/intents';
import { loadTemplates, resetTemplateCache, saveTemplate, templateSettings } from '../renderer/components/transform/store';

/* The palette's "Saved" list: saved Transform tasks and server Workflows in
 * one searchable list. */

const IMG = 'C:/media/photo.jpg';
const AUD = 'C:/media/song.mp3';
const WORKFLOWS = [
  { id: 'w1', name: 'Nightly tagging' },
  { id: 'w2', name: 'Archive to S3' },
];

beforeEach(() => {
  window.localStorage.clear();
  resetTemplateCache();
});

function setup(paths: string[] = [IMG], workflows = WORKFLOWS) {
  const onRunWorkflow = jest.fn();
  const onOpenTransform = jest.fn();
  render(<SavedPicker workflows={workflows} transformPaths={paths} onRunWorkflow={onRunWorkflow} onOpenTransform={onOpenTransform} />);
  return { onRunWorkflow, onOpenTransform, search: screen.getByLabelText('Search saved transforms and workflows') as HTMLInputElement };
}

const names = () => screen.getAllByRole('option').map((o) => o.querySelector('.saved-item-name')?.textContent);

describe('entries', () => {
  test('transforms that fit the selection and every workflow, by name', () => {
    saveTemplate('edit', 'Watercolor', 'turn it into a watercolor painting');
    saveTemplate('direct', 'Dance', 'She dances to <Audio 1>');
    saveTemplate('upscale', 'Big', '', { ...settingsFor('upscale'), scale: 4 });
    const all = [...['edit', 'direct', 'upscale'].flatMap((i) => loadTemplates(i as any).map((t) => ({ intent: i as any, t })))];
    expect(savedEntries(WORKFLOWS, all, [IMG]).map((e) => e.name)).toEqual(['Archive to S3', 'Big', 'Dance', 'Nightly tagging', 'Watercolor']);
    // Audio alone: only Direct can take it.
    expect(savedEntries([], all, [AUD]).map((e) => e.name)).toEqual(['Dance']);
    // Transform can't run here: workflows only.
    expect(savedEntries(WORKFLOWS, all, []).every((e) => e.kind === 'workflow')).toBe(true);
  });

  test('search matches name, summary and kind, every word', () => {
    const [entry] = savedEntries([], [{ intent: 'edit', t: { id: '1', name: 'Snow day', prompt: 'make it a snowy winter day' } }], [IMG]);
    expect(matchesSearch(entry, 'snow')).toBe(true);
    expect(matchesSearch(entry, 'winter edit')).toBe(true);
    expect(matchesSearch(entry, 'transform')).toBe(true);
    expect(matchesSearch(entry, 'workflow')).toBe(false);
    expect(matchesSearch(entry, 'winter beach')).toBe(false);
    const [wf] = savedEntries(WORKFLOWS.slice(0, 1), [], []);
    expect(matchesSearch(wf, 'workflow nightly')).toBe(true);
  });
});

describe('the list', () => {
  test('filters both kinds as you type; a workflow runs, a transform opens the flow', () => {
    saveTemplate('edit', 'Night', 'make it night');
    const { search, onRunWorkflow, onOpenTransform } = setup();
    expect(names()).toEqual(['Archive to S3', 'Night', 'Nightly tagging']);
    fireEvent.change(search, { target: { value: 'night' } });
    expect(names()).toEqual(['Night', 'Nightly tagging']);
    fireEvent.change(search, { target: { value: 'night workflow' } });
    expect(names()).toEqual(['Nightly tagging']);
    fireEvent.keyDown(search, { key: 'Enter' });
    expect(onRunWorkflow).toHaveBeenCalledWith(WORKFLOWS[0]);

    fireEvent.change(search, { target: { value: 'transform' } });
    fireEvent.click(screen.getByText('Night'));
    expect(onOpenTransform).toHaveBeenCalledWith('edit', expect.objectContaining({ name: 'Night', prompt: 'make it night' }));
    fireEvent.change(search, { target: { value: 'nothing like it' } });
    expect(screen.getByText('No matches')).toBeTruthy();
  });

  test("a saved transform opens with its saved settings (a preset's included)", () => {
    saveTemplate('upscale', 'Big and gentle', '', { ...settingsFor('upscale'), scale: 4, manualPrompt: 'upscale gently' });
    const { onOpenTransform } = setup([IMG], []);
    fireEvent.click(screen.getByText('Big and gentle'));
    const [intent, t] = onOpenTransform.mock.calls[0];
    expect(intent).toBe('upscale');
    expect(templateSettings(intent, t)).toMatchObject({ scale: 4, manualPrompt: 'upscale gently' });
  });

  test('delete with × or Shift+Delete, and undo puts it back in place', () => {
    saveTemplate('edit', 'A first', 'one');
    saveTemplate('edit', 'B second', 'two');
    saveTemplate('edit', 'C third', 'three');
    const { search } = setup([IMG], []);
    fireEvent.click(screen.getByLabelText('Delete saved transform B second'));
    expect(names()).toEqual(['A first', 'C third']);
    expect(loadTemplates('edit').map((t) => t.name)).toEqual(['A first', 'C third']);
    expect(screen.getByRole('status').textContent).toMatch(/Deleted “B second”/);
    fireEvent.click(screen.getByText('Undo'));
    expect(loadTemplates('edit').map((t) => t.name)).toEqual(['A first', 'B second', 'C third']);
    expect(screen.queryByRole('status')).toBeNull();

    // keyboard: the selected one
    fireEvent.keyDown(search, { key: 'ArrowDown' });
    fireEvent.keyDown(search, { key: 'Delete', shiftKey: true });
    expect(loadTemplates('edit').map((t) => t.name)).toEqual(['A first', 'C third']);
  });

  test('workflows cannot be deleted from here', () => {
    setup([IMG]);
    expect(screen.queryByLabelText(/Delete saved/)).toBeNull();
  });

  test('nothing saved: nothing shown', () => {
    const { container } = render(<SavedPicker workflows={[]} transformPaths={[IMG]} onRunWorkflow={jest.fn()} onOpenTransform={jest.fn()} />);
    expect(container.firstChild).toBeNull();
  });
});
