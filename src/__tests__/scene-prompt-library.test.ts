import { deletePrompt, resetPromptLibraryCache, savePrompt, searchPrompts, suggestName, type SavedPrompt } from '../renderer/components/scene/prompt-library';

describe('scene prompt library', () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetPromptLibraryCache();
  });

  it('saves per layer kind and persists', () => {
    savePrompt('still', 'Night', 'make it night with warm window light');
    savePrompt('motion', 'Flow', 'the water flows gently');
    const stored = JSON.parse(window.localStorage.getItem('loki.scene.prompts.v1') || '{}');
    expect(stored.still.map((s: SavedPrompt) => s.name)).toEqual(['Night']);
    expect(stored.motion.map((s: SavedPrompt) => s.name)).toEqual(['Flow']);
  });

  it('replaces one with the same name or text, and refuses empty ones', () => {
    savePrompt('still', 'Night', 'a');
    savePrompt('still', 'night', 'b');
    savePrompt('still', 'Other', 'b');
    expect(savePrompt('still', '', 'x')).toBeNull();
    expect(savePrompt('still', 'x', '  ')).toBeNull();
    const stored = JSON.parse(window.localStorage.getItem('loki.scene.prompts.v1') || '{}');
    expect(stored.still.map((s: SavedPrompt) => [s.name, s.prompt])).toEqual([['Other', 'b']]);
  });

  it('deletes', () => {
    const s = savePrompt('still', 'Night', 'a');
    deletePrompt('still', s?.id || '');
    expect(JSON.parse(window.localStorage.getItem('loki.scene.prompts.v1') || '{}').still).toEqual([]);
  });

  it('searches name and text for every word, recent first', () => {
    const list: SavedPrompt[] = [
      { id: '1', name: 'Snow', prompt: 'add soft falling snow', saved: 1 },
      { id: '2', name: 'Night', prompt: 'make it night with warm window light', saved: 2 },
      { id: '3', name: 'Storm', prompt: 'stormy night sky', saved: 3, used: 10 },
    ];
    expect(searchPrompts(list, '').map((s) => s.id)).toEqual(['3', '2', '1']);
    expect(searchPrompts(list, 'NIGHT').map((s) => s.id)).toEqual(['3', '2']);
    expect(searchPrompts(list, 'night warm').map((s) => s.id)).toEqual(['2']);
    expect(searchPrompts(list, 'snow')).toHaveLength(1);
    expect(searchPrompts(list, 'zebra')).toHaveLength(0);
  });

  it('suggests a short name from the prompt', () => {
    expect(suggestName('  add   snow ')).toBe('add snow');
    expect(suggestName('replace the sky with a dramatic stormy sunset over the hills')).toBe('replace the sky with a dramatic…');
  });
});
