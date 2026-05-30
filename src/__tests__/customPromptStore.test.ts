import {
  getLastCustomPrompt,
  setLastCustomPrompt,
  getCachedDefaultPrompt,
  setCachedDefaultPrompt,
  __resetCustomPromptStoreForTests,
} from '../renderer/components/metadata/customPromptStore';

describe('customPromptStore', () => {
  beforeEach(() => {
    __resetCustomPromptStoreForTests();
  });

  it('returns empty string for last prompt by default', () => {
    expect(getLastCustomPrompt()).toBe('');
  });

  it('persists last prompt across reads within the session', () => {
    setLastCustomPrompt('hello world');
    expect(getLastCustomPrompt()).toBe('hello world');
  });

  it('ignores empty / whitespace-only writes to last prompt', () => {
    setLastCustomPrompt('keep me');
    setLastCustomPrompt('');
    setLastCustomPrompt('   ');
    expect(getLastCustomPrompt()).toBe('keep me');
  });

  it('returns null for cached default until set', () => {
    expect(getCachedDefaultPrompt()).toBeNull();
  });

  it('caches the fetched default prompt', () => {
    setCachedDefaultPrompt('default text');
    expect(getCachedDefaultPrompt()).toBe('default text');
  });

  it('overwrites last prompt when a new non-empty value is submitted', () => {
    setLastCustomPrompt('first prompt');
    setLastCustomPrompt('second prompt');
    expect(getLastCustomPrompt()).toBe('second prompt');
  });

  it('overwrites the cached default on a subsequent set', () => {
    setCachedDefaultPrompt('first');
    setCachedDefaultPrompt('second');
    expect(getCachedDefaultPrompt()).toBe('second');
  });
});
