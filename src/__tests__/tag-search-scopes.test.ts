// The command palette and the taxonomy sidebar search DIFFERENT slices of the
// tag table, and that split is load-bearing for palette latency.
//
// "Suggested" is the ONNX autotagger's catch-all bucket: 183,548 of 189,122
// tags on a real library (97%). Fetching it costs ~475ms of SQLite plus an IPC
// clone of the whole table, and Fuse-indexing it means every palette keystroke
// fuzzy-matches 189K entries. The palette doesn't show those tags prominently
// and doesn't need them; the sidebar exists to browse the whole taxonomy and
// does. Measured after the split: 5,574 rows in 37.6ms.
//
// These tests pin the contract, because a regression here is invisible — the
// palette would still WORK, just slowly.
import {
  SUGGESTED_CATEGORY,
  applyScope,
  excludedCategories,
} from '../renderer/search/tag-scopes';

const mockInvoke = jest.fn(async (..._args: any[]) => [] as unknown[]);

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  invoke: (...args: any[]) => mockInvoke(...args),
}));

jest.mock('../renderer/state', () => {
  const React = require('react');
  return {
    __esModule: true,
    GlobalStateContext: React.createContext({ libraryService: {} }),
  };
});

jest.mock('@xstate/react', () => ({
  useSelector: () => '',
}));

jest.mock('../renderer/search/tag-search-service', () => ({
  __esModule: true,
  indexTags: jest.fn(),
  searchTags: jest.fn(),
}));

// Imported after the mocks above, which must be registered first.
import { loadTagsForScope, tagScopeQueryKey } from '../renderer/hooks/useTagSearch';

beforeEach(() => mockInvoke.mockClear());

describe('excludedCategories', () => {
  it('leaves nothing out of the full scope', () => {
    expect(excludedCategories('all')).toEqual([]);
  });

  it('drops the autotagger bucket from the curated scope', () => {
    expect(excludedCategories('curated')).toEqual([SUGGESTED_CATEGORY]);
  });
});

describe('applyScope', () => {
  const tags = [
    { label: 'moody', category: 'Style' },
    { label: '1girl', category: SUGGESTED_CATEGORY },
    { label: 'orphan' }, // no category at all
  ];

  it('is a pass-through for the full scope', () => {
    expect(applyScope(tags, 'all')).toBe(tags);
  });

  it('drops suggested tags but keeps uncategorised ones', () => {
    expect(applyScope(tags, 'curated').map((t) => t.label)).toEqual([
      'moody',
      'orphan',
    ]);
  });
});

describe('loadTagsForScope', () => {
  it('asks the backend for everything in the full scope', async () => {
    await loadTagsForScope('all');
    expect(mockInvoke).toHaveBeenCalledWith('load-all-tags', []);
  });

  it('pushes the exclusion DOWN to the backend for the curated scope', async () => {
    // The point is that these rows are never read, never serialized across IPC
    // and never indexed — filtering client-side would save none of that.
    await loadTagsForScope('curated');
    expect(mockInvoke).toHaveBeenCalledWith('load-all-tags', [
      [SUGGESTED_CATEGORY],
    ]);
  });

  it('still filters client-side if the backend ignores the exclusion', async () => {
    // An older media-server won't know the excludeCategory param. The index
    // must not silently balloon back to 189K entries because of it.
    mockInvoke.mockResolvedValueOnce([
      { label: 'moody', category: 'Style', weight: 0 },
      { label: '1girl', category: SUGGESTED_CATEGORY, weight: 0 },
    ]);
    const rows = await loadTagsForScope('curated');
    expect(rows.map((r) => r.label)).toEqual(['moody']);
  });
});

describe('tagScopeQueryKey', () => {
  it('keys the two scopes apart so they cache side by side', () => {
    // Sharing a key would make the sidebar's full fetch evict the palette's
    // curated one (and vice versa), reintroducing the cost on every switch.
    expect(tagScopeQueryKey('all', 's1')).not.toEqual(
      tagScopeQueryKey('curated', 's1')
    );
  });

  it('re-keys per session so a DB swap refetches', () => {
    expect(tagScopeQueryKey('curated', 's1')).not.toEqual(
      tagScopeQueryKey('curated', 's2')
    );
  });
});
