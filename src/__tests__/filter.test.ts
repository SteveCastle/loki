import filter from '../renderer/filter';
import type { Item } from '../renderer/state';

// Mock the file-types module
jest.mock('file-types', () => ({
  getMediaType: (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext || '')) {
      return 'static';
    }
    if (['mp4', 'mov', 'webm', 'mkv'].includes(ext || '')) {
      return 'motion';
    }
    if (['mp3', 'wav', 'flac'].includes(ext || '')) {
      return 'audio';
    }
    return 'all';
  },
}));

describe('filter', () => {
  const createItem = (
    path: string,
    mtimeMs: number = Date.now(),
    options: Partial<Item> = {}
  ): Item => ({
    path,
    mtimeMs,
    ...options,
  });

  const sampleLibrary: Item[] = [
    createItem('/photos/zebra.jpg', 1000),
    createItem('/photos/apple.png', 2000),
    createItem('/videos/movie.mp4', 3000),
    createItem('/music/song.mp3', 4000),
    createItem('/photos/banana.jpg', 5000),
  ];

  beforeEach(() => {
    // Reset memoization cache between tests by using unique libraryLoadIds
  });

  describe('media type filtering', () => {
    it('should return all items when filter is "all"', () => {
      const result = filter('test-1', '', sampleLibrary, 'all', 'name');
      expect(result).toHaveLength(5);
    });

    it('should filter to only static images when filter is "static"', () => {
      const result = filter('test-2', '', sampleLibrary, 'static', 'name');
      expect(result).toHaveLength(3);
      expect(result.every((item) => item.path.match(/\.(jpg|png)$/))).toBe(
        true
      );
    });

    it('should filter to only videos when filter is "video"', () => {
      const result = filter('test-3', '', sampleLibrary, 'video', 'name');
      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('/videos/movie.mp4');
    });

    it('should filter to only audio when filter is "audio"', () => {
      const result = filter('test-4', '', sampleLibrary, 'audio', 'name');
      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('/music/song.mp3');
    });
  });

  describe('sorting by name', () => {
    it('should sort alphabetically by path using natural compare', () => {
      const result = filter('test-5', '', sampleLibrary, 'all', 'name');
      const paths = result.map((item) => item.path);
      expect(paths).toEqual([
        '/music/song.mp3',
        '/photos/apple.png',
        '/photos/banana.jpg',
        '/photos/zebra.jpg',
        '/videos/movie.mp4',
      ]);
    });

    it('should handle natural sorting with numbers correctly', () => {
      const numberedItems: Item[] = [
        createItem('/photo1.jpg', 1000),
        createItem('/photo10.jpg', 2000),
        createItem('/photo2.jpg', 3000),
        createItem('/photo20.jpg', 4000),
      ];
      const result = filter('test-6', '', numberedItems, 'all', 'name');
      const paths = result.map((item) => item.path);
      expect(paths).toEqual([
        '/photo1.jpg',
        '/photo2.jpg',
        '/photo10.jpg',
        '/photo20.jpg',
      ]);
    });
  });

  describe('sorting by date', () => {
    it('should sort by modification time descending (newest first)', () => {
      const result = filter('test-7', '', sampleLibrary, 'all', 'date');
      const times = result.map((item) => item.mtimeMs);
      expect(times).toEqual([5000, 4000, 3000, 2000, 1000]);
    });
  });

  describe('sorting by weight', () => {
    it('should sort by weight ascending', () => {
      const weightedItems: Item[] = [
        createItem('/a.jpg', 1000, { weight: 30 }),
        createItem('/b.jpg', 2000, { weight: 10 }),
        createItem('/c.jpg', 3000, { weight: 20 }),
      ];
      const result = filter('test-8', '', weightedItems, 'all', 'weight');
      expect(result.map((item) => item.weight)).toEqual([10, 20, 30]);
    });

    it('should handle undefined weights as 0', () => {
      const weightedItems: Item[] = [
        createItem('/a.jpg', 1000, { weight: 10 }),
        createItem('/b.jpg', 2000),
        createItem('/c.jpg', 3000, { weight: 5 }),
      ];
      const result = filter('test-9', '', weightedItems, 'all', 'weight');
      expect(result.map((item) => item.weight)).toEqual([undefined, 5, 10]);
    });
  });

  describe('sorting by elo', () => {
    it('should sort by elo descending (highest first)', () => {
      const eloItems: Item[] = [
        createItem('/a.jpg', 1000, { elo: 1400 }),
        createItem('/b.jpg', 2000, { elo: 1600 }),
        createItem('/c.jpg', 3000, { elo: 1500 }),
      ];
      const result = filter('test-10', '', eloItems, 'all', 'elo');
      expect(result.map((item) => item.elo)).toEqual([1600, 1500, 1400]);
    });

    it('should default missing elo to 1500', () => {
      const eloItems: Item[] = [
        createItem('/a.jpg', 1000, { elo: 1400 }),
        createItem('/b.jpg', 2000),
        createItem('/c.jpg', 3000, { elo: 1600 }),
      ];
      const result = filter('test-11', '', eloItems, 'all', 'elo');
      // 1600 > 1500 (default) > 1400
      expect(result.map((item) => item.path)).toEqual([
        '/c.jpg',
        '/b.jpg',
        '/a.jpg',
      ]);
    });
  });

  describe('shuffle sorting', () => {
    it('should produce deterministic order based on libraryLoadId', () => {
      const result1 = filter('shuffle-1', '', sampleLibrary, 'all', 'shuffle');
      const result2 = filter('shuffle-1', '', sampleLibrary, 'all', 'shuffle');
      expect(result1.map((i) => i.path)).toEqual(result2.map((i) => i.path));
    });

    it('should produce different order with different libraryLoadId', () => {
      const result1 = filter('shuffle-a', '', sampleLibrary, 'all', 'shuffle');
      const result2 = filter('shuffle-b', '', sampleLibrary, 'all', 'shuffle');
      // Different seeds should (very likely) produce different orders
      const order1 = result1.map((i) => i.path).join(',');
      const order2 = result2.map((i) => i.path).join(',');
      // This could theoretically fail if we're extremely unlucky
      expect(order1).not.toBe(order2);
    });

    it('should prioritize items without elo ratings in shuffle', () => {
      const mixedEloItems: Item[] = [
        createItem('/a.jpg', 1000, { elo: 1500 }),
        createItem('/b.jpg', 2000),
        createItem('/c.jpg', 3000, { elo: 1600 }),
        createItem('/d.jpg', 4000),
      ];
      const result = filter(
        'shuffle-elo',
        '',
        mixedEloItems,
        'all',
        'shuffle'
      );
      // Items without elo should come first
      const firstTwo = result.slice(0, 2);
      expect(firstTwo.every((item) => !item.elo)).toBe(true);
    });
  });

  describe('stream sorting', () => {
    it('should preserve original order when using stream sort', () => {
      const result = filter('stream-1', '', sampleLibrary, 'all', 'stream');
      expect(result.map((i) => i.path)).toEqual(sampleLibrary.map((i) => i.path));
    });
  });

  describe('memoization', () => {
    it('should return cached result for same cacheKey', () => {
      const lib = [...sampleLibrary];
      const result1 = filter('memo-test', '', lib, 'all', 'name');
      // Mutate the array (this shouldn't affect cached result)
      lib.push(createItem('/new.jpg', 9999));
      const result2 = filter('memo-test', '', lib, 'all', 'name');
      // Should return cached result, not include the new item
      expect(result1).toBe(result2);
      expect(result2).toHaveLength(5);
    });

    it('should invalidate cache when libraryLoadId changes', () => {
      const lib = [...sampleLibrary];
      const result1 = filter('memo-1', '', lib, 'all', 'name');
      const result2 = filter('memo-2', '', lib, 'all', 'name');
      // Different libraryLoadId should produce fresh result
      expect(result1).not.toBe(result2);
    });

    it('should invalidate cache when filter changes', () => {
      const result1 = filter('filter-cache', '', sampleLibrary, 'all', 'name');
      const result2 = filter('filter-cache', '', sampleLibrary, 'static', 'name');
      expect(result1.length).not.toBe(result2.length);
    });

    it('should invalidate cache when sortBy changes', () => {
      const result1 = filter('sort-cache', '', sampleLibrary, 'all', 'name');
      const result2 = filter('sort-cache', '', sampleLibrary, 'all', 'date');
      expect(result1[0].path).not.toBe(result2[0].path);
    });
  });

  describe('edge cases', () => {
    it('should handle empty library', () => {
      const result = filter('empty', '', [], 'all', 'name');
      expect(result).toEqual([]);
    });

    it('should handle single item', () => {
      const single = [createItem('/single.jpg', 1000)];
      const result = filter('single', '', single, 'all', 'name');
      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('/single.jpg');
    });

    it('should return library as-is when no filtering or sorting needed', () => {
      // When textFilter, filters, and sortBy are all falsy
      const result = filter('passthrough', '', sampleLibrary, '' as any, '' as any);
      expect(result).toBe(sampleLibrary);
    });
  });
});
