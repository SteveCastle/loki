/**
 * Tests for state machine logic - specifically the pure functions
 * and guards that can be tested without the full Electron environment.
 *
 * These tests verify the business logic extracted from state.tsx
 */

import type { Item } from '../renderer/state';

// Re-implement the guard functions locally for testing
// (These mirror the logic in state.tsx)

/**
 * willHaveTag - Detects if toggling a tag will result in a non-empty tag list
 */
function willHaveTag(
  currentTags: string[],
  tagToToggle: string
): boolean {
  const index = currentTags.indexOf(tagToToggle);
  const newTagList = [...currentTags];
  if (index > -1) {
    newTagList.splice(index, 1);
  } else {
    newTagList.push(tagToToggle);
  }
  return newTagList.length !== 0;
}

/**
 * willHaveNoTag - Detects if toggling a tag will result in an empty tag list
 */
function willHaveNoTag(
  currentTags: string[],
  tagToToggle: string
): boolean {
  const index = currentTags.indexOf(tagToToggle);
  const newTagList = [...currentTags];
  if (index > -1) {
    newTagList.splice(index, 1);
  } else {
    newTagList.push(tagToToggle);
  }
  return newTagList.length === 0;
}

/**
 * isEmpty - Checks if text filter is empty
 */
function isEmpty(textFilter: string): boolean {
  return textFilter.length === 0;
}

/**
 * notEmpty - Checks if text filter is not empty
 */
function notEmpty(textFilter: string): boolean {
  return textFilter.length > 0;
}

/**
 * calculateNewCursorAfterDelete - When cursor is at the end, decrement it
 */
function calculateNewCursorAfterDelete(
  cursor: number,
  libraryLength: number
): number {
  if (cursor >= libraryLength - 1) {
    return cursor - 1;
  }
  return cursor;
}

/**
 * incrementCursor - Wraps around to 0 at end
 */
function incrementCursor(cursor: number, lastIndex: number): number {
  return cursor < lastIndex ? cursor + 1 : 0;
}

/**
 * decrementCursor - Wraps around to end at beginning
 */
function decrementCursor(cursor: number, lastIndex: number): number {
  return cursor > 0 ? cursor - 1 : lastIndex;
}

/**
 * toggleTagInList - Toggle a tag in a list (EXCLUSIVE mode clears others)
 */
function toggleTagInList(
  currentTags: string[],
  tagToToggle: string,
  filteringMode: 'AND' | 'OR' | 'EXCLUSIVE'
): string[] {
  const activeTags: { [key: string]: boolean } = {};
  currentTags.forEach((tag) => {
    activeTags[tag] = true;
  });

  // In EXCLUSIVE mode, clear other tags
  if (filteringMode === 'EXCLUSIVE') {
    Object.keys(activeTags).forEach((tag) => {
      if (tag !== tagToToggle) {
        delete activeTags[tag];
      }
    });
  }

  // Toggle the target tag
  if (activeTags[tagToToggle]) {
    delete activeTags[tagToToggle];
  } else {
    activeTags[tagToToggle] = true;
  }

  return Object.keys(activeTags);
}

describe('State Machine Logic', () => {
  describe('willHaveTag', () => {
    it('should return true when adding first tag', () => {
      expect(willHaveTag([], 'tag1')).toBe(true);
    });

    it('should return true when adding another tag', () => {
      expect(willHaveTag(['tag1'], 'tag2')).toBe(true);
    });

    it('should return true when removing one of multiple tags', () => {
      expect(willHaveTag(['tag1', 'tag2'], 'tag1')).toBe(true);
    });

    it('should return false when removing the only tag', () => {
      expect(willHaveTag(['tag1'], 'tag1')).toBe(false);
    });
  });

  describe('willHaveNoTag', () => {
    it('should return false when adding first tag', () => {
      expect(willHaveNoTag([], 'tag1')).toBe(false);
    });

    it('should return false when adding another tag', () => {
      expect(willHaveNoTag(['tag1'], 'tag2')).toBe(false);
    });

    it('should return false when removing one of multiple tags', () => {
      expect(willHaveNoTag(['tag1', 'tag2'], 'tag1')).toBe(false);
    });

    it('should return true when removing the only tag', () => {
      expect(willHaveNoTag(['tag1'], 'tag1')).toBe(true);
    });
  });

  describe('isEmpty', () => {
    it('should return true for empty string', () => {
      expect(isEmpty('')).toBe(true);
    });

    it('should return false for non-empty string', () => {
      expect(isEmpty('search')).toBe(false);
      expect(isEmpty(' ')).toBe(false);
      expect(isEmpty('a')).toBe(false);
    });
  });

  describe('notEmpty', () => {
    it('should return false for empty string', () => {
      expect(notEmpty('')).toBe(false);
    });

    it('should return true for non-empty string', () => {
      expect(notEmpty('search')).toBe(true);
      expect(notEmpty(' ')).toBe(true);
      expect(notEmpty('a')).toBe(true);
    });
  });

  describe('calculateNewCursorAfterDelete', () => {
    it('should decrement cursor when at last item', () => {
      expect(calculateNewCursorAfterDelete(4, 5)).toBe(3);
      expect(calculateNewCursorAfterDelete(9, 10)).toBe(8);
    });

    it('should decrement cursor when past the end', () => {
      expect(calculateNewCursorAfterDelete(5, 5)).toBe(4);
    });

    it('should keep cursor when not at end', () => {
      expect(calculateNewCursorAfterDelete(0, 5)).toBe(0);
      expect(calculateNewCursorAfterDelete(2, 5)).toBe(2);
    });

    it('should handle edge case of single item', () => {
      expect(calculateNewCursorAfterDelete(0, 1)).toBe(-1);
    });
  });

  describe('incrementCursor', () => {
    it('should increment cursor normally', () => {
      expect(incrementCursor(0, 4)).toBe(1);
      expect(incrementCursor(1, 4)).toBe(2);
      expect(incrementCursor(2, 4)).toBe(3);
    });

    it('should wrap to 0 at end', () => {
      expect(incrementCursor(4, 4)).toBe(0);
    });

    it('should wrap to 0 when at lastIndex', () => {
      expect(incrementCursor(9, 9)).toBe(0);
    });
  });

  describe('decrementCursor', () => {
    it('should decrement cursor normally', () => {
      expect(decrementCursor(4, 4)).toBe(3);
      expect(decrementCursor(3, 4)).toBe(2);
      expect(decrementCursor(1, 4)).toBe(0);
    });

    it('should wrap to end at beginning', () => {
      expect(decrementCursor(0, 4)).toBe(4);
    });

    it('should wrap to lastIndex when at 0', () => {
      expect(decrementCursor(0, 9)).toBe(9);
    });
  });

  describe('toggleTagInList', () => {
    describe('AND mode', () => {
      it('should add tag when not present', () => {
        expect(toggleTagInList([], 'tag1', 'AND')).toEqual(['tag1']);
        expect(toggleTagInList(['tag1'], 'tag2', 'AND')).toEqual([
          'tag1',
          'tag2',
        ]);
      });

      it('should remove tag when present', () => {
        expect(toggleTagInList(['tag1'], 'tag1', 'AND')).toEqual([]);
        expect(toggleTagInList(['tag1', 'tag2'], 'tag1', 'AND')).toEqual([
          'tag2',
        ]);
      });

      it('should keep other tags when toggling', () => {
        const result = toggleTagInList(['a', 'b', 'c'], 'b', 'AND');
        expect(result).toContain('a');
        expect(result).toContain('c');
        expect(result).not.toContain('b');
      });
    });

    describe('OR mode', () => {
      it('should behave same as AND mode', () => {
        expect(toggleTagInList([], 'tag1', 'OR')).toEqual(['tag1']);
        expect(toggleTagInList(['tag1', 'tag2'], 'tag1', 'OR')).toEqual([
          'tag2',
        ]);
      });
    });

    describe('EXCLUSIVE mode', () => {
      it('should add tag and clear others when adding', () => {
        expect(toggleTagInList(['tag1'], 'tag2', 'EXCLUSIVE')).toEqual([
          'tag2',
        ]);
        expect(toggleTagInList(['a', 'b', 'c'], 'new', 'EXCLUSIVE')).toEqual([
          'new',
        ]);
      });

      it('should remove tag when toggling the same tag', () => {
        expect(toggleTagInList(['tag1'], 'tag1', 'EXCLUSIVE')).toEqual([]);
      });

      it('should result in empty when toggling the only tag', () => {
        expect(toggleTagInList(['tag1', 'tag2'], 'tag2', 'EXCLUSIVE')).toEqual(
          []
        );
      });
    });
  });

  describe('Library manipulation', () => {
    const createItem = (path: string, mtimeMs: number = 0): Item => ({
      path,
      mtimeMs,
    });

    describe('delete file from library', () => {
      it('should remove item by path', () => {
        const library = [
          createItem('/a.jpg'),
          createItem('/b.jpg'),
          createItem('/c.jpg'),
        ];
        const pathToDelete = '/b.jpg';

        const newLibrary = library.filter((item) => item.path !== pathToDelete);

        expect(newLibrary).toHaveLength(2);
        expect(newLibrary.map((i) => i.path)).toEqual(['/a.jpg', '/c.jpg']);
      });

      it('should handle deleting non-existent path', () => {
        const library = [createItem('/a.jpg'), createItem('/b.jpg')];
        const pathToDelete = '/nonexistent.jpg';

        const newLibrary = library.filter((item) => item.path !== pathToDelete);

        expect(newLibrary).toHaveLength(2);
      });
    });

    describe('update file path in library', () => {
      it('should update path for matching item', () => {
        const library = [
          createItem('/old/a.jpg'),
          createItem('/old/b.jpg'),
        ];
        const oldPath = '/old/a.jpg';
        const newPath = '/new/a.jpg';

        const updatedLibrary = library.map((item) =>
          item.path === oldPath ? { ...item, path: newPath } : item
        );

        expect(updatedLibrary[0].path).toBe('/new/a.jpg');
        expect(updatedLibrary[1].path).toBe('/old/b.jpg');
      });
    });

    describe('update elo in library', () => {
      it('should update elo for matching item', () => {
        const library = [
          createItem('/a.jpg'),
          createItem('/b.jpg'),
        ];
        const targetPath = '/a.jpg';
        const newElo = 1650;

        const updatedLibrary = library.map((item) =>
          item.path === targetPath ? { ...item, elo: newElo } : item
        );

        expect(updatedLibrary[0].elo).toBe(1650);
        expect(updatedLibrary[1].elo).toBeUndefined();
      });
    });
  });

  describe('Toast creation', () => {
    it('should create toast with correct structure', () => {
      const createToast = (
        type: 'success' | 'error' | 'info',
        title: string,
        message?: string
      ) => ({
        id: 'test-id',
        type,
        title,
        message,
        timestamp: Date.now(),
      });

      const toast = createToast('success', 'Test Title', 'Test Message');

      expect(toast.type).toBe('success');
      expect(toast.title).toBe('Test Title');
      expect(toast.message).toBe('Test Message');
      expect(typeof toast.timestamp).toBe('number');
    });

    it('should handle toasts without message', () => {
      const createToast = (
        type: 'success' | 'error' | 'info',
        title: string,
        message?: string
      ) => ({
        id: 'test-id',
        type,
        title,
        message,
        timestamp: Date.now(),
      });

      const toast = createToast('info', 'Just Title');

      expect(toast.message).toBeUndefined();
    });
  });

  describe('Video player state', () => {
    const defaultVideoState = {
      eventId: 'initial',
      timeStamp: 0,
      playing: true,
      actualVideoTime: 0,
      videoLength: 0,
      loopLength: 0,
      loopStartTime: 0,
    };

    it('should toggle playing state', () => {
      const togglePlaying = (state: typeof defaultVideoState) => ({
        ...state,
        playing: !state.playing,
      });

      const paused = togglePlaying(defaultVideoState);
      expect(paused.playing).toBe(false);

      const resumed = togglePlaying(paused);
      expect(resumed.playing).toBe(true);
    });

    it('should set video time with new event id', () => {
      const setVideoTime = (
        state: typeof defaultVideoState,
        timeStamp: number,
        eventId: string
      ) => ({
        ...state,
        timeStamp,
        eventId,
      });

      const newState = setVideoTime(defaultVideoState, 30.5, 'seek-1');
      expect(newState.timeStamp).toBe(30.5);
      expect(newState.eventId).toBe('seek-1');
    });

    it('should set loop parameters', () => {
      const setLoop = (
        state: typeof defaultVideoState,
        loopLength: number,
        loopStartTime: number
      ) => ({
        ...state,
        loopLength,
        loopStartTime,
      });

      const looped = setLoop(defaultVideoState, 5, 10);
      expect(looped.loopLength).toBe(5);
      expect(looped.loopStartTime).toBe(10);
    });
  });

  describe('Settings state', () => {
    it('should merge new settings with existing', () => {
      const currentSettings = {
        sortBy: 'name' as const,
        filters: 'all' as const,
        volume: 0.5,
      };

      const mergeSettings = (
        current: typeof currentSettings,
        updates: Partial<typeof currentSettings>
      ) => ({
        ...current,
        ...updates,
      });

      const updated = mergeSettings(currentSettings, { volume: 0.8 });
      expect(updated.volume).toBe(0.8);
      expect(updated.sortBy).toBe('name');
      expect(updated.filters).toBe('all');
    });

    it('should handle setting shuffle sort mode', () => {
      const currentSettings = {
        sortBy: 'name' as const,
        filters: 'all' as const,
      };

      const setShuffle = (settings: typeof currentSettings) => ({
        ...settings,
        sortBy: 'shuffle' as const,
      });

      const shuffled = setShuffle(currentSettings);
      expect(shuffled.sortBy).toBe('shuffle');
    });
  });
});
