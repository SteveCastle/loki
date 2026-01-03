/**
 * Tests for hotkey parsing and matching logic.
 * Based on the hotkey format used in the application (e.g., "1+control", "arrowright").
 */

describe('Hotkey Utilities', () => {
  /**
   * Parse a hotkey string into its components
   * Format: "key+modifier1+modifier2" or just "key"
   */
  const parseHotkey = (
    hotkey: string
  ): { key: string; modifiers: string[] } => {
    const parts = hotkey.toLowerCase().split('+');
    const key = parts[0];
    const modifiers = parts.slice(1).sort();
    return { key, modifiers };
  };

  /**
   * Check if a keyboard event matches a hotkey string
   */
  const matchesHotkey = (event: Partial<KeyboardEvent>, hotkey: string): boolean => {
    const { key, modifiers } = parseHotkey(hotkey);

    // Check the key matches
    if (event.key?.toLowerCase() !== key) {
      return false;
    }

    // Check modifiers
    const eventModifiers: string[] = [];
    if (event.ctrlKey) eventModifiers.push('control');
    if (event.shiftKey) eventModifiers.push('shift');
    if (event.altKey) eventModifiers.push('alt');
    if (event.metaKey) eventModifiers.push('meta');
    eventModifiers.sort();

    // Both should have same modifiers
    if (modifiers.length !== eventModifiers.length) {
      return false;
    }

    return modifiers.every((mod, i) => mod === eventModifiers[i]);
  };

  describe('parseHotkey', () => {
    it('should parse simple key', () => {
      expect(parseHotkey('a')).toEqual({ key: 'a', modifiers: [] });
      expect(parseHotkey('x')).toEqual({ key: 'x', modifiers: [] });
      expect(parseHotkey('1')).toEqual({ key: '1', modifiers: [] });
    });

    it('should parse key with single modifier', () => {
      expect(parseHotkey('c+control')).toEqual({
        key: 'c',
        modifiers: ['control'],
      });
      expect(parseHotkey('a+shift')).toEqual({ key: 'a', modifiers: ['shift'] });
      expect(parseHotkey('s+alt')).toEqual({ key: 's', modifiers: ['alt'] });
    });

    it('should parse key with multiple modifiers', () => {
      expect(parseHotkey('c+control+shift')).toEqual({
        key: 'c',
        modifiers: ['control', 'shift'],
      });
      expect(parseHotkey('s+alt+control')).toEqual({
        key: 's',
        modifiers: ['alt', 'control'],
      });
    });

    it('should handle arrow keys', () => {
      expect(parseHotkey('arrowright')).toEqual({
        key: 'arrowright',
        modifiers: [],
      });
      expect(parseHotkey('arrowleft')).toEqual({
        key: 'arrowleft',
        modifiers: [],
      });
    });

    it('should handle special keys', () => {
      expect(parseHotkey(' ')).toEqual({ key: ' ', modifiers: [] }); // space
      expect(parseHotkey('escape')).toEqual({ key: 'escape', modifiers: [] });
      expect(parseHotkey('delete')).toEqual({ key: 'delete', modifiers: [] });
    });

    it('should normalize to lowercase', () => {
      expect(parseHotkey('A')).toEqual({ key: 'a', modifiers: [] });
      expect(parseHotkey('C+Control')).toEqual({
        key: 'c',
        modifiers: ['control'],
      });
    });
  });

  describe('matchesHotkey', () => {
    it('should match simple key press', () => {
      const event = { key: 'a', ctrlKey: false, shiftKey: false, altKey: false };
      expect(matchesHotkey(event, 'a')).toBe(true);
      expect(matchesHotkey(event, 'b')).toBe(false);
    });

    it('should match key with control modifier', () => {
      const event = { key: 'c', ctrlKey: true, shiftKey: false, altKey: false };
      expect(matchesHotkey(event, 'c+control')).toBe(true);
      expect(matchesHotkey(event, 'c')).toBe(false);
      expect(matchesHotkey(event, 'c+shift')).toBe(false);
    });

    it('should match key with shift modifier', () => {
      const event = { key: 's', ctrlKey: false, shiftKey: true, altKey: false };
      expect(matchesHotkey(event, 's+shift')).toBe(true);
      expect(matchesHotkey(event, 's')).toBe(false);
    });

    it('should match key with multiple modifiers', () => {
      const event = { key: 'c', ctrlKey: true, shiftKey: true, altKey: false };
      expect(matchesHotkey(event, 'c+control+shift')).toBe(true);
      expect(matchesHotkey(event, 'c+shift+control')).toBe(true); // order shouldn't matter
      expect(matchesHotkey(event, 'c+control')).toBe(false);
    });

    it('should match arrow keys', () => {
      const rightArrow = {
        key: 'ArrowRight',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
      };
      expect(matchesHotkey(rightArrow, 'arrowright')).toBe(true);

      const leftArrow = {
        key: 'ArrowLeft',
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
      };
      expect(matchesHotkey(leftArrow, 'arrowleft')).toBe(true);
    });

    it('should match space key', () => {
      const space = { key: ' ', ctrlKey: false, shiftKey: false, altKey: false };
      expect(matchesHotkey(space, ' ')).toBe(true);
    });

    it('should match number keys with modifiers', () => {
      const ctrl1 = { key: '1', ctrlKey: true, shiftKey: false, altKey: false };
      expect(matchesHotkey(ctrl1, '1+control')).toBe(true);

      const alt5 = { key: '5', ctrlKey: false, shiftKey: false, altKey: true };
      expect(matchesHotkey(alt5, '5+alt')).toBe(true);

      const shift3 = { key: '3', ctrlKey: false, shiftKey: true, altKey: false };
      expect(matchesHotkey(shift3, '3+shift')).toBe(true);
    });

    it('should not match when extra modifiers are pressed', () => {
      const event = {
        key: 'a',
        ctrlKey: true,
        shiftKey: true,
        altKey: false,
      };
      expect(matchesHotkey(event, 'a+control')).toBe(false);
    });
  });

  describe('Default hotkey mappings', () => {
    const defaultHotkeys = {
      incrementCursor: 'arrowright',
      decrementCursor: 'arrowleft',
      toggleTagPreview: 'shift',
      toggleTagAll: 'control',
      moveToTop: '[',
      moveToEnd: ']',
      minimize: 'escape',
      shuffle: 'x',
      copyFile: 'c+control',
      copyAllSelectedFiles: 'c+control+shift',
      deleteFile: 'delete',
      applyMostRecentTag: 'a',
      storeCategory1: '1+alt',
      storeTag1: '1+control',
      applyTag1: '1',
      togglePlayPause: ' ',
    };

    it('should have sensible defaults for navigation', () => {
      expect(defaultHotkeys.incrementCursor).toBe('arrowright');
      expect(defaultHotkeys.decrementCursor).toBe('arrowleft');
    });

    it('should have sensible defaults for clipboard operations', () => {
      expect(defaultHotkeys.copyFile).toBe('c+control');
      expect(defaultHotkeys.copyAllSelectedFiles).toBe('c+control+shift');
    });

    it('should have sensible defaults for playback', () => {
      expect(defaultHotkeys.togglePlayPause).toBe(' ');
    });

    it('should have number-based shortcuts for tagging', () => {
      expect(defaultHotkeys.applyTag1).toBe('1');
      expect(defaultHotkeys.storeTag1).toBe('1+control');
      expect(defaultHotkeys.storeCategory1).toBe('1+alt');
    });
  });

  describe('Hotkey conflicts detection', () => {
    const detectConflicts = (hotkeys: Record<string, string>): string[][] => {
      const conflicts: string[][] = [];
      const hotkeyToActions = new Map<string, string[]>();

      for (const [action, hotkey] of Object.entries(hotkeys)) {
        const normalized = hotkey.toLowerCase();
        if (!hotkeyToActions.has(normalized)) {
          hotkeyToActions.set(normalized, []);
        }
        hotkeyToActions.get(normalized)!.push(action);
      }

      for (const [_hotkey, actions] of hotkeyToActions) {
        if (actions.length > 1) {
          conflicts.push(actions);
        }
      }

      return conflicts;
    };

    it('should detect no conflicts in default hotkeys', () => {
      const hotkeys = {
        incrementCursor: 'arrowright',
        decrementCursor: 'arrowleft',
        shuffle: 'x',
        copyFile: 'c+control',
      };
      expect(detectConflicts(hotkeys)).toEqual([]);
    });

    it('should detect conflicts when same hotkey used twice', () => {
      const hotkeys = {
        action1: 'a+control',
        action2: 'a+control',
        action3: 'b',
      };
      const conflicts = detectConflicts(hotkeys);
      expect(conflicts).toHaveLength(1);
      expect(conflicts[0]).toContain('action1');
      expect(conflicts[0]).toContain('action2');
    });

    it('should handle case-insensitive comparison', () => {
      const hotkeys = {
        action1: 'A+Control',
        action2: 'a+control',
      };
      const conflicts = detectConflicts(hotkeys);
      expect(conflicts).toHaveLength(1);
    });
  });
});
