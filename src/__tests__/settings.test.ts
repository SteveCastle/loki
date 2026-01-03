import {
  getNextFilterMode,
  clampVolume,
  SCALE_MODES,
  SORT_BY,
  FILTERS,
  SETTINGS,
  type FilterModeOption,
} from '../settings';

describe('settings', () => {
  describe('getNextFilterMode', () => {
    it('should cycle from AND to OR', () => {
      expect(getNextFilterMode('AND')).toBe('OR');
    });

    it('should cycle from OR to EXCLUSIVE', () => {
      expect(getNextFilterMode('OR')).toBe('EXCLUSIVE');
    });

    it('should cycle from EXCLUSIVE to AND', () => {
      expect(getNextFilterMode('EXCLUSIVE')).toBe('AND');
    });

    it('should complete full cycle', () => {
      let mode: FilterModeOption = 'AND';
      mode = getNextFilterMode(mode);
      expect(mode).toBe('OR');
      mode = getNextFilterMode(mode);
      expect(mode).toBe('EXCLUSIVE');
      mode = getNextFilterMode(mode);
      expect(mode).toBe('AND');
    });

    it('should throw for invalid mode', () => {
      expect(() => getNextFilterMode('INVALID' as FilterModeOption)).toThrow(
        'Invalid filter mode: INVALID'
      );
    });
  });

  describe('clampVolume', () => {
    it('should return 0 for negative values', () => {
      expect(clampVolume(-1)).toBe(0);
      expect(clampVolume(-0.5)).toBe(0);
      expect(clampVolume(-100)).toBe(0);
    });

    it('should return 1 for values greater than 1', () => {
      expect(clampVolume(1.5)).toBe(1);
      expect(clampVolume(2)).toBe(1);
      expect(clampVolume(100)).toBe(1);
    });

    it('should return the value for values between 0 and 1', () => {
      expect(clampVolume(0)).toBe(0);
      expect(clampVolume(0.5)).toBe(0.5);
      expect(clampVolume(1)).toBe(1);
      expect(clampVolume(0.75)).toBe(0.75);
    });

    it('should handle edge cases', () => {
      expect(clampVolume(0.0)).toBe(0);
      expect(clampVolume(1.0)).toBe(1);
      expect(clampVolume(0.999999)).toBeCloseTo(0.999999);
    });
  });

  describe('SCALE_MODES', () => {
    it('should have required properties', () => {
      expect(SCALE_MODES.title).toBe('Scale Mode');
      expect(SCALE_MODES.reload).toBe(false);
      expect(SCALE_MODES.display).toBe('image');
    });

    it('should have cover option', () => {
      expect(SCALE_MODES.options.cover.label).toBe('Cover');
      expect(SCALE_MODES.options.cover.value).toBe('cover');
    });

    it('should have fit option', () => {
      expect(SCALE_MODES.options.fit.label).toBe('Fit');
      expect(SCALE_MODES.options.fit.value).toBe('fit');
    });

    it('should have overscan option with increment', () => {
      expect(SCALE_MODES.options.overscan.label).toBe('Zoom');
      expect(SCALE_MODES.options.overscan.value).toBe(140);
      expect(SCALE_MODES.options.overscan.increment).toBe(10);
    });

    it('should have actual option', () => {
      expect(SCALE_MODES.options.actual.label).toBe('Actual');
      expect(SCALE_MODES.options.actual.value).toBe('actual');
    });
  });

  describe('SORT_BY', () => {
    it('should have required properties', () => {
      expect(SORT_BY.title).toBe('Sort By');
      expect(SORT_BY.resetCursor).toBe(true);
    });

    it('should have all sort options', () => {
      expect(SORT_BY.options.name.value).toBe('name');
      expect(SORT_BY.options.date.value).toBe('date');
      expect(SORT_BY.options.weight.value).toBe('weight');
      expect(SORT_BY.options.elo.value).toBe('elo');
      expect(SORT_BY.options.shuffle.value).toBe('shuffle');
    });

    it('should have human-readable labels', () => {
      expect(SORT_BY.options.name.label).toBe('Name');
      expect(SORT_BY.options.date.label).toBe('Date');
      expect(SORT_BY.options.weight.label).toBe('Custom');
      expect(SORT_BY.options.elo.label).toBe('Elo');
      expect(SORT_BY.options.shuffle.label).toBe('Shuffle');
    });
  });

  describe('FILTERS', () => {
    it('should have required properties', () => {
      expect(FILTERS.title).toBe('Media Types');
      expect(FILTERS.resetCursor).toBe(true);
    });

    it('should have all filter options', () => {
      expect(FILTERS.options.all.value).toBe('all');
      expect(FILTERS.options.static.value).toBe('static');
      expect(FILTERS.options.video.value).toBe('video');
      expect(FILTERS.options.audio.value).toBe('audio');
    });

    it('should have human-readable labels', () => {
      expect(FILTERS.options.all.label).toBe('All');
      expect(FILTERS.options.static.label).toBe('Still');
      expect(FILTERS.options.video.label).toBe('Motion');
      expect(FILTERS.options.audio.label).toBe('Audio');
    });
  });

  describe('SETTINGS object', () => {
    it('should contain all expected setting categories', () => {
      expect(SETTINGS).toHaveProperty('scaleMode');
      expect(SETTINGS).toHaveProperty('sortBy');
      expect(SETTINGS).toHaveProperty('filters');
      expect(SETTINGS).toHaveProperty('playSound');
      expect(SETTINGS).toHaveProperty('volume');
      expect(SETTINGS).toHaveProperty('comicMode');
      expect(SETTINGS).toHaveProperty('showTagCount');
      expect(SETTINGS).toHaveProperty('battleMode');
      expect(SETTINGS).toHaveProperty('libraryLayout');
      expect(SETTINGS).toHaveProperty('followTranscript');
      expect(SETTINGS).toHaveProperty('showTags');
      expect(SETTINGS).toHaveProperty('showFileInfo');
      expect(SETTINGS).toHaveProperty('showControls');
      expect(SETTINGS).toHaveProperty('recursive');
      expect(SETTINGS).toHaveProperty('controlMode');
      expect(SETTINGS).toHaveProperty('autoPlay');
      expect(SETTINGS).toHaveProperty('autoPlayTime');
      expect(SETTINGS).toHaveProperty('autoPlayVideoLoops');
      expect(SETTINGS).toHaveProperty('alwaysOnTop');
      expect(SETTINGS).toHaveProperty('layoutMode');
    });

    it('should have consistent structure for all settings', () => {
      Object.values(SETTINGS).forEach((setting) => {
        expect(setting).toHaveProperty('title');
        expect(setting).toHaveProperty('reload');
        expect(setting).toHaveProperty('display');
        expect(setting).toHaveProperty('options');
        expect(typeof setting.title).toBe('string');
        expect(typeof setting.reload).toBe('boolean');
        expect(typeof setting.options).toBe('object');
      });
    });
  });
});
