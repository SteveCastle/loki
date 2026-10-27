import { hslToRgb, hsvToRgb, oklchToRgb, parseHex, readSpace, rgbToHsl, rgbToHsv, rgbToOklch, toHex, writeSpace, type RGB } from '../renderer/components/scene/color';

const close = (a: number[], b: number[], tol = 1) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

describe('scene colour picker maths', () => {
  test('hex parses long and short forms and round-trips', () => {
    expect(parseHex('#e2533b')).toEqual([226, 83, 59]);
    expect(parseHex('fff')).toEqual([255, 255, 255]);
    expect(parseHex('#12345')).toBeNull();
    expect(toHex([226, 83, 59])).toBe('#e2533b');
  });

  test('every space round-trips a range of colours through sRGB', () => {
    const samples: RGB[] = [[226, 83, 59], [12, 200, 150], [40, 40, 40], [255, 255, 255], [0, 0, 0], [59, 130, 246], [245, 208, 181]];
    for (const c of samples) {
      expect(close(hsvToRgb(rgbToHsv(c)), c)).toBe(true);
      expect(close(hslToRgb(rgbToHsl(c)), c)).toBe(true);
      expect(close(oklchToRgb(rgbToOklch(c)), c)).toBe(true);
      for (const space of ['rgb', 'hsl', 'hsv', 'oklch'] as const) {
        const hsv = rgbToHsv(c);
        expect(close(hsvToRgb(writeSpace(space, readSpace(space, hsv), hsv)), c)).toBe(true);
      }
    }
  });

  test('grey keeps the hue it had, so dragging to grey and back does not lose it', () => {
    const red = rgbToHsv([226, 83, 59]);
    const grey = rgbToHsv([128, 128, 128], red);
    expect(grey.h).toBeCloseTo(red.h);
    expect(writeSpace('oklch', [50, 0, 0], red).h).toBeCloseTo(red.h);
  });

  test('OKLCH out of sRGB keeps its lightness and hue and loses chroma', () => {
    const rgb = oklchToRgb([0.6, 0.37, 140]);
    expect(rgb.every((v) => v >= 0 && v <= 255)).toBe(true);
    const [L, , h] = rgbToOklch(rgb);
    expect(L).toBeCloseTo(0.6, 2);
    expect(Math.abs(h - 140)).toBeLessThan(2);
  });
});
