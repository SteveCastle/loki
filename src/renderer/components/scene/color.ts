// Colour maths for the paint colour picker: one colour, read and edited in
// several spaces. The picker keeps HSV as its state (hue survives grey and
// black, saturation survives black), so every other space converts to and from
// it through sRGB. OKLCH is perceptual: equal steps of L look equally lighter,
// and changing H keeps the lightness, which HSL does not. No React in here.

export type ColorSpace = 'hex' | 'rgb' | 'hsl' | 'hsv' | 'oklch';

/** Hue 0-360, saturation and value 0-1. */
export interface HSV {
  h: number;
  s: number;
  v: number;
}

export type RGB = [number, number, number]; // 0-255

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const wrapHue = (h: number) => ((h % 360) + 360) % 360;

/** '#rrggbb' (or '#rgb', with or without '#'); null when it is not one. */
export function parseHex(s: string): RGB | null {
  let t = s.trim().replace(/^#/, '');
  if (/^[0-9a-f]{3}$/i.test(t)) t = t.split('').map((c) => c + c).join('');
  if (!/^[0-9a-f]{6}$/i.test(t)) return null;
  return [parseInt(t.slice(0, 2), 16), parseInt(t.slice(2, 4), 16), parseInt(t.slice(4, 6), 16)];
}

export function toHex([r, g, b]: RGB): string {
  return `#${[r, g, b].map((v) => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')).join('')}`;
}

export function hsvToRgb({ h, s, v }: HSV): RGB {
  const f = (n: number) => {
    const k = (n + wrapHue(h) / 60) % 6;
    return v - v * s * Math.max(0, Math.min(k, 4 - k, 1));
  };
  return [f(5) * 255, f(3) * 255, f(1) * 255];
}

/** prev's hue (and saturation) is kept where the colour has none: grey has no hue, black no saturation either. */
export function rgbToHsv([r, g, b]: RGB, prev?: HSV): HSV {
  const R = r / 255;
  const G = g / 255;
  const B = b / 255;
  const max = Math.max(R, G, B);
  const d = max - Math.min(R, G, B);
  let h = prev?.h ?? 0;
  if (d > 1e-9) {
    if (max === R) h = 60 * (((G - B) / d) % 6);
    else if (max === G) h = 60 * ((B - R) / d + 2);
    else h = 60 * ((R - G) / d + 4);
  }
  const s = max > 1e-9 ? d / max : prev?.s ?? 0;
  return { h: wrapHue(h), s, v: max };
}

/** HSL as [h 0-360, s 0-100, l 0-100]. */
export function rgbToHsl(rgb: RGB, hue = 0): [number, number, number] {
  const [r, g, b] = rgb.map((v) => v / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  const s = d < 1e-9 ? 0 : d / (1 - Math.abs(2 * l - 1));
  return [d < 1e-9 ? hue : rgbToHsv(rgb).h, s * 100, l * 100];
}

export function hslToRgb([h, s, l]: [number, number, number]): RGB {
  const S = clamp(s, 0, 100) / 100;
  const L = clamp(l, 0, 100) / 100;
  const a = S * Math.min(L, 1 - L);
  const f = (n: number) => {
    const k = (n + wrapHue(h) / 30) % 12;
    return L - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0) * 255, f(8) * 255, f(4) * 255];
}

const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fromLin = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);

/** OKLCH as [L 0-1, C 0-~0.37, h 0-360] (Björn Ottosson's OKLab, polar). */
export function rgbToOklch(rgb: RGB, hue = 0): [number, number, number] {
  const [r, g, b] = rgb.map((v) => toLin(v / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  const C = Math.hypot(A, B);
  return [L, C, C < 1e-4 ? hue : wrapHue((Math.atan2(B, A) * 180) / Math.PI)];
}

/**
 * OKLCH back to sRGB. A colour outside sRGB (high chroma at that lightness and hue) is brought in by lowering its
 * chroma until it fits, keeping lightness and hue: what you set is what you see, as close as the screen can show.
 */
export function oklchToRgb([L, C, h]: [number, number, number]): RGB {
  const conv = (c: number): [number, number, number] => {
    const A = c * Math.cos((h * Math.PI) / 180);
    const B = c * Math.sin((h * Math.PI) / 180);
    const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
    const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
    const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
    return [
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ];
  };
  const inGamut = (v: number[]) => v.every((x) => x >= -1e-4 && x <= 1 + 1e-4);
  let lin = conv(C);
  if (!inGamut(lin)) {
    let lo = 0;
    let hi = C;
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2;
      if (inGamut(conv(mid))) lo = mid;
      else hi = mid;
    }
    lin = conv(lo);
  }
  return lin.map((x) => clamp(fromLin(clamp(x, 0, 1)), 0, 1) * 255) as RGB;
}

/** The fields a space is edited with: label, range, step, and how many decimals to show. */
export const SPACE_FIELDS: Record<Exclude<ColorSpace, 'hex'>, Array<{ label: string; min: number; max: number; step: number; digits: number }>> = {
  rgb: [
    { label: 'R', min: 0, max: 255, step: 1, digits: 0 },
    { label: 'G', min: 0, max: 255, step: 1, digits: 0 },
    { label: 'B', min: 0, max: 255, step: 1, digits: 0 },
  ],
  hsl: [
    { label: 'H', min: 0, max: 360, step: 1, digits: 0 },
    { label: 'S', min: 0, max: 100, step: 1, digits: 0 },
    { label: 'L', min: 0, max: 100, step: 1, digits: 0 },
  ],
  hsv: [
    { label: 'H', min: 0, max: 360, step: 1, digits: 0 },
    { label: 'S', min: 0, max: 100, step: 1, digits: 0 },
    { label: 'V', min: 0, max: 100, step: 1, digits: 0 },
  ],
  oklch: [
    { label: 'L', min: 0, max: 100, step: 0.5, digits: 1 },
    { label: 'C', min: 0, max: 0.37, step: 0.005, digits: 3 },
    { label: 'H', min: 0, max: 360, step: 1, digits: 0 },
  ],
};

/** hsv as the three numbers of space (OKLCH's L in percent). */
export function readSpace(space: Exclude<ColorSpace, 'hex'>, hsv: HSV): [number, number, number] {
  const rgb = hsvToRgb(hsv);
  switch (space) {
    case 'rgb':
      return rgb.map((v) => Math.round(v)) as [number, number, number];
    case 'hsl':
      return rgbToHsl(rgb, hsv.h);
    case 'hsv':
      return [hsv.h, hsv.s * 100, hsv.v * 100];
    default: {
      const [L, C, h] = rgbToOklch(rgb, hsv.h);
      return [L * 100, C, h];
    }
  }
}

/** The colour three numbers of space name, as HSV (prev keeps the hue of greys). */
export function writeSpace(space: Exclude<ColorSpace, 'hex'>, v: [number, number, number], prev: HSV): HSV {
  switch (space) {
    case 'rgb':
      return rgbToHsv(v.map((x) => clamp(x, 0, 255)) as RGB, prev);
    case 'hsl': {
      const out = rgbToHsv(hslToRgb(v), prev);
      return v[1] <= 0 || v[2] <= 0 || v[2] >= 100 ? { ...out, h: wrapHue(v[0]) } : out;
    }
    case 'hsv':
      return { h: wrapHue(v[0]), s: clamp(v[1], 0, 100) / 100, v: clamp(v[2], 0, 100) / 100 };
    default: {
      const out = rgbToHsv(oklchToRgb([clamp(v[0], 0, 100) / 100, Math.max(0, v[1]), wrapHue(v[2])]), prev);
      return v[1] < 1e-4 ? { ...out, h: prev.h } : out;
    }
  }
}
