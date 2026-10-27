import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { hsvToRgb, parseHex, readSpace, rgbToHsv, SPACE_FIELDS, toHex, writeSpace, type ColorSpace, type HSV } from './color';
import './color-picker.css';

// The paint colour picker: a saturation/value field and hue strip for picking
// by eye, an opacity strip, and the exact numbers in any of HEX, RGB, HSL, HSV
// or OKLCH (perceptual: steps of L look even, and H turns without changing
// how light it is). It opens beside the tool rail's colour well; every change
// applies live, Escape or a click outside closes it.

const SPACE_KEY = 'loki.scene.colorSpace.v1';
const SPACES: Array<{ id: ColorSpace; label: string; title: string }> = [
  { id: 'hex', label: 'HEX', title: 'Hex code' },
  { id: 'rgb', label: 'RGB', title: 'Red, green, blue (0-255)' },
  { id: 'hsl', label: 'HSL', title: 'Hue, saturation, lightness' },
  { id: 'hsv', label: 'HSV', title: 'Hue, saturation, value (the field and strip above)' },
  { id: 'oklch', label: 'OKLCH', title: 'Perceptual lightness, chroma, hue: even steps, and hue turns keep the lightness' },
];

function loadSpace(): ColorSpace {
  try {
    const v = window.localStorage.getItem(SPACE_KEY);
    return SPACES.some((s) => s.id === v) ? (v as ColorSpace) : 'hex';
  } catch {
    return 'hex';
  }
}

export interface ColorPickerProps {
  color: string;
  alpha: number;
  onChange: (hex: string) => void;
  onAlpha: (a: number) => void;
  /** Recent colours, newest first. */
  swatches: string[];
  /** Where it opens: beside this rect (the colour well). */
  anchor: DOMRect;
  onClose: () => void;
  /** Pick a colour from the screen (absent where the browser can't). */
  onEyedropper?: () => void;
}

/** A pointer drag over el: f gets the pointer's position as 0-1 fractions of el's box. */
function useDrag(f: (x: number, y: number) => void) {
  const ref = useRef<HTMLDivElement>(null);
  const fn = useRef(f);
  fn.current = f;
  const at = (e: React.PointerEvent) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    fn.current(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)));
  };
  return {
    ref,
    onPointerDown: (e: React.PointerEvent) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      at(e);
    },
    onPointerMove: (e: React.PointerEvent) => {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) at(e);
    },
  };
}

export default function ColorPicker(props: ColorPickerProps) {
  const { color, alpha, onChange, onAlpha, onClose } = props;
  const [hsv, setHsv] = useState<HSV>(() => rgbToHsv(parseHex(color) || [255, 255, 255]));
  const [space, setSpace] = useState<ColorSpace>(loadSpace);
  const [draft, setDraft] = useState<string[] | null>(null);
  const opened = useRef(color);
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: props.anchor.right + 10, top: props.anchor.top });

  // A colour set from outside (a swatch, the eyedropper) moves the picker; its own changes don't bounce back.
  useEffect(() => {
    if (toHex(hsvToRgb(hsv)).toLowerCase() !== color.toLowerCase()) {
      const rgb = parseHex(color);
      if (rgb) setHsv((prev) => rgbToHsv(rgb, prev));
    }
  }, [color]); // eslint-disable-line

  // Keep it on screen.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({
      left: Math.max(8, Math.min(props.anchor.right + 10, window.innerWidth - r.width - 8)),
      top: Math.max(8, Math.min(props.anchor.top - 40, window.innerHeight - r.height - 8)),
    });
  }, [props.anchor]);

  useEffect(() => {
    const down = (e: PointerEvent) => {
      if (box.current && !box.current.contains(e.target as Node) && !(e.target as HTMLElement).closest?.('.sc-ink-current')) onClose();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('pointerdown', down, true);
    window.addEventListener('keydown', key, true);
    return () => {
      window.removeEventListener('pointerdown', down, true);
      window.removeEventListener('keydown', key, true);
    };
  }, [onClose]);

  const set = (next: HSV) => {
    setHsv(next);
    setDraft(null);
    onChange(toHex(hsvToRgb(next)));
  };
  const field = useDrag((x, y) => set({ ...hsv, s: x, v: 1 - y }));
  const hue = useDrag((x) => set({ ...hsv, h: x * 360 }));
  const opacity = useDrag((x) => onAlpha(Math.max(0.05, Math.round(x * 100) / 100)));

  const hex = toHex(hsvToRgb(hsv));
  const pure = toHex(hsvToRgb({ h: hsv.h, s: 1, v: 1 }));
  const pickSpace = (s: ColorSpace) => {
    setSpace(s);
    setDraft(null);
    try {
      window.localStorage.setItem(SPACE_KEY, s);
    } catch {
      // private window: the choice just isn't remembered
    }
  };

  const fields = space === 'hex' ? null : SPACE_FIELDS[space];
  const values = space === 'hex' ? null : readSpace(space, hsv);
  const commitNumber = (i: number, raw: string) => {
    if (!fields || !values) return;
    const n = Number(raw);
    if (raw.trim() === '' || !Number.isFinite(n)) return;
    const next = values.slice() as [number, number, number];
    next[i] = Math.min(fields[i].max, Math.max(fields[i].min, n));
    set(writeSpace(space as Exclude<ColorSpace, 'hex'>, next, hsv));
  };

  return (
    <div ref={box} className="sc-cpick" style={{ left: pos.left, top: pos.top }} role="dialog" aria-label="Paint colour">
      <div
        className="sc-cpick-field"
        style={{ background: pure }}
        {...field}
        role="slider"
        aria-label="Saturation and brightness"
        aria-valuetext={`saturation ${Math.round(hsv.s * 100)}%, brightness ${Math.round(hsv.v * 100)}%`}
      >
        <div className="sc-cpick-field-white" />
        <div className="sc-cpick-field-black" />
        <span className="sc-cpick-knob" style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%`, background: hex }} />
      </div>
      <div className="sc-cpick-strips">
        <div className="sc-cpick-previews" title={`Was ${opened.current}: click to go back`}>
          <button type="button" className="sc-cpick-was" style={{ background: opened.current }} onClick={() => onChange(opened.current)} aria-label="Back to the colour it was" />
          <span className="sc-cpick-now" style={{ background: hex, opacity: alpha }} />
        </div>
        <div className="sc-cpick-bars">
          <div className="sc-cpick-hue" {...hue} role="slider" aria-label="Hue" aria-valuenow={Math.round(hsv.h)} aria-valuemin={0} aria-valuemax={360}>
            <span className="sc-cpick-thumb" style={{ left: `${(hsv.h / 360) * 100}%` }} />
          </div>
          <div className="sc-cpick-alpha" {...opacity} role="slider" aria-label="Opacity" aria-valuenow={Math.round(alpha * 100)} aria-valuemin={5} aria-valuemax={100}>
            <div className="sc-cpick-alpha-fill" style={{ background: `linear-gradient(to right, transparent, ${hex})` }} />
            <span className="sc-cpick-thumb" style={{ left: `${alpha * 100}%` }} />
          </div>
        </div>
        {props.onEyedropper && (
          <button type="button" className="sc-cpick-drop" onClick={props.onEyedropper} title="Pick a colour from the screen">
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
              <path d="M10.5 2.5l3 3-1.5 1.5-3-3zM9.5 4.5l2 2L5 13H3v-2z" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>

      <div className="sc-seg sc-cpick-spaces" role="radiogroup" aria-label="Colour space">
        {SPACES.map((s) => (
          <button key={s.id} type="button" role="radio" aria-checked={space === s.id} className={space === s.id ? 'on' : ''} title={s.title} onClick={() => pickSpace(s.id)}>
            {s.label}
          </button>
        ))}
      </div>

      <div className="sc-cpick-values">
        {space === 'hex' ? (
          <label className="sc-cpick-value sc-cpick-hex">
            <span>#</span>
            <input
              className="sc-input"
              value={draft ? draft[0] : hex.slice(1)}
              spellCheck={false}
              onChange={(e) => {
                setDraft([e.target.value]);
                const rgb = parseHex(e.target.value);
                if (rgb && e.target.value.replace('#', '').length === 6) {
                  setHsv((prev) => rgbToHsv(rgb, prev));
                  onChange(toHex(rgb));
                }
              }}
              onBlur={() => setDraft(null)}
              aria-label="Hex code"
            />
          </label>
        ) : (
          fields?.map((f, i) => (
            <label key={`${space}-${f.label}`} className="sc-cpick-value" title={`${f.label}: ${f.min}-${f.max}`}>
              <span>{f.label}</span>
              <input
                className="sc-input"
                type="number"
                min={f.min}
                max={f.max}
                step={f.step}
                value={draft ? draft[i] : (values as number[])[i].toFixed(f.digits)}
                onChange={(e) => {
                  const d = draft ? draft.slice() : (values as number[]).map((v, j) => v.toFixed(fields[j].digits));
                  d[i] = e.target.value;
                  commitNumber(i, e.target.value);
                  // What is typed stays as typed (a half-written "0." included) until the field is left.
                  setDraft(d);
                }}
                onBlur={() => setDraft(null)}
                aria-label={f.label}
              />
            </label>
          ))
        )}
        <label className="sc-cpick-value sc-cpick-a" title="Opacity">
          <span>A</span>
          <input
            className="sc-input"
            type="number"
            min={5}
            max={100}
            step={1}
            value={Math.round(alpha * 100)}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) onAlpha(Math.min(100, Math.max(5, n)) / 100);
            }}
            aria-label="Opacity percent"
          />
        </label>
      </div>

      {props.swatches.length > 0 && (
        <div className="sc-cpick-swatches" aria-label="Recent colours">
          {props.swatches.map((c) => (
            <button key={c} type="button" className={c.toLowerCase() === hex ? 'on' : ''} style={{ background: c }} title={c} onClick={() => onChange(c)} aria-label={`Use ${c}`} />
          ))}
        </div>
      )}
    </div>
  );
}
