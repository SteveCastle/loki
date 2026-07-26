/*
 * slangfx studio — audio parameter widgets.
 *
 * Sliders are the wrong instrument for a filter. Frequency is heard
 * logarithmically, so a linear 20 Hz–20 kHz track spends 90% of its travel
 * above 2 kHz and gives you three usable pixels where the bass lives; and
 * a cutoff plus a resonance number tells you nothing about the shape you're
 * actually going to hear.
 *
 * So filters get what every real audio editor gives them: their frequency
 * response, drawn. Drag a handle sideways for cutoff, up and down for the
 * knee — a resonant peak right at the corner, or a shelf/bell's gain. The
 * curve is not an illustration: it's measured from real BiquadFilterNodes
 * configured exactly like the ones in the signal path
 * (getFrequencyResponse), so what you see is the filter you get.
 *
 * UI-free in the same sense as audio-fx.js: the host passes accessors and
 * this module never touches the comp.
 */

const F_MIN = 20;
const F_MAX = 20000;
const DB_RANGE = 24;      // vertical half-range of the plot
const POINTS = 220;

/* One offline context, only ever used to instantiate biquads for their
 * getFrequencyResponse() — it is never rendered. */
let analysisCtx = null;
const ctxFor = () => (analysisCtx ??= new OfflineAudioContext(1, 1, 48000));

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const log = Math.log;

/** Log-mapped position (0..1) of a frequency, and back. */
export const freqToPos = (f) => log(clamp(f, F_MIN, F_MAX) / F_MIN) / log(F_MAX / F_MIN);
export const posToFreq = (p) => F_MIN * (F_MAX / F_MIN) ** clamp(p, 0, 1);

/** Generic log mapping for any positive range (used by log sliders too). */
export const logPos = (v, min, max) => log(clamp(v, min, max) / min) / log(max / min);
export const logVal = (p, min, max) => min * (max / min) ** clamp(p, 0, 1);

const FREQS = (() => {
  const a = new Float32Array(POINTS);
  for (let i = 0; i < POINTS; i++) a[i] = posToFreq(i / (POINTS - 1));
  return a;
})();

/**
 * A draggable frequency-response plot.
 *
 * @param {object} spec  widget descriptor from the effect definition:
 *        {bands: [{type, freq, q?, gain?, label}]}
 * @param {object} io    {get(name), set(name, v), begin(), commit(), meta(name)}
 *        meta returns the parameter def so the widget can respect its
 *        range and step.
 * @returns {{el: HTMLElement, redraw: () => void}}
 */
export function responseWidget(spec, io) {
  const el = document.createElement('div');
  el.className = 'fx-curve';
  // The one thing about this control that isn't self-evident, kept as a
  // tooltip rather than a line of text nobody needs twice.
  el.title = 'drag a handle — sideways: frequency, up: knee · wheel: width';
  const canvas = document.createElement('canvas');
  el.appendChild(canvas);
  const readout = document.createElement('span');
  readout.className = 'fx-curve-read';
  el.appendChild(readout);

  const nodes = spec.bands.map((b) => {
    const n = ctxFor().createBiquadFilter();
    n.type = b.type;
    return n;
  });
  const mag = new Float32Array(POINTS);
  const phase = new Float32Array(POINTS);
  const sum = new Float32Array(POINTS);

  /** Push current parameter values into the measuring nodes. */
  const syncNodes = () => {
    spec.bands.forEach((b, i) => {
      const n = nodes[i];
      n.frequency.value = clamp(io.get(b.freq), 10, 22000);
      if (b.q) n.Q.value = clamp(io.get(b.q), -20, 40);
      else if (b.type === 'peaking' || b.type === 'lowshelf' || b.type === 'highshelf') n.Q.value = 0.9;
      if (b.gain) n.gain.value = clamp(io.get(b.gain), -40, 40);
    });
  };

  /** Total response in dB across the chain. */
  const measure = () => {
    sum.fill(0);
    for (const n of nodes) {
      n.getFrequencyResponse(FREQS, mag, phase);
      for (let i = 0; i < POINTS; i++) sum[i] += 20 * Math.log10(Math.max(mag[i], 1e-6));
    }
  };

  const dbAt = (f) => {
    const p = freqToPos(f) * (POINTS - 1);
    const i = clamp(Math.round(p), 0, POINTS - 1);
    return sum[i];
  };

  const size = () => ({ w: canvas.clientWidth || 1, h: canvas.clientHeight || 1 });
  const xOf = (f) => freqToPos(f) * size().w;
  const yOf = (db) => (0.5 - clamp(db, -DB_RANGE, DB_RANGE) / (2 * DB_RANGE)) * size().h;

  /** Where a band's handle sits: on its own gain for shelves and bells,
   * on the measured curve at the corner for plain filters — which is
   * exactly the resonant peak you're dragging. */
  const handleDb = (b) => (b.gain ? io.get(b.gain) : dbAt(io.get(b.freq)));

  let hover = -1;
  let drag = null;

  function draw() {
    const { w, h } = size();
    const dpr = devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    // grid: decades + the labelled thirds inside them
    c.strokeStyle = 'rgba(255,255,255,0.07)';
    c.fillStyle = 'rgba(255,255,255,0.28)';
    c.font = '9px "Segoe UI", system-ui, sans-serif';
    c.textBaseline = 'bottom';
    c.lineWidth = 1;
    for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
      const x = Math.round(xOf(f)) + 0.5;
      c.beginPath();
      c.moveTo(x, 0);
      c.lineTo(x, h);
      c.stroke();
      if (f === 100 || f === 1000 || f === 10000)
        c.fillText(f >= 1000 ? `${f / 1000}k` : String(f), x + 3, h - 2);
    }
    // 0 dB line, and ±12 dB guides
    for (const db of [-12, 0, 12]) {
      const y = Math.round(yOf(db)) + 0.5;
      c.strokeStyle = db === 0 ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.06)';
      c.beginPath();
      c.moveTo(0, y);
      c.lineTo(w, y);
      c.stroke();
    }

    syncNodes();
    measure();

    // curve + a soft fill down to the 0 dB line
    const zero = yOf(0);
    c.beginPath();
    for (let i = 0; i < POINTS; i++) {
      const x = (i / (POINTS - 1)) * w;
      const y = yOf(sum[i]);
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    const stroke = c.getLineDash ? '#4fc3f7' : '#4fc3f7';
    c.save();
    c.lineTo(w, zero);
    c.lineTo(0, zero);
    c.closePath();
    c.fillStyle = 'rgba(79,195,247,0.13)';
    c.fill();
    c.restore();
    c.strokeStyle = stroke;
    c.lineWidth = 1.6;
    c.beginPath();
    for (let i = 0; i < POINTS; i++) {
      const x = (i / (POINTS - 1)) * w;
      const y = yOf(sum[i]);
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.stroke();

    // handles
    spec.bands.forEach((b, i) => {
      const x = xOf(io.get(b.freq));
      const y = yOf(handleDb(b));
      const active = i === hover || drag?.band === i;
      c.beginPath();
      c.arc(x, y, active ? 6 : 4.5, 0, Math.PI * 2);
      c.fillStyle = active ? '#4fc3f7' : 'rgba(79,195,247,0.75)';
      c.fill();
      c.strokeStyle = 'rgba(0,0,0,0.55)';
      c.lineWidth = 1;
      c.stroke();
      if (spec.bands.length > 1) {
        c.fillStyle = 'rgba(255,255,255,0.5)';
        c.textBaseline = 'top';
        c.fillText(b.label ?? '', x + 7, y - 4);
      }
    });
  }

  const fmtHz = (f) => (f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 1 : 2)} kHz` : `${Math.round(f)} Hz`);

  const showReadout = (i) => {
    const b = spec.bands[i];
    if (!b) { readout.textContent = ''; return; }
    const bits = [fmtHz(io.get(b.freq))];
    if (b.gain) bits.push(`${io.get(b.gain) >= 0 ? '+' : ''}${io.get(b.gain).toFixed(1)} dB`);
    if (b.q) bits.push(`Q ${io.get(b.q).toFixed(2)}`);
    readout.textContent = bits.join(' · ');
  };

  const nearest = (px, py) => {
    let best = -1;
    let bestD = Infinity;
    spec.bands.forEach((b, i) => {
      const dx = px - xOf(io.get(b.freq));
      const dy = py - yOf(handleDb(b));
      const d = Math.hypot(dx, dy * 0.6);   // favour horizontal proximity
      if (d < bestD) { bestD = d; best = i; }
    });
    return { i: best, d: bestD };
  };

  const local = (e) => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const quant = (v, step) => (step ? Math.round(v / step) * step : v);

  /** Vertical drag → the band's knee. Three cases, because Web Audio's Q
   * means different things per filter type:
   *   gain bands (shelf / bell)  the dB under the pointer IS the gain
   *   qMode 'db'  (low/high-pass) Q is stated in decibels, and that number
   *               is the height of the resonant peak — so, again, direct
   *   qMode 'factor' (bandpass / notch) Q is a plain quality factor and
   *               nothing on the dB axis corresponds to it; drag height
   *               walks the parameter's own (log) range instead. */
  const applyY = (b, db, py, h) => {
    if (b.gain) {
      const m = io.meta(b.gain);
      io.set(b.gain, clamp(quant(db, m.step), m.min, m.max));
      return;
    }
    if (!b.q) return;
    const m = io.meta(b.q);
    if (b.qMode === 'factor') {
      const pos = clamp(1 - py / h, 0, 1);       // up = narrower
      io.set(b.q, clamp(quant(logVal(pos, m.min, m.max), m.step), m.min, m.max));
    } else {
      io.set(b.q, clamp(quant(db, m.step), m.min, m.max));
    }
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const [px, py] = local(e);
    const { i } = nearest(px, py);
    drag = { band: i };
    io.begin();
    try { canvas.setPointerCapture(e.pointerId); } catch {}
    move(e);
    e.preventDefault();
  });

  function move(e) {
    const [px, py] = local(e);
    if (!drag) {
      const { i, d } = nearest(px, py);
      const h = d < 26 ? i : -1;
      if (h !== hover) { hover = h; draw(); }
      showReadout(h >= 0 ? h : (spec.bands.length === 1 ? 0 : -1));
      return;
    }
    const b = spec.bands[drag.band];
    const { w, h } = size();
    const fm = io.meta(b.freq);
    io.set(b.freq, clamp(quant(posToFreq(px / w), fm.step), fm.min, fm.max));
    applyY(b, (0.5 - py / h) * 2 * DB_RANGE, py, h);
    showReadout(drag.band);
    draw();
  }

  canvas.addEventListener('pointermove', move);
  const end = () => {
    if (!drag) return;
    drag = null;
    io.commit();
    draw();
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => {
    if (drag) return;
    hover = -1;
    readout.textContent = '';
    draw();
  });

  // Wheel narrows / widens the band under the cursor — the other half of
  // "knee" for a bell, and resonance for a plain filter.
  canvas.addEventListener('wheel', (e) => {
    const [px, py] = local(e);
    const { i, d } = nearest(px, py);
    const b = spec.bands[i];
    if (!b?.q || d > 40) return;
    e.preventDefault();
    const m = io.meta(b.q);
    io.begin();
    io.set(b.q, clamp(io.get(b.q) * (e.deltaY < 0 ? 1.18 : 1 / 1.18), m.min, m.max));
    io.commit();
    showReadout(i);
    draw();
  }, { passive: false });

  const ro = new ResizeObserver(() => draw());
  ro.observe(canvas);

  return { el, redraw: draw };
}
