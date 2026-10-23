import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke, isElectron, mediaUrl, send } from '../../platform';
import { clampRect, imageOp, motionRegion, pickedOp, maskImagesReady, opsDeep, preloadMaskImages, rasterizeLayerMask, rasterizeProtect, regionAround, type MaskLookup, type MaskOp, type MaskOpts, type Rect } from './mask';
import {
  generationsInFlight,
  layerRate,
  MAX_GENERATIONS,
  mergeServer,
  sceneApi,
  sceneBusy,
  type LayerKind,
  type LayerOptions,
  type LoopKind,
  type Scene,
  type SceneLayer,
  type SceneSummary,
  type SegPrompt,
  versionStale,
} from './scene-api';
import Stage, { isTyping, type MaskView, type RotoView, type Tool } from './stage';
import { pickOps, usePick, type PickTarget } from './pick';
import { PickBar, RotoBar } from './pick-bar';
import MaskPanel from './mask-panel';
import DreamDialog from './dream-dialog';
import PaintPanel from './paint-panel';
import { Inspector, KEY_GENERATE, KEY_UNDO, LayerList } from './panel';
import { requestLibraryPick } from './pick-bridge';
import { hasVoice, newRefId, refTokens, remapRefTokens, tokenIssues, type RefKind } from './refs';

// The Scene editor: one picture, built up a masked region at a time. Each
// layer is an Edit (a still change of its area) or Motion (its area brought to
// life); the server crops each layer's area from the stills below it, runs the
// engine on the crop and composites the result back through the mask. What the
// stage shows is what the export renders.
//
// The editor owns the document's editable fields and autosaves them; the
// server owns generations (versions, jobs, errors) and the export, folded in
// by mergeServer on every answer and on a poll while anything is running.

const TOOLS: Array<{ id: Tool; key: string; label: string; title: string }> = [
  { id: 'select', key: 'V', label: 'Select', title: 'Select a layer by clicking its area; drag to pan' },
  {
    id: 'pick',
    key: 'Q',
    label: 'Select object',
    title: 'Click an object to select it (Alt-click: not this), drag a box round it, or find it by name; Tab: part / object / whole; Enter adds it to the mask',
  },
  { id: 'box', key: 'B', label: 'Box', title: 'Box an area (Shift = square, Alt = subtract)' },
  { id: 'brush', key: 'P', label: 'Brush', title: 'Paint an area ([ and ] change the size, Alt = erase)' },
  { id: 'lasso', key: 'L', label: 'Lasso', title: 'Draw around an area (Alt = subtract)' },
  { id: 'pen', key: 'N', label: 'Pen', title: 'Click points around an area; click the first point, double-click or Enter to close (Backspace = undo a point, Alt on the first click = subtract)' },
  { id: 'erase', key: 'E', label: 'Erase', title: 'Erase from the mask' },
  { id: 'protect', key: 'S', label: 'Protect', title: 'Paint areas this layer must leave untouched: they stay exactly the picture below (blue)' },
  { id: 'shape', key: 'T', label: 'Shape', title: 'Click a shape of the mask to select it; drag to move, drag a corner to scale (Shift = keep proportions), drag a point to reshape' },
  { id: 'hand', key: 'H', label: 'Pan', title: 'Pan (or hold Space); scroll to zoom, F to fit' },
];

function ToolIcon({ id }: { id: Tool }) {
  const p = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (id) {
    case 'select':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M4 3l9 5.5-4 1-1.8 4.2z" {...p} />
        </svg>
      );
    case 'box':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <rect x="3" y="4" width="12" height="10" rx="1" {...p} strokeDasharray="2.5 2" />
        </svg>
      );
    case 'brush':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M14.5 3.5l-6.5 7M8 10.5c-2 0-3 1.2-3 2.6 0 .9-.6 1.6-1.6 1.9 3.3.8 5.6-.4 5.6-2.6z" {...p} />
        </svg>
      );
    case 'lasso':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M9 3.5c3.6 0 6 1.8 6 4s-2.4 4-6 4-6-1.8-6-4 2.4-4 6-4zM5.5 10.5c-.5 1.3-.2 2.5.8 3.2s1 1.8.2 2.3" {...p} />
        </svg>
      );
    case 'pick':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M3 15l7-7M10.5 3v2.5M14.5 7H12M13.3 4.2l-1.8 1.8M7 4.2l1.8 1.8" {...p} />
          <circle cx="10.5" cy="7.5" r="1.2" {...p} />
        </svg>
      );
    case 'pen':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M4 14l2.5-6.5L12 4l2 2-3.5 5.5zM4 14l3.6-3.6" {...p} />
          <circle cx="8.6" cy="9.4" r="1" {...p} />
        </svg>
      );
    case 'shape':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <rect x="3.5" y="3.5" width="11" height="11" {...p} strokeDasharray="2 2" />
          <rect x="2" y="2" width="3" height="3" {...p} fill="currentColor" />
          <rect x="13" y="13" width="3" height="3" {...p} fill="currentColor" />
        </svg>
      );
    case 'erase':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M3.5 11.5l6.8-6.8 4 4-5 5H6.5zM7 14.7h8" {...p} />
        </svg>
      );
    case 'protect':
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M9 2.5l5.5 2v4.2c0 3.3-2.3 5.7-5.5 6.8-3.2-1.1-5.5-3.5-5.5-6.8V4.5z" {...p} />
        </svg>
      );
    default:
      return (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <path d="M6 9V4.5a1.2 1.2 0 012.4 0V8m0-4.5V3a1.2 1.2 0 012.4 0v5m0-4a1.2 1.2 0 012.4 0v5.5c0 3-2 5-4.8 5S5 13.8 4 12L2.8 9.6a1.1 1.1 0 011.9-1.1L6 10" {...p} />
        </svg>
      );
  }
}

function newId(): string {
  return `l${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function defaultFeather(s: Scene): number {
  return Math.round(Math.min(64, Math.max(4, Math.min(s.width, s.height) * 0.02)));
}

/** How layer ops of s find the layer they name. */
function lookupIn(s: Scene): MaskLookup {
  return (id) => s.layers.find((l) => l.id === id) || null;
}

/**
 * The layer with its mask raster and next-generation region recomputed from its shapes and mask settings. While an
 * image piece of the mask is still decoding it stays as it was (patchLayer re-rasterizes once it is ready).
 */
function withMask(s: Scene, l: SceneLayer): SceneLayer {
  const paint = l.kind === 'paint';
  const r = rasterizeLayerMask(l.shapes || [], l.maskOpts, s.width, s.height, l.feather, lookupIn(s), l.id, paint);
  if (!r) return l;
  // A paint layer is never generated: its region is just what is painted (for selecting it on the stage).
  if (paint) return { ...l, mask: r.mask, region: r.bounds ? clampRect(r.bounds, s.width, s.height) : { x: 0, y: 0, w: 0, h: 0 } };
  const region = regionAround(r.bounds, s.width, s.height, {
    context: l.options?.context || 'normal',
    feather: l.feather,
    minSide: l.kind === 'motion' ? 320 : 96,
  });
  // Motion: the trained shape the server grows it to, so the box shows what the engine really sees.
  const shaped = region && l.kind === 'motion' ? motionRegion(clampRect(region, s.width, s.height), s.width, s.height) : region;
  return { ...l, mask: r.mask, region: shaped || { x: 0, y: 0, w: 0, h: 0 } };
}

const linksMasks = (l: SceneLayer) => (l.shapes || []).some((op) => op.t === 'layer');

/** s with layer id re-masked, then every layer that links a mask (they follow the layers they name live). */
function remask(s: Scene, id: string): Scene {
  const once = { ...s, layers: s.layers.map((l) => (l.id === id ? withMask(s, l) : l)) };
  return { ...once, layers: once.layers.map((l) => (l.id !== id && linksMasks(l) ? withMask(once, l) : l)) };
}

// New layers start from the settings last used for their kind (never the text: prompt, subject, seed).
const DEFAULTS_KEY = 'loki.scene.layerDefaults.v1';
const REMEMBERED: Record<LayerKind, Array<keyof LayerOptions>> = {
  still: ['steps', 'context'],
  motion: ['duration', 'model', 'megapixels', 'sound', 'voices', 'lock', 'detail', 'context'],
  paint: ['guide'],
};
const BASE_OPTIONS: Record<LayerKind, LayerOptions> = {
  still: { context: 'normal', steps: 25 },
  motion: { context: 'normal', duration: 5, model: 'h3', sound: false, megapixels: 0.5 },
  paint: {},
};

// The screen colour picker (Chromium; TypeScript 5.0 has no types for it).
const EyeDropperApi = (window as unknown as { EyeDropper?: new () => { open: () => Promise<{ sRGBHex: string }> } }).EyeDropper;

// The paint colours used last (newest first), kept across scenes.
const SWATCHES_KEY = 'loki.scene.swatches.v1';
const START_SWATCHES = ['#e2533b', '#f2b33d', '#4caf50', '#3b82f6', '#8b5cf6', '#f5d0b5', '#ffffff', '#111111'];

function loadSwatches(): string[] {
  try {
    const v = JSON.parse(window.localStorage.getItem(SWATCHES_KEY) || 'null');
    return Array.isArray(v) && v.length ? v.slice(0, 8) : START_SWATCHES;
  } catch {
    return START_SWATCHES;
  }
}


function layerDefaults(kind: LayerKind): LayerOptions {
  try {
    const all = JSON.parse(window.localStorage.getItem(DEFAULTS_KEY) || '{}');
    return { ...BASE_OPTIONS[kind], ...(all[kind] || {}) };
  } catch {
    return { ...BASE_OPTIONS[kind] };
  }
}

function rememberLayerDefaults(kind: LayerKind, options: LayerOptions): void {
  try {
    const all = JSON.parse(window.localStorage.getItem(DEFAULTS_KEY) || '{}');
    const picked: Record<string, unknown> = {};
    REMEMBERED[kind].forEach((k) => {
      if (options[k] !== undefined) picked[k] = options[k];
    });
    all[kind] = picked;
    window.localStorage.setItem(DEFAULTS_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: defaults just don't stick
  }
}

const basename = (p: string) => p.split(/[\\/]/).pop() || p;

interface Undo {
  id: string;
  shapes: MaskOp[];
  maskOpts?: MaskOpts;
  /** Changes with the same key in quick succession (a slider drag) are one step. */
  key?: string;
  at?: number;
}

export default function SceneEditor({ initial, onSwitch }: { initial: Scene; onSwitch: (s: Scene) => void }) {
  const [doc, setDoc] = useState<Scene>(initial);
  const docRef = useRef(doc);
  docRef.current = doc;
  const [selectedId, setSelectedId] = useState<string | null>(initial.layers.length ? initial.layers[initial.layers.length - 1].id : null);
  const [tool, setTool] = useState<Tool>('select');
  const [brush, setBrush] = useState(() => Math.max(6, Math.round(Math.min(initial.width, initial.height) * 0.035)));
  const [maskView, setMaskView] = useState<MaskView>('tint');
  const [hardness, setHardness] = useState(1);
  const [ink, setInk] = useState(() => loadSwatches()[0]);
  const [inkAlpha, setInkAlpha] = useState(1);
  const [swatches, setSwatches] = useState(loadSwatches);
  const [selectedShape, setSelectedShape] = useState<number | null>(null);
  const [compare, setCompare] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [soundOn, setSoundOn] = useState(false);
  const [syncKey, setSyncKey] = useState(0);
  const [fitKey, setFitKey] = useState(0);
  const [saveState, setSaveState] = useState<'saved' | 'dirty' | 'saving' | 'error'>('saved');
  const [notice, setNotice] = useState<{ tone: 'error' | 'info'; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [others, setOthers] = useState<SceneSummary[]>([]);
  const [dreamOpen, setDreamOpen] = useState(false);
  const undo = useRef<Undo[]>([]);
  const redo = useRef<Undo[]>([]);

  const selected = doc.layers.find((l) => l.id === selectedId) || null;
  // The shape being edited belongs to one layer.
  useEffect(() => setSelectedShape(null), [selectedId]);
  useEffect(() => {
    if (selectedShape !== null && selectedShape >= (selected?.shapes || []).length) setSelectedShape(null);
  }, [selected, selectedShape]);

  const flash = useCallback((tone: 'error' | 'info', text: string) => setNotice({ tone, text }), []);
  useEffect(() => {
    if (!notice) return undefined;
    const t = window.setTimeout(() => setNotice(null), notice.tone === 'error' ? 9000 : 4000);
    return () => window.clearTimeout(t);
  }, [notice]);

  // ---- saving ----------------------------------------------------------------

  const saveTimer = useRef<number | undefined>();
  const saving = useRef<Promise<void> | null>(null);
  const dirty = useRef(false);

  const save = useCallback(async (): Promise<void> => {
    if (saving.current) {
      await saving.current;
      if (!dirty.current) return;
    }
    window.clearTimeout(saveTimer.current);
    if (!dirty.current) return;
    dirty.current = false;
    setSaveState('saving');
    const run = (async () => {
      try {
        const server = await sceneApi.save(docRef.current);
        setDoc((prev) => mergeServer(prev, server));
        setSaveState(dirty.current ? 'dirty' : 'saved');
      } catch (e) {
        dirty.current = true;
        setSaveState('error');
        throw e;
      }
    })();
    saving.current = run.finally(() => {
      saving.current = null;
    });
    await run;
  }, []);

  const touch = useCallback(() => {
    dirty.current = true;
    setSaveState('dirty');
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      save().catch((e) => flash('error', `Not saved: ${e instanceof Error ? e.message : e}`));
    }, 700);
  }, [save, flash]);

  // Last-chance save when the window closes (best effort: masks can exceed keepalive's limit).
  useEffect(() => {
    const onHide = () => {
      if (dirty.current) save().catch(() => undefined);
    };
    window.addEventListener('pagehide', onHide);
    window.addEventListener('beforeunload', onHide);
    return () => {
      window.removeEventListener('pagehide', onHide);
      window.removeEventListener('beforeunload', onHide);
    };
  }, [save]);

  const mutate = useCallback(
    (fn: (s: Scene) => Scene) => {
      setDoc((prev) => fn(prev));
      touch();
    },
    [touch]
  );

  const patchLayer = useCallback(
    (id: string, patch: Partial<SceneLayer>, again = false): void => {
      mutate((s) => {
        const next = { ...s, layers: s.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)) };
        return again ? remask(next, id) : next;
      });
      // An image piece still decoding kept the old raster: redo it once the piece is ready.
      const shapes = patch.shapes;
      const deep = shapes ? opsDeep(shapes, lookupIn(docRef.current)) : [];
      if (again && shapes && !maskImagesReady(deep)) {
        preloadMaskImages(deep).then(() => {
          const l = docRef.current.layers.find((x) => x.id === id);
          if (l && l.shapes === shapes) patchLayer(id, {}, true);
        });
      }
    },
    [mutate]
  );

  // Decode every image piece of every mask up front, so later edits re-rasterize synchronously.
  useEffect(() => {
    preloadMaskImages(initial.layers.flatMap((l) => l.shapes || []));
  }, [initial]);

  /** Remember a layer's mask before a change (one undo step; changes with the same key within a second merge). */
  const pushUndo = useCallback((l: SceneLayer, key?: string) => {
    const last = undo.current[undo.current.length - 1];
    const now = Date.now();
    if (key && last && last.id === l.id && last.key === key && now - (last.at || 0) < 1000) {
      last.at = now;
      return;
    }
    undo.current.push({ id: l.id, shapes: l.shapes || [], maskOpts: l.maskOpts, key, at: now });
    if (undo.current.length > 200) undo.current.shift();
    redo.current = [];
  }, []);

  // ---- polling while the server works ------------------------------------------

  const busyOnServer = sceneBusy(doc);
  const refresh = useCallback(async () => {
    try {
      const server = await sceneApi.get(docRef.current.id);
      setDoc((prev) => mergeServer(prev, server));
    } catch {
      // transient: the next tick retries
    }
  }, []);
  useEffect(() => {
    if (!busyOnServer) return undefined;
    const t = window.setInterval(refresh, 1500);
    return () => window.clearInterval(t);
  }, [busyOnServer, refresh]);
  useEffect(() => {
    window.addEventListener('focus', refresh);
    return () => window.removeEventListener('focus', refresh);
  }, [refresh]);

  // Finished exports announce themselves once.
  const lastRender = useRef(doc.render?.state);
  useEffect(() => {
    const r = doc.render;
    if (r && r.state !== lastRender.current) {
      if (r.state === 'completed' && r.output) flash('info', `Exported ${basename(r.output)} next to the original.`);
      if (r.state === 'error' || r.state === 'cancelled') flash('error', `Export failed${r.error ? `: ${r.error}` : ''}`);
    }
    lastRender.current = r?.state;
  }, [doc.render, flash]);

  // Other scenes of this source, for the switcher.
  useEffect(() => {
    sceneApi.list(doc.source).then(setOthers, () => undefined);
  }, [doc.source, doc.id]);

  // ---- layer operations ------------------------------------------------------------

  const addLayer = useCallback(
    (kind: LayerKind) => {
      const s = docRef.current;
      const n = s.layers.filter((l) => l.kind === kind).length + 1;
      const options = layerDefaults(kind);
      const names: Record<LayerKind, string> = { still: 'Edit', motion: 'Motion', paint: 'Paint' };
      const layer: SceneLayer = {
        id: newId(),
        name: `${names[kind]} ${n}`,
        kind,
        region: { x: 0, y: 0, w: 0, h: 0 },
        mask: '',
        shapes: [],
        feather: kind === 'paint' ? 0 : defaultFeather(s),
        prompt: '',
        options,
        versions: [],
        active: -1,
      };
      mutate((d) => ({ ...d, layers: [...d.layers, layer] }));
      setSelectedId(layer.id);
      setTool(kind === 'still' ? 'box' : 'brush');
      return layer.id;
    },
    [mutate]
  );

  /** A paint stroke takes the current colour, which becomes the newest swatch. */
  const rememberInk = useCallback((c: string) => {
    setInk(c);
    setSwatches((prev) => {
      const next = [c, ...prev.filter((x) => x.toLowerCase() !== c.toLowerCase())].slice(0, 8);
      try {
        window.localStorage.setItem(SWATCHES_KEY, JSON.stringify(next));
      } catch {
        // storage unavailable: swatches just don't stick
      }
      return next;
    });
  }, []);

  const commitOp = useCallback(
    (op: MaskOp) => {
      if (!selected) return;
      let next = op;
      if (selected.kind === 'paint') {
        // Paint has no Protect; everything else is paint in the current colour, or erase.
        if (op.keep) return;
        if (!op.erase) {
          next = { ...op, c: ink, a: inkAlpha < 1 ? inkAlpha : undefined };
          rememberInk(ink);
        }
      }
      const before = selected.shapes || [];
      pushUndo(selected);
      patchLayer(selected.id, { shapes: [...before, next] }, true);
    },
    [selected, patchLayer, pushUndo, ink, inkAlpha, rememberInk]
  );

  /**
   * "Finish with AI" on a paint layer: an Edit layer just above it, masked to what is painted (a live link, grown a
   * little: rough paint stops short of edges), prompted to finish it. A guide is given to it as a reference.
   */
  const finishPaint = useCallback(
    (paint: SceneLayer) => {
      const s = docRef.current;
      const guide = !!paint.options?.guide;
      const refs = guide ? [{ id: newRefId(), path: '', layer: paint.id, note: 'the rough colour guide' }] : [];
      // The engine sees the whole picture: what a sketch should become depends on everything around it (a disc in a
      // crop of flat sky reads as a ball or the moon; with the land and the house in view it is the sun).
      const context = 'full';
      const layer: SceneLayer = {
        id: newId(),
        name: `Finish ${paint.name || 'paint'}`,
        kind: 'still',
        region: { x: 0, y: 0, w: 0, h: 0 },
        mask: '',
        shapes: [{ t: 'layer', id: paint.id }],
        maskOpts: { expand: Math.round(Math.min(s.width, s.height) * 0.01) || undefined },
        feather: defaultFeather(s),
        prompt: '',
        options: { ...layerDefaults('still'), context, refs, fit: true, finish: guide ? 'colour' : 'sketch', paint: paint.id },
        versions: [],
        active: -1,
      };
      mutate((d) => {
        const at = d.layers.findIndex((l) => l.id === paint.id) + 1;
        const layers = [...d.layers];
        layers.splice(at > 0 ? at : layers.length, 0, layer);
        return remask({ ...d, layers }, layer.id);
      });
      setSelectedId(layer.id);
      setTool('select');
      flash('info', guide ? 'Added an Edit layer that colours the picture from your guide: add any notes, then generate.' : 'Added an Edit layer that makes your sketch real: say what it is, then generate.');
      window.setTimeout(() => document.getElementById('sc-prompt')?.focus(), 50);
    },
    [mutate, flash]
  );

  /** Replace the selected layer's shapes (the mask panel, the Shape tool): one undo step. */
  const setShapes = useCallback(
    (ops: MaskOp[]) => {
      if (!selected) return;
      pushUndo(selected);
      patchLayer(selected.id, { shapes: ops }, true);
    },
    [selected, patchLayer, pushUndo]
  );

  const stepHistory = useCallback(
    (from: React.MutableRefObject<Undo[]>, to: React.MutableRefObject<Undo[]>) => {
      const entry = from.current.pop();
      if (!entry) return;
      const layer = docRef.current.layers.find((l) => l.id === entry.id);
      if (!layer) return;
      to.current.push({ id: entry.id, shapes: layer.shapes || [], maskOpts: layer.maskOpts });
      setSelectedId(entry.id);
      patchLayer(entry.id, { shapes: entry.shapes, maskOpts: entry.maskOpts }, true);
    },
    [patchLayer]
  );

  /** Move layer id to index `to` of the stack (bottom = 0). */
  const moveLayerTo = useCallback(
    (id: string, to: number) =>
      mutate((s) => {
        const i = s.layers.findIndex((l) => l.id === id);
        if (i < 0) return s;
        const layers = s.layers.filter((l) => l.id !== id);
        layers.splice(Math.max(0, Math.min(layers.length, to)), 0, s.layers[i]);
        return { ...s, layers };
      }),
    [mutate]
  );

  const moveLayer = useCallback(
    (id: string, dir: 1 | -1) => {
      const i = docRef.current.layers.findIndex((l) => l.id === id);
      if (i >= 0) moveLayerTo(id, i + dir);
    },
    [moveLayerTo]
  );

  const deleteLayer = useCallback(
    (id: string) => {
      // Masks that linked it let go of it.
      mutate((s) => {
        const linked = (op: MaskOp) => op.t === 'layer' && op.id === id;
        const unref = (l: SceneLayer): SceneLayer => {
          const refs = l.options?.refs || [];
          if (!refs.some((r) => r.layer === id)) return l;
          const kept = refs.filter((r) => r.layer !== id);
          const prompt = remapRefTokens(l.prompt, l.kind, refs, kept, hasVoice(l.options));
          return { ...l, prompt, options: { ...l.options, refs: kept } };
        };
        const next = {
          ...s,
          layers: s.layers
            .filter((l) => l.id !== id)
            .map(unref)
            .map((l) => ((l.shapes || []).some(linked) ? { ...l, shapes: (l.shapes || []).filter((op) => !linked(op)) } : l)),
        };
        return { ...next, layers: next.layers.map((l) => (linksMasks(l) || l !== s.layers.find((x) => x.id === l.id) ? withMask(next, l) : l)) };
      });
      undo.current = undo.current.filter((u) => u.id !== id);
      redo.current = redo.current.filter((u) => u.id !== id);
      setSelectedId(null);
    },
    [mutate]
  );

  const serverAction = useCallback(
    async (fn: () => Promise<Scene>, failure: string) => {
      setBusy(true);
      try {
        await save();
        const server = await fn();
        setDoc((prev) => mergeServer(prev, server));
      } catch (e) {
        flash('error', `${failure}: ${e instanceof Error ? e.message : e}`);
      } finally {
        setBusy(false);
      }
    },
    [save, flash]
  );

  const generate = useCallback(() => {
    const l = selected;
    if (!l || l.kind === 'paint' || generationsInFlight(l) >= MAX_GENERATIONS) return;
    if (!l.mask) {
      flash('info', 'Paint or box the area this layer should change first.');
      return;
    }
    if (l.kind === 'still' && !l.prompt.trim() && !l.options?.finish) {
      flash('info', 'Describe the edit first.');
      return;
    }
    const bad = tokenIssues(l.prompt, l.kind, l.options?.refs || [], hasVoice(l.options));
    if (bad.length) {
      flash('error', `${bad.join(', ')} in the prompt ${bad.length > 1 ? "don't" : "doesn't"} name any reference of this layer.`);
      return;
    }
    serverAction(() => sceneApi.generate(doc.id, l.id), 'Could not start the generation');
  }, [selected, doc.id, serverAction, flash]);

  // ---- versions: loops, motion maps ---------------------------------------------------

  const loopVersion = useCallback(
    (layerId: string, n: number, kind: LoopKind) => {
      const label = kind === 'natural' ? 'Looking for a natural loop…' : 'Making the loop…';
      flash('info', label);
      serverAction(() => sceneApi.loop(doc.id, layerId, n, kind), 'Could not make the loop');
    },
    [doc.id, serverAction, flash]
  );

  const [motionView, setMotionView] = useState<{ layerId: string; n: number; path: string; dataUrl: string; region: Rect } | null>(null);
  const motionMap = useCallback(
    async (l: SceneLayer) => {
      const v = l.versions[l.active];
      if (!v) return null;
      if (motionView && motionView.layerId === l.id && motionView.path === v.path) return motionView;
      const m = await sceneApi.motionMap(doc.id, l.id, l.active);
      const view = { layerId: l.id, n: l.active, path: v.path, dataUrl: m.dataUrl, region: m.region };
      setMotionView(view);
      return view;
    },
    [doc.id, motionView]
  );

  const toggleMotion = useCallback(
    async (l: SceneLayer) => {
      if (motionView && motionView.layerId === l.id && motionView.n === l.active) {
        setMotionView(null);
        return;
      }
      setBusy(true);
      try {
        await motionMap(l);
      } catch (e) {
        flash('error', `Could not measure the motion: ${e instanceof Error ? e.message : e}`);
      } finally {
        setBusy(false);
      }
    },
    [motionView, motionMap, flash]
  );

  const growToMotion = useCallback(
    async (l: SceneLayer) => {
      setBusy(true);
      try {
        const view = await motionMap(l);
        if (!view) return;
        const op = await imageOp(view.dataUrl, view.region, 48);
        if (!op) {
          flash('info', 'This version barely moves: nothing to add.');
          return;
        }
        const before = l.shapes || [];
        pushUndo(l);
        patchLayer(l.id, { shapes: [...before, op] }, true);
        flash('info', 'Added where it moves to the mask (Ctrl+Z to undo).');
      } catch (e) {
        flash('error', `Could not grow the mask: ${e instanceof Error ? e.message : e}`);
      } finally {
        setBusy(false);
      }
    },
    [motionMap, patchLayer, pushUndo, flash]
  );

  // Fit mask to result: a still layer with `fit` carries the change map of its shown version as one image op,
  // replaced whenever another version is shown (each version is fitted once per session; a failure is not retried).
  const fitting = useRef(new Set<string>());
  useEffect(() => {
    doc.layers.forEach((l) => {
      if (l.kind !== 'still') return;
      const ops = l.shapes || [];
      const fitted = ops.filter((op) => op.t === 'image' && op.fit);
      const v = l.options?.fit ? l.versions[l.active] : undefined;
      if (!v) {
        if (fitted.length) patchLayer(l.id, { shapes: ops.filter((op) => !(op.t === 'image' && op.fit)) }, true);
        return;
      }
      if (fitted.length === 1 && fitted[0].fit === v.path) return;
      const key = `${l.id}|${v.path}`;
      if (fitting.current.has(key)) return;
      fitting.current.add(key);
      sceneApi
        .changeMap(doc.id, l.id, l.active)
        .then(async (m) => {
          const op = await imageOp(m.dataUrl, m.region, 40);
          const cur = docRef.current.layers.find((x) => x.id === l.id);
          if (!cur || cur.versions[cur.active]?.path !== v.path) return;
          const rest = (cur.shapes || []).filter((x) => !(x.t === 'image' && x.fit));
          patchLayer(l.id, { shapes: op ? [...rest, { ...op, fit: v.path }] : rest }, true);
        })
        .catch((e) => flash('error', `Could not fit the mask to the result: ${e instanceof Error ? e.message : e}`));
    });
  }, [doc.layers, doc.id, patchLayer, flash]);

  // A voice reference: a native dialog in Electron, the server's file browser in the web build.
  const chooseVoice = useCallback(async (): Promise<string | null> => {
    try {
      if (isElectron) return ((await invoke('select-voice-file', [])) as string | null) || null;
      const { openFileBrowser } = await import('../controls/file-browser-modal');
      return (await openFileBrowser('file')) || null;
    } catch {
      return null;
    }
  }, []);

  // References from outside the library: a native multi-file dialog, or the server's file browser on the web.
  const pickFiles = useCallback(async (kinds: RefKind[]): Promise<string[]> => {
    try {
      if (isElectron) return ((await invoke('select-reference-files', [kinds])) as string[]) || [];
      const { openFileBrowser } = await import('../controls/file-browser-modal');
      const p = await openFileBrowser('file');
      return p ? [p] : [];
    } catch {
      return [];
    }
  }, []);

  const protectMask = useMemo(
    () => (selected ? rasterizeProtect(selected.shapes || [], doc.width, doc.height, selected.feather) : ''),
    [selected, doc.width, doc.height]
  );

  const exportAs = useCallback(
    (format: 'video' | 'still') => serverAction(() => sceneApi.render(doc.id, format), 'Could not start the export'),
    [doc.id, serverAction]
  );

  // ---- selecting objects (Select tool) and roto ------------------------------------------

  const picker = usePick(doc.id, useCallback((msg: string) => flash('error', `Could not select: ${msg}`), [flash]));
  const { pick } = picker;
  // A roto: picking on a motion version's clip, frame by frame, then tracking it.
  const [roto, setRoto] = useState<{ layerId: string; n: number; frame: number } | null>(null);
  const [rotoPrompts, setRotoPrompts] = useState<SegPrompt[]>([]);
  const [showMatte, setShowMatte] = useState(true);
  const rotoLayer = roto ? doc.layers.find((l) => l.id === roto.layerId) || null : null;
  const rotoVersion = roto && rotoLayer ? rotoLayer.versions[roto.n] || null : null;
  const ROTO_FPS = 24;
  const rotoFrames = rotoVersion ? rotoVersion.track?.frames || Math.max(1, Math.round((rotoVersion.duration || 5) * ROTO_FPS)) : 0;
  // The clip's own pixel size (picks on it are in clip pixels).
  const [clipSize, setClipSize] = useState<{ path: string; w: number; h: number } | null>(null);
  useEffect(() => {
    if (!rotoVersion || clipSize?.path === rotoVersion.path) return;
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.muted = true;
    v.onloadedmetadata = () => setClipSize({ path: rotoVersion.path, w: v.videoWidth, h: v.videoHeight });
    v.src = mediaUrl(rotoVersion.path);
  }, [rotoVersion, clipSize]);

  const pickTarget = useCallback((): PickTarget => (roto ? { kind: 'clip', layer: roto.layerId, version: roto.n, frame: roto.frame } : { kind: 'picture' }), [roto]);

  /** Frame pixels -> what is picked on (clip pixels in a roto); null when outside the clip. */
  const toTarget = useCallback(
    (x: number, y: number): [number, number] | null => {
      if (!roto || !rotoVersion) return [x, y];
      const r = rotoVersion.region;
      if (!clipSize || x < r.x || y < r.y || x > r.x + r.w || y > r.y + r.h) return null;
      return [((x - r.x) * clipSize.w) / r.w, ((y - r.y) * clipSize.h) / r.h];
    },
    [roto, rotoVersion, clipSize]
  );

  const onPickClick = useCallback(
    (x: number, y: number, keep: boolean) => {
      const t = toTarget(x, y);
      if (t) picker.click(pickTarget(), t[0], t[1], keep);
      else flash('info', 'Click on the clip (inside its area) to pick in it.');
    },
    [toTarget, picker, pickTarget, flash]
  );
  const onPickBox = useCallback(
    (r: Rect) => {
      const a = toTarget(r.x, r.y);
      const b = toTarget(r.x + r.w, r.y + r.h);
      if (a && b) picker.box(pickTarget(), { x: a[0], y: a[1], w: b[0] - a[0], h: b[1] - a[1] });
      else flash('info', 'Box inside the clip’s area to pick in it.');
    },
    [toTarget, picker, pickTarget, flash]
  );

  /** The selection into the selected layer's mask (added, or cut). */
  const commitPick = useCallback(
    (cut: boolean) => {
      if (!pick || pick.busy || !selected) return;
      const ops = pickOps(pick, cut, doc.width, doc.height);
      if (!ops.length) return;
      setShapes([...(selected.shapes || []), ...ops]);
      picker.clear();
      flash('info', `${cut ? 'Cut from' : 'Added to'} ${selected.name}’s mask (${KEY_UNDO} to undo).`);
    },
    [pick, selected, doc.width, doc.height, setShapes, picker, flash]
  );

  /** The selection as the mask of a new layer. */
  const pickToLayer = useCallback(
    (kind: 'still' | 'motion') => {
      if (!pick || pick.busy) return;
      const ops = pickOps(pick, false, doc.width, doc.height);
      if (!ops.length) return;
      const id = addLayer(kind);
      patchLayer(id, { shapes: ops }, true);
      picker.clear();
    },
    [pick, doc.width, doc.height, addLayer, patchLayer, picker]
  );

  const enterRoto = useCallback(
    (layerId: string, n: number) => {
      const l = docRef.current.layers.find((x) => x.id === layerId);
      setRoto({ layerId, n, frame: 0 });
      setRotoPrompts(l?.versions[n]?.track?.prompts || []);
      setTool('pick');
      setPlaying(false);
      picker.clear();
    },
    [picker]
  );
  const rotoFrame = useCallback(
    (f: number) => {
      setRoto((r) => r && { ...r, frame: Math.max(0, Math.min(rotoFrames - 1, Math.round(f))) });
      picker.clear();
    },
    [rotoFrames, picker]
  );
  /** Keep the pending pick as prompts on its frame. */
  const keepPick = useCallback(() => {
    if (!pick || pick.busy || !roto || pick.target.kind !== 'clip') return;
    const p: SegPrompt = { obj: 1, frame: pick.target.frame };
    if (pick.text) p.text = pick.text;
    else {
      if (pick.points.length) p.points = pick.points;
      if (pick.box) p.box = pick.box;
      // The reading chosen with Tab is the one tracked.
      if (pick.candidate > 0) p.candidate = pick.candidate;
    }
    setRotoPrompts((list) => [...list, p]);
    picker.clear();
  }, [pick, roto, picker]);
  const trackRoto = useCallback(() => {
    if (!roto || !rotoPrompts.length) return;
    serverAction(() => sceneApi.track(doc.id, roto.layerId, roto.n, rotoPrompts), 'Could not start tracking');
  }, [roto, rotoPrompts, doc.id, serverAction]);
  /**
   * Clean plate: an Edit layer just below a rotoscoped motion layer that removes its subject from the picture under
   * it (masked to where the subject is in the still: the matte's first frame, grown a little), so the subject can
   * move away without its still copy showing through.
   */
  const cleanPlate = useCallback(
    async (layerId: string, n: number) => {
      const s = docRef.current;
      const motion = s.layers.find((l) => l.id === layerId);
      const v = motion?.versions[n];
      if (!motion || !v) return;
      setBusy(true);
      try {
        const m = await sceneApi.matteFrame(s.id, layerId, n, 0);
        const r = v.region;
        const sx = r.w / m.width;
        const sy = r.h / m.height;
        const at = (q: Rect): Rect => ({ x: r.x + q.x * sx, y: r.y + q.y * sy, w: q.w * sx, h: q.h * sy });
        const what = (v.track?.prompts || []).find((p) => p.text)?.text || 'the person or thing in the middle';
        const layer: SceneLayer = {
          id: newId(),
          name: `Clean plate · ${motion.name}`,
          kind: 'still',
          region: { x: 0, y: 0, w: 0, h: 0 },
          mask: '',
          shapes: [],
          maskOpts: { expand: Math.round(Math.min(s.width, s.height) * 0.025) },
          feather: defaultFeather(s),
          prompt: `Remove ${what} from <image1> and fill where it was with what lies behind it, matching the surrounding background, light, texture and style exactly. Keep everything else as it is.`,
          options: { ...layerDefaults('still'), context: 'normal', cleanFor: layerId },
          versions: [],
          active: -1,
        };
        mutate((d) => {
          const at2 = d.layers.findIndex((l) => l.id === layerId);
          const layers = [...d.layers];
          layers.splice(Math.max(0, at2), 0, layer);
          return { ...d, layers };
        });
        // Through patchLayer: it waits for the matte piece to decode before rasterizing the mask.
        patchLayer(layer.id, { shapes: [pickedOp(m.mask, at(m.rect), at(m.rect), 'subject')] }, true);
        setSelectedId(layer.id);
        setRoto(null);
        setTool('select');
        flash('info', `Added a clean plate under ${motion.name}: check the prompt, then generate.`);
      } catch (e) {
        flash('error', `Could not make the clean plate: ${e instanceof Error ? e.message : e}`);
      } finally {
        setBusy(false);
      }
    },
    [mutate, patchLayer, flash]
  );

  const leaveRoto = useCallback(() => {
    setRoto(null);
    picker.clear();
    setTool('select');
    setPlaying(true);
  }, [picker]);

  const rotoView: RotoView | null =
    roto && rotoVersion
      ? (() => {
          const t = rotoVersion.track;
          let matte = '';
          if (showMatte && t?.state === 'completed' && t.dir) {
            const sep = t.dir.includes('\\') ? '\\' : '/';
            matte = `${t.dir}${sep}${String(roto.frame).padStart(5, '0')}.png`;
          }
          return { layerId: roto.layerId, version: rotoVersion, frame: roto.frame, fps: ROTO_FPS, matte };
        })()
      : null;

  // The keys of the Select tool and the roto, read through a ref so the key handler needn't re-register.
  const pickKeys = useRef<{ enter: (cut: boolean) => void; cycle: () => void; clear: () => void; pending: () => boolean; step: (d: number) => void; inRoto: () => boolean }>({
    enter: () => undefined,
    cycle: () => undefined,
    clear: () => undefined,
    pending: () => false,
    step: () => undefined,
    inRoto: () => false,
  });
  pickKeys.current = {
    enter: (cut: boolean) => (roto ? keepPick() : commitPick(cut)),
    cycle: picker.cycle,
    clear: picker.clear,
    pending: () => !!pick,
    step: (d: number) => {
      if (roto) rotoFrame(roto.frame + d);
    },
    inRoto: () => !!roto,
  };
  const toolRef = useRef(tool);
  toolRef.current = tool;

  // ---- keyboard -----------------------------------------------------------------------

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = isTyping(e.target);
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key === 'Enter') {
        e.preventDefault();
        generate();
        return;
      }
      if (typing) return;
      const pk = pickKeys.current;
      if (toolRef.current === 'pick') {
        if (e.key === 'Enter') {
          e.preventDefault();
          pk.enter(e.altKey);
          return;
        }
        if (e.key === 'Tab') {
          e.preventDefault();
          pk.cycle();
          return;
        }
        if (e.key === 'Escape' && pk.pending()) {
          pk.clear();
          return;
        }
      }
      if (pk.inRoto() && (e.key === ',' || e.key === '.')) {
        pk.step(e.key === ',' ? -1 : 1);
        return;
      }
      if (mod && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        if (e.shiftKey) stepHistory(redo, undo);
        else stepHistory(undo, redo);
        return;
      }
      if (mod && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        stepHistory(redo, undo);
        return;
      }
      if (mod && e.key === 's') {
        e.preventDefault();
        save().catch(() => undefined);
        return;
      }
      if (mod || e.altKey) return;
      const t = TOOLS.find((x) => x.key.toLowerCase() === e.key.toLowerCase());
      if (t) {
        setTool(t.id);
        return;
      }
      switch (e.key) {
        case '[':
          setBrush((b) => Math.max(2, Math.round(b / 1.25)));
          break;
        case ']':
          setBrush((b) => Math.min(800, Math.round(b * 1.25)));
          break;
        case 'm':
        case 'M':
          setMaskView((v) => (v === 'tint' ? 'matte' : v === 'matte' ? 'off' : 'tint'));
          break;
        case 'k':
        case 'K':
          setPlaying((v) => !v);
          break;
        case 'f':
        case 'F':
          setFitKey((n) => n + 1);
          break;
        case '\\':
          setCompare(true);
          break;
        case 'Escape':
          setTool('select');
          break;
        default:
      }
    };
    const onUp = (e: KeyboardEvent) => e.key === '\\' && setCompare(false);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onUp);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onUp);
    };
  }, [generate, stepHistory, save]);

  // ---- render -----------------------------------------------------------------------------

  const maskPanelFor = (l: SceneLayer) => (
    <MaskPanel
      layer={l}
      layers={doc.layers}
      paint={l.kind === 'paint'}
      selectedShape={selectedShape}
      onSelectShape={(i) => {
        setSelectedShape(i);
        if (i !== null && tool !== 'shape') setTool('shape');
      }}
      onShapes={setShapes}
      onMaskOpts={(patch) => {
        pushUndo(l, `opts:${Object.keys(patch).join(',')}`);
        const next = { ...(l.maskOpts || {}), ...patch };
        (Object.keys(next) as Array<keyof MaskOpts>).forEach((k) => next[k] === undefined && delete next[k]);
        patchLayer(l.id, { maskOpts: next }, true);
      }}
      onFeather={(feather) => patchLayer(l.id, { feather }, true)}
    />
  );

  const motionCount = doc.layers.filter((l) => l.kind === 'motion' && l.versions.length && !l.hidden).length;
  const autoLength = useMemo(() => {
    let d = 0;
    doc.layers.forEach((l) => {
      const v = l.versions[l.active];
      if (l.kind === 'motion' && v && !l.hidden) d = Math.max(d, v.duration || 0);
    });
    return d;
  }, [doc.layers]);
  const render = doc.render;
  const exporting = !!render && ['pending', 'in_progress', 'paused'].includes(render.state);
  const generating = doc.layers.reduce((n, l) => n + generationsInFlight(l), 0);

  return (
    <div className="sc-app">
      <header className="sc-header">
        <div className="sc-title">
          <input className="sc-scene-name" value={doc.name} onChange={(e) => mutate((s) => ({ ...s, name: e.target.value }))} aria-label="Scene name" />
          <span className="sc-faint" title={doc.source}>
            {basename(doc.source)}
            {doc.sourceTime ? ` @ ${doc.sourceTime.toFixed(2)}s` : ''} · {doc.width}×{doc.height}
          </span>
          <select
            className="sc-select"
            value={doc.id}
            title="Scenes made from this file"
            onFocus={() => sceneApi.list(doc.source).then(setOthers, () => undefined)}
            onChange={async (e) => {
              const v = e.target.value;
              if (v === doc.id) return;
              if (v === '__dream') {
                setDreamOpen(true);
                return;
              }
              try {
                await save();
                const next = v === '__new' ? await sceneApi.open(doc.source, doc.sourceTime || 0, true) : await sceneApi.get(v);
                onSwitch(next);
              } catch (err) {
                flash('error', `Could not open that scene: ${err instanceof Error ? err.message : err}`);
              }
            }}
          >
            {others.some((o) => o.id === doc.id) ? null : <option value={doc.id}>{doc.name}</option>}
            {others.map((o) => (
              <option key={o.id} value={o.id}>
                {o.name} · {o.layers} layer{o.layers === 1 ? '' : 's'}
              </option>
            ))}
            <option value="__new">＋ New scene from this file</option>
            <option value="__dream">＋ New scene from a prompt…</option>
          </select>
        </div>
        <div className="sc-header-mid">
          <button type="button" className={`sc-btn${playing ? '' : ' on'}`} onClick={() => setPlaying((v) => !v)} title="Play / pause the motion layers (K)">
            {playing ? '❚❚ Pause' : '▶ Play'}
          </button>
          <button type="button" className="sc-btn" onClick={() => setSyncKey((n) => n + 1)} title="Restart every motion layer together">
            ⟲ Restart
          </button>
          <button type="button" className={`sc-btn${soundOn ? ' on' : ''}`} onClick={() => setSoundOn((v) => !v)} title="Hear the layers that generate sound">
            {soundOn ? '🔊 Sound' : '🔇 Muted'}
          </button>
          <button
            type="button"
            className={`sc-btn${compare ? ' on' : ''}`}
            onPointerDown={() => setCompare(true)}
            onPointerUp={() => setCompare(false)}
            onPointerLeave={() => setCompare(false)}
            title="Hold to see the original (or hold \)"
          >
            Compare
          </button>
          <button
            type="button"
            className={`sc-btn${maskView !== 'off' ? ' on' : ''}`}
            onClick={() => setMaskView((v) => (v === 'tint' ? 'matte' : v === 'matte' ? 'off' : 'tint'))}
            title="The selected layer's mask: tinted over the picture, as a black-and-white matte, or hidden (M)"
          >
            {maskView === 'matte' ? 'Mask: matte' : maskView === 'off' ? 'Mask: off' : 'Mask: tint'}
          </button>
          <button
            type="button"
            className={`sc-btn${doc.freeLoops ? '' : ' on'}`}
            onClick={() => mutate((s) => ({ ...s, freeLoops: !s.freeLoops }))}
            title={
              doc.freeLoops
                ? 'Loops: free. Every motion layer loops at its own length (the export may jump where they do). Click to sync them.'
                : 'Loops: synced. Every motion layer plays at the speed that fits a whole number of its loops into the scene length, so the export loops as one. Click to let them loop freely.'
            }
          >
            {doc.freeLoops ? 'Loops: free' : 'Loops: synced'}
          </button>
        </div>
        <div className="sc-header-right">
          <label className="sc-length" title="Length of the exported video; auto = the longest motion layer">
            Length
            <input
              className="sc-input"
              inputMode="decimal"
              placeholder={autoLength ? `auto ${autoLength.toFixed(1)}s` : 'auto'}
              value={doc.duration ? String(doc.duration) : ''}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                mutate((s) => ({ ...s, duration: Number.isFinite(v) && v > 0 ? Math.min(60, v) : 0 }));
              }}
            />
          </label>
          <button type="button" className="sc-btn" disabled={busy || exporting} onClick={() => exportAs('still')} title="Flatten the still edits to a PNG next to the original">
            Export still
          </button>
          <button
            type="button"
            className="sc-btn sc-btn-primary"
            disabled={busy || exporting}
            onClick={() => exportAs('video')}
            title={motionCount ? 'Render the living scene to an mp4 next to the original' : 'No motion yet: the video will be the still scene'}
          >
            {exporting ? `Exporting${render && render.total ? ` ${Math.round((render.done / render.total) * 100)}%` : '…'}` : 'Export video'}
          </button>
          {render?.state === 'completed' && render.output && isElectron && (
            <button type="button" className="sc-btn sc-btn-quiet" onClick={() => send('show-item-in-folder', [render.output])} title={render.output}>
              Show file
            </button>
          )}
        </div>
      </header>

      <div className="sc-body">
        <nav className="sc-tools" aria-label="Tools">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tool === t.id ? 'on' : ''}
              title={`${t.label} (${t.key}) · ${t.title}`}
              onClick={() => setTool(t.id)}
              disabled={t.id === 'protect' && selected?.kind === 'paint'}
            >
              <ToolIcon id={t.id} />
            </button>
          ))}
          <div className="sc-tools-sep" />
          <div className="sc-brush-size" title="Brush size ([ / ])">
            <input type="range" min={2} max={400} value={brush} onChange={(e) => setBrush(Number(e.target.value))} aria-label="Brush size" />
            <span>{brush}</span>
          </div>
          {selected?.kind === 'paint' && (
            <div className="sc-ink" aria-label="Paint colour">
              <label className="sc-ink-current" title="Paint colour" style={{ background: ink }}>
                <input type="color" value={ink} onChange={(e) => setInk(e.target.value)} aria-label="Paint colour" />
              </label>
              {EyeDropperApi && (
                <button
                  type="button"
                  className="sc-ink-pick"
                  title="Pick a colour from the screen"
                  onClick={async () => {
                    try {
                      const r = await new EyeDropperApi().open();
                      if (r?.sRGBHex) rememberInk(r.sRGBHex);
                    } catch {
                      // cancelled
                    }
                  }}
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
                    <path d="M10.5 2.5l3 3-1.5 1.5-3-3zM9.5 4.5l2 2L5 13H3v-2z" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                  </svg>
                </button>
              )}
              <div className="sc-ink-swatches">
                {swatches.map((c) => (
                  <button key={c} type="button" className={c.toLowerCase() === ink.toLowerCase() ? 'on' : ''} style={{ background: c }} title={c} onClick={() => setInk(c)} aria-label={`Use ${c}`} />
                ))}
              </div>
              <div className="sc-brush-size" title="Paint opacity">
                <input type="range" min={5} max={100} value={Math.round(inkAlpha * 100)} onChange={(e) => setInkAlpha(Number(e.target.value) / 100)} aria-label="Paint opacity" />
                <span>{Math.round(inkAlpha * 100)}%</span>
              </div>
            </div>
          )}
          <div className="sc-brush-size" title="Brush hardness: 100 = a hard edge, lower = a soft one">
            <input type="range" min={0} max={100} value={Math.round(hardness * 100)} onChange={(e) => setHardness(Number(e.target.value) / 100)} aria-label="Brush hardness" />
            <span>{Math.round(hardness * 100)}</span>
          </div>
        </nav>

        <main className="sc-main">
          <Stage
            scene={doc}
            selected={selected}
            tool={tool}
            brush={brush}
            hardness={hardness}
            ink={selected?.kind === 'paint' ? ink : null}
            maskView={maskView}
            selectedShape={selectedShape}
            onSelectShape={setSelectedShape}
            onReplaceOp={(i, op) => {
              if (!selected) return;
              setShapes((selected.shapes || []).map((x, j) => (j === i ? op : x)));
            }}
            pick={tool === 'pick' ? pick : null}
            pickArea={roto && rotoVersion ? rotoVersion.region : null}
            onPickClick={onPickClick}
            onPickBox={onPickBox}
            roto={rotoView}
            compare={compare}
            playing={playing}
            soundOn={soundOn}
            syncKey={syncKey}
            fitKey={fitKey}
            protectMask={protectMask}
            motionOverlay={motionView && selected && motionView.layerId === selected.id && motionView.n === selected.active ? motionView : null}
            onCommit={commitOp}
            onSelect={setSelectedId}
          />
          {tool === 'pick' && (
            <PickBar
              pick={pick}
              roto={!!roto}
              hasLayer={!!selected && selected.kind !== 'paint'}
              onFind={(text) => picker.find(pickTarget(), text)}
              onPreset={(which) => picker.preset(pickTarget(), which)}
              onCycle={picker.cycle}
              onToggle={picker.toggle}
              onAdd={commitPick}
              onNewLayer={pickToLayer}
              onKeep={keepPick}
              onClear={picker.clear}
            />
          )}
          {roto && rotoVersion && (
            <RotoBar
              frame={roto.frame}
              frames={rotoFrames}
              fps={ROTO_FPS}
              prompts={rotoPrompts}
              track={rotoVersion.track}
              showMatte={showMatte}
              onFrame={rotoFrame}
              onRemovePrompt={(i) => setRotoPrompts((list) => list.filter((_, j) => j !== i))}
              onTrack={trackRoto}
              onShowMatte={setShowMatte}
              onDone={leaveRoto}
            />
          )}
          {notice && (
            <div className={`sc-notice ${notice.tone}`} role="status">
              {notice.text}
              <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss">
                ×
              </button>
            </div>
          )}
        </main>

        <aside className="sc-panel">
          <LayerList
            layers={doc.layers}
            selectedId={selectedId}
            baseName={basename(doc.source)}
            onSelect={setSelectedId}
            onAdd={addLayer}
            onToggleHidden={(id) => {
              const l = docRef.current.layers.find((x) => x.id === id);
              if (l) patchLayer(id, { hidden: !l.hidden });
            }}
            onMove={moveLayer}
            onMoveTo={moveLayerTo}
            stale={new Set(doc.layers.filter((l) => versionStale(doc, l)).map((l) => l.id))}
          />
          {selected && selected.kind === 'paint' ? (
            <PaintPanel
              key={selected.id}
              layer={selected}
              maskPanel={maskPanelFor(selected)}
              usedBy={doc.layers.flatMap((l) => {
                const refs = l.options?.refs || [];
                const i = refs.findIndex((r) => r.layer === selected.id);
                return i < 0 ? [] : [{ id: l.id, name: l.name || l.kind, token: refTokens(l.kind, refs, hasVoice(l.options))[i] }];
              })}
              onPatch={(p) => patchLayer(selected.id, p)}
              onOptions={(o) => {
                const next = { ...(selected.options || {}), ...o };
                patchLayer(selected.id, { options: next });
                rememberLayerDefaults('paint', next);
              }}
              onFinish={() => finishPaint(selected)}
              onSelectLayer={setSelectedId}
              onDelete={() => deleteLayer(selected.id)}
            />
          ) : selected ? (
            <Inspector
              key={selected.id}
              layer={selected}
              busy={busy}
              stale={versionStale(doc, selected)}
              maskPanel={maskPanelFor(selected)}
              paintLayers={doc.layers.filter((l) => l.kind === 'paint')}
              onPatch={(p) => patchLayer(selected.id, p, 'feather' in p || 'kind' in p)}
              onOptions={(o) => {
                const next = { ...(selected.options || {}), ...o };
                patchLayer(selected.id, { options: next }, 'context' in o);
                rememberLayerDefaults(selected.kind, next);
              }}
              onGenerate={generate}
              onCancel={(job) => serverAction(() => sceneApi.cancel(doc.id, selected.id, job), 'Could not cancel')}
              onDelete={() => deleteLayer(selected.id)}
              onClearMask={() => setShapes((selected.shapes || []).filter((op) => op.keep))}
              onClearProtect={() => setShapes((selected.shapes || []).filter((op) => !op.keep))}
              onDeleteVersion={(n) => serverAction(() => sceneApi.deleteVersion(doc.id, selected.id, n), 'Could not discard the version')}
              onKeepOnly={(n) => serverAction(() => sceneApi.keepOnly(doc.id, selected.id, n), 'Could not discard the other versions')}
              onLoop={(n, kind) => loopVersion(selected.id, n, kind)}
              onRefinish={(n) => serverAction(() => sceneApi.refinish(doc.id, selected.id, n), 'Could not re-finish the version')}
              motionShown={!!motionView && motionView.layerId === selected.id && motionView.n === selected.active}
              onShowMotion={() => toggleMotion(selected)}
              onGrowToMotion={() => growToMotion(selected)}
              loopSpeed={layerRate(doc, selected)}
              onChooseVoice={chooseVoice}
              onPickLibrary={(kinds) => requestLibraryPick(kinds, `${selected.name || 'a layer'} · ${doc.name}`)}
              onPickFiles={pickFiles}
              onRoto={(n) => enterRoto(selected.id, n)}
              rotoActive={!!roto && roto.layerId === selected.id}
              onUntrack={(n) => serverAction(() => sceneApi.untrack(doc.id, selected.id, n), 'Could not remove the roto')}
              onCleanPlate={(n) => cleanPlate(selected.id, n)}
            />
          ) : (
            <section className="sc-section sc-help">
              <h3>Build the scene a region at a time</h3>
              <ol>
                <li>
                  <strong>Edit</strong> layers change how an area looks: box or paint it, describe the change, generate.
                </li>
                <li>
                  <strong>Motion</strong> layers bring an area to life: paint it, say how it moves, animate.
                </li>
                <li>Each layer is made from the picture below it, so stack edits first and motion on top.</li>
                <li>Refine a mask after generating: the result is re-blended live. Pick between versions any time.</li>
                <li>
                  <strong>Export video</strong> renders every layer looping together, next to the original.
                </li>
              </ol>
            </section>
          )}
        </aside>
      </div>

      {dreamOpen && (
        <DreamDialog
          onClose={() => setDreamOpen(false)}
          onScene={async (next) => {
            setDreamOpen(false);
            try {
              await save();
            } catch {
              // the scene being left keeps its unsaved state for next time
            }
            onSwitch(next);
          }}
        />
      )}

      <footer className="sc-footer">
        <span>
          {saveState === 'saving' ? 'Saving…' : saveState === 'dirty' ? 'Unsaved changes' : saveState === 'error' ? 'Not saved (retrying on the next change)' : 'Saved'}
        </span>
        {generating > 0 && <span>{generating} generating</span>}
        <span className="sc-keys">V select · B box · P brush · L lasso · N pen · E erase · S protect · T shape · [ ] size · Alt subtract · Space pan · F fit · \ compare · K play · {KEY_UNDO} undo · {KEY_GENERATE} generate</span>
      </footer>
    </div>
  );
}
