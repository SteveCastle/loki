import React from 'react';
import { opLabel, type MaskOp, type MaskOpts } from './mask';
import type { SceneLayer } from './scene-api';

// The selected layer's Mask section: every shape the mask is drawn with, in
// drawing order (each can be selected for the Shape tool, switched between
// adding and cutting, turned off or removed), other layers' masks linked in
// live (how overlapping layers are kept apart: the sky minus the people in
// front), and the whole-mask settings applied after the shapes: expand or
// choke, invert, soft edge, opacity.

export interface MaskPanelProps {
  layer: SceneLayer;
  /** Every layer of the scene, bottom to top. */
  layers: SceneLayer[];
  selectedShape: number | null;
  onSelectShape: (i: number | null) => void;
  /** Replace the layer's ops (one undo step). */
  onShapes: (ops: MaskOp[]) => void;
  /** Change whole-mask settings (a slider drag is one undo step). */
  onMaskOpts: (patch: MaskOpts) => void;
  onFeather: (px: number) => void;
  /** A paint layer: strokes are paint (each in its colour) or erase; no links, expand or invert. */
  paint?: boolean;
}

function OpIcon({ op }: { op: MaskOp }) {
  const p = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (op.t) {
    case 'rect':
      return (
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
          <rect x="2.5" y="3.5" width="11" height="9" {...p} />
        </svg>
      );
    case 'lasso':
      return (
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
          <path d="M3 12L5 3.5l8 2.5-2 7z" {...p} />
        </svg>
      );
    case 'stroke':
      return (
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
          <path d="M2.5 11c2-5 4 1 6-3s3-3 5-4" {...p} strokeWidth={2.6} />
        </svg>
      );
    case 'layer':
      return (
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
          <path d="M8 2.5l5.5 3L8 8.5 2.5 5.5zM2.5 8.5L8 11.5l5.5-3" {...p} />
        </svg>
      );
    default:
      return (
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
          <path d="M3 13c0-4 3-3 4-6s4-4 6-4" {...p} />
          <circle cx="11" cy="10" r="2" {...p} />
        </svg>
      );
  }
}

function modeOf(op: MaskOp): 'add' | 'cut' | 'protect' {
  if (op.keep) return 'protect';
  return op.erase ? 'cut' : 'add';
}

export default function MaskPanel(props: MaskPanelProps) {
  const { layer: l, layers, selectedShape, paint } = props;
  const ops = l.shapes || [];
  const mo = l.maskOpts || {};
  const nameOf = (id: string) => {
    const x = layers.find((y) => y.id === id);
    return x ? x.name || (x.kind === 'motion' ? 'Motion' : 'Edit') : null;
  };
  const myIndex = layers.findIndex((x) => x.id === l.id);
  const others = layers.filter((x) => x.id !== l.id);
  const linked = new Set(ops.filter((op) => op.t === 'layer').map((op) => (op as { id: string }).id));
  // Layers drawn over this one whose masks this one doesn't cut yet.
  const inFront = layers.slice(myIndex + 1).filter((x) => !x.hidden && (x.shapes || []).length > 0 && !linked.has(x.id));

  const setOp = (i: number, patch: Partial<MaskOp>) => props.onShapes(ops.map((op, j) => (j === i ? ({ ...op, ...patch } as MaskOp) : op)));
  const removeOp = (i: number) => {
    props.onShapes(ops.filter((_, j) => j !== i));
    if (selectedShape !== null) props.onSelectShape(selectedShape === i ? null : selectedShape > i ? selectedShape - 1 : selectedShape);
  };
  const moveOp = (i: number, dir: 1 | -1) => {
    const j = i + dir;
    if (j < 0 || j >= ops.length) return;
    const next = ops.slice();
    [next[i], next[j]] = [next[j], next[i]];
    props.onShapes(next);
    if (selectedShape === i) props.onSelectShape(j);
  };
  const cycleMode = (i: number) => {
    const op = ops[i];
    if (op.keep) return;
    setOp(i, { erase: op.erase ? undefined : true });
  };

  // Newest first: the shape drawn last is the one you most often want to fix.
  const rows = ops.map((op, i) => ({ op, i })).reverse();

  return (
    <div className="sc-maskpanel">
      <div className="sc-maskpanel-head">
        <span className="sc-label">{paint ? 'Strokes' : 'Mask'}</span>
        <span className="sc-faint">{ops.length ? `${ops.length} ${paint ? 'stroke' : 'shape'}${ops.length === 1 ? '' : 's'}` : 'nothing drawn yet'}</span>
      </div>

      {rows.length > 0 && (
        <ol className="sc-shapes" aria-label="Mask shapes, newest first">
          {rows.map(({ op, i }) => {
            const mode = modeOf(op);
            return (
              <li key={i} className={`sc-shape${selectedShape === i ? ' selected' : ''}${op.off ? ' off' : ''} ${mode}`}>
                <button
                  type="button"
                  className="sc-shape-main"
                  onClick={() => props.onSelectShape(selectedShape === i ? null : i)}
                  title={op.t === 'layer' ? 'Linked live to that layer’s mask' : 'Select it, then use the Shape tool (T) to move, scale or reshape it'}
                >
                  {paint && !op.erase ? <span className="sc-swatch-dot" style={{ background: op.c || '#fff', opacity: op.a ?? 1 }} /> : <OpIcon op={op} />}
                  <span className="sc-shape-name">{opLabel(op, nameOf)}</span>
                </button>
                <button
                  type="button"
                  className={`sc-shape-mode ${mode}`}
                  disabled={mode === 'protect'}
                  onClick={() => cycleMode(i)}
                  title={mode === 'protect' ? 'Protect: the layer leaves this exactly as the picture below' : mode === 'cut' ? 'Cuts the mask: click to add instead' : 'Adds to the mask: click to cut instead'}
                >
                  {mode === 'protect' ? 'protect' : paint ? (mode === 'cut' ? '− erase' : '+ paint') : mode === 'cut' ? '− cut' : '+ add'}
                </button>
                <span className="sc-row-tools">
                  <button type="button" title="Earlier (drawn before the shapes below it in the list)" disabled={i === 0} onClick={() => moveOp(i, -1)}>
                    ▼
                  </button>
                  <button type="button" title="Later (drawn after, so it wins)" disabled={i === ops.length - 1} onClick={() => moveOp(i, 1)}>
                    ▲
                  </button>
                  <button type="button" title={op.off ? 'Turn on' : 'Turn off (kept, not used)'} onClick={() => setOp(i, { off: op.off ? undefined : true })}>
                    {op.off ? '○' : '●'}
                  </button>
                  <button type="button" title="Remove" onClick={() => removeOp(i)}>
                    ×
                  </button>
                </span>
              </li>
            );
          })}
        </ol>
      )}

      {!paint && others.length > 0 && (
        <div className="sc-mask-links">
          <select
            className="sc-select"
            value=""
            onChange={(e) => {
              const [how, id] = e.target.value.split(':');
              if (!id) return;
              props.onShapes([...ops, how === 'cut' ? { t: 'layer', id, erase: true } : { t: 'layer', id }]);
            }}
            title="Use another layer's mask in this one, live: it follows every change to that layer"
            aria-label="Link another layer's mask"
          >
            <option value="">Link a layer’s mask…</option>
            <optgroup label="Cut it out of this mask">
              {others.map((x) => (
                <option key={`c${x.id}`} value={`cut:${x.id}`}>
                  − {x.name || x.kind}
                </option>
              ))}
            </optgroup>
            <optgroup label="Add it to this mask">
              {others.map((x) => (
                <option key={`a${x.id}`} value={`add:${x.id}`}>
                  + {x.name || x.kind}
                </option>
              ))}
            </optgroup>
          </select>
          {inFront.length > 0 && (
            <button
              type="button"
              className="sc-btn sc-btn-tiny"
              onClick={() => props.onShapes([...ops, ...inFront.map((x): MaskOp => ({ t: 'layer', id: x.id, erase: true }))])}
              title={`Cut ${inFront.map((x) => x.name).join(', ')} out of this mask, live, so the layers in front never overlap it`}
            >
              Cut layers in front ({inFront.length})
            </button>
          )}
        </div>
      )}

      <div className="sc-grid sc-mask-grid">
        <span className="sc-label" title={paint ? 'Blur the painting' : 'How softly the area blends into the picture around it'}>
          {paint ? 'Soften' : 'Soft edge'}
        </span>
        <div className="sc-slider">
          <input type="range" min={0} max={160} step={1} value={l.feather} onChange={(e) => props.onFeather(Number(e.target.value))} aria-label="Soft edge" />
          <span>{Math.round(l.feather)} px</span>
        </div>
{!paint && (
<>
        <span className="sc-label" title="Grow the mask outward (+) or choke it inward (−), before the soft edge">
          Expand
        </span>
        <div className="sc-slider">
          <input type="range" min={-120} max={120} step={1} value={mo.expand || 0} onChange={(e) => props.onMaskOpts({ expand: Number(e.target.value) || undefined })} aria-label="Expand" />
          <span>
            {(mo.expand || 0) > 0 ? '+' : ''}
            {mo.expand || 0} px
          </span>
        </div>
</>
)}
        <span className="sc-label" title="How strongly the layer shows through its mask">
          Opacity
        </span>
        <div className="sc-slider">
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round((mo.opacity ?? 1) * 100)}
            onChange={(e) => {
              const v = Number(e.target.value) / 100;
              props.onMaskOpts({ opacity: v >= 1 ? undefined : v });
            }}
            aria-label="Opacity"
          />
          <span>{Math.round((mo.opacity ?? 1) * 100)}%</span>
        </div>
{!paint && (
<>
        <span className="sc-label">Invert</span>
        <div className="sc-seg" role="radiogroup">
          <button type="button" role="radio" aria-checked={!mo.invert} className={mo.invert ? '' : 'on'} onClick={() => props.onMaskOpts({ invert: undefined })}>
            Inside
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={!!mo.invert}
            className={mo.invert ? 'on' : ''}
            onClick={() => props.onMaskOpts({ invert: true })}
            title="The layer covers everything except what the shapes cover"
          >
            Outside
          </button>
        </div>
</>
)}
      </div>
    </div>
  );
}
