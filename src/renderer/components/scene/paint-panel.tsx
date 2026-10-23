import React, { useState } from 'react';
import type { LayerOptions, SceneLayer } from './scene-api';

// The inspector of a Paint layer: the user's own colour painting. In the
// picture it is part of what every layer above is made from (rough colour an
// Edit layer finishes, img2img style) and of the export; as a guide it stays
// out of the picture and is given to Edit / Motion layers as a reference,
// painted over the picture they are made from and lined up with their crop
// (colourise line art: <image1> the clean lines, <image2> the rough colour).

export interface PaintPanelProps {
  layer: SceneLayer;
  /** The Strokes section (mask-panel.tsx in paint mode). */
  maskPanel: React.ReactNode;
  /** Layers that reference this one, with the token their prompt uses for it. */
  usedBy: Array<{ id: string; name: string; token: string | null }>;
  onPatch: (patch: Partial<SceneLayer>) => void;
  onOptions: (patch: LayerOptions) => void;
  /** Add an Edit layer above that finishes this painting (masked to it, prompted for it). */
  onFinish: () => void;
  onSelectLayer: (id: string) => void;
  onDelete: () => void;
}

export default function PaintPanel(props: PaintPanelProps) {
  const { layer: l, usedBy } = props;
  const guide = !!l.options?.guide;
  const [confirmDelete, setConfirmDelete] = useState(false);
  return (
    <section className="sc-section sc-inspector">
      <div className="sc-field-row">
        <input className="sc-name" value={l.name} placeholder="Paint" onChange={(e) => props.onPatch({ name: e.target.value })} aria-label="Layer name" />
      </div>
      <div className="sc-grid">
        <span className="sc-label">Use</span>
        <div className="sc-seg" role="radiogroup">
          <button
            type="button"
            role="radio"
            aria-checked={!guide}
            className={guide ? '' : 'on'}
            onClick={() => props.onOptions({ guide: false })}
            title="Part of the picture: the layers above are made from it, and it is exported"
          >
            In the picture
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={guide}
            className={guide ? 'on' : ''}
            onClick={() => props.onOptions({ guide: true })}
            title="Only a reference for Edit and Motion layers: shown while you work, never in what they are made from or in the export"
          >
            Guide only
          </button>
        </div>
      </div>
      <p className="sc-faint sc-paint-help">
        {guide
          ? 'A guide the AI follows: add it to an Edit or Motion layer’s References and name it in the prompt (e.g. “colour <image1> like <image2>”).'
          : 'Part of the picture: Edit and Motion layers above it are made from the picture with this paint on it.'}{' '}
        Paint loosely with the Brush, Box, Lasso or Pen: the AI corrects shapes and fills whole areas, so rough is fine.
      </p>

      {props.maskPanel}

      <div className="sc-generate">
        <button
          type="button"
          className="sc-btn sc-btn-primary sc-btn-wide"
          disabled={!l.mask}
          onClick={props.onFinish}
          title={
            guide
              ? 'Adds an Edit layer above, masked to what you painted, that colours the picture the way your paint shows (this layer as its <image2>)'
              : 'Adds an Edit layer above, masked to what you painted, that turns the rough paint into finished colour'
          }
        >
          {l.mask ? 'Finish with AI' : 'Paint something first'}
        </button>
      </div>

      {usedBy.length > 0 && (
        <div className="sc-paint-used">
          <span className="sc-label">Used by</span>
          {usedBy.map((u) => (
            <button key={u.id} type="button" className="sc-chip" onClick={() => props.onSelectLayer(u.id)} title="Select that layer">
              {u.name}
              {u.token ? ` · ${u.token}` : ''}
            </button>
          ))}
        </div>
      )}

      <div className="sc-layer-foot">
        <span />
        {confirmDelete ? (
          <span className="sc-confirm">
            Delete layer?
            <button type="button" className="sc-btn sc-btn-danger" onClick={props.onDelete}>
              Delete
            </button>
            <button type="button" className="sc-btn sc-btn-quiet" onClick={() => setConfirmDelete(false)}>
              Keep
            </button>
          </span>
        ) : (
          <button type="button" className="sc-btn sc-btn-quiet" onClick={() => setConfirmDelete(true)}>
            Delete layer
          </button>
        )}
      </div>
    </section>
  );
}
