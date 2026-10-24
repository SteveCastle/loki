import React, { useState } from 'react';
import { POSE_PRESETS } from './pose';
import { FINISH_MODES, type FinishMode, type LayerOptions, type SceneLayer } from './scene-api';

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
  /** Add an Edit layer above that finishes this painting in a mode (masked to it, prompted for it). */
  onFinish: (mode: FinishMode) => void;
  /** Whether the Mannequin tool is the current tool. */
  poseTool: boolean;
  onPoseTool: () => void;
  /** Turn every figure on the layer to face the other way. */
  onMirrorPose: () => void;
  /** A ready pose (POSE_PRESETS id) for the last figure, or a new one. */
  onPosePreset: (id: string) => void;
  /** Start a figure from the selected character's detected pose. */
  onDetectPose: () => void;
  /** Whether a character is selected (Select object) to detect a pose from. */
  hasPick: boolean;
  onSelectLayer: (id: string) => void;
  onDelete: () => void;
}

export default function PaintPanel(props: PaintPanelProps) {
  const { layer: l, usedBy } = props;
  const guide = !!l.options?.guide;
  const hasPose = (l.shapes || []).some((op) => op.t === 'pose' && !op.off);
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

      <div className="sc-pose-row">
        <button
          type="button"
          className={`sc-btn${props.poseTool ? ' on' : ''}`}
          onClick={props.onPoseTool}
          title="Mannequin (M): place a figure and drag its joints into a pose. It is drawn as an OpenPose skeleton, the pose format image models read best; Finish → Repose gives it to the engine on black"
        >
          Mannequin
        </button>
        <button
          type="button"
          className={`sc-btn${props.hasPick ? ' on' : ''}`}
          onClick={props.onDetectPose}
          title="Start a figure from the pose of a character in the picture: select them first with Select object (Q), or Ctrl+drag a box round them with the Mannequin tool"
        >
          Detect pose
        </button>
        <select
          className="sc-select"
          value=""
          onChange={(e) => e.target.value && props.onPosePreset(e.target.value)}
          aria-label="Preset pose"
          title={hasPose ? 'Put the figure in a ready pose (it keeps its place and size)' : 'Add a figure in a ready pose'}
        >
          <option value="">Preset…</option>
          {POSE_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        {hasPose && (
          <button type="button" className="sc-btn sc-btn-quiet" onClick={props.onMirrorPose} title="Turn the figure to face the other way">
            Mirror
          </button>
        )}
        <span className="sc-faint">
          {props.poseTool
            ? 'Click or drag head to feet to add a figure, Ctrl+drag round a character to copy their pose; drag joints to pose it.'
            : 'Pose a figure for Repose.'}
        </span>
      </div>

      {props.maskPanel}

      <div className="sc-finish">
        <span className="sc-label">Finish with AI</span>
        {l.mask ? (
          <div className="sc-finish-modes">
            {(Object.keys(FINISH_MODES) as FinishMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className="sc-finish-mode"
                onClick={() => props.onFinish(mode)}
                title={`Adds an Edit layer above, masked to what you painted.${
                  FINISH_MODES[mode].guide === guide ? '' : FINISH_MODES[mode].guide ? ' This paint becomes a guide only.' : ' This paint goes back into the picture.'
                }`}
              >
                <strong>{FINISH_MODES[mode].label}</strong>
                <span>{FINISH_MODES[mode].help}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="sc-faint">Paint something first.</p>
        )}
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
