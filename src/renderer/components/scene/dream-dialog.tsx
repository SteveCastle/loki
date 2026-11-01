import React, { useEffect, useRef, useState } from 'react';
import { sceneApi, type DreamStatus, type Scene } from './scene-api';
import ModelPicker from '../dream/model-picker';
import { applyPromptGuide, dreamPromptGuide, effectiveDreamModel, useDreamModels } from '../dream/dream-models';

// "New scene from a prompt": a picture made by loki-dream (Kromoss, or any
// model the server lists) becomes the base of a new scene. The prompt and the
// negative start as a sample written for the chosen model. The picture lands
// in the library (its Generated
// folder) like any other, and the scene is started from it.

const SHAPES: Array<{ id: string; label: string; size: string }> = [
  { id: 'square', label: 'Square', size: '1024x1024' },
  { id: 'landscape', label: 'Landscape', size: '1344x768' },
  { id: 'portrait', label: 'Portrait', size: '768x1344' },
];

export interface DreamDialogProps {
  onClose: () => void;
  /** The new scene, started from the generated picture. */
  onScene: (s: Scene) => void;
}

export default function DreamDialog({ onClose, onScene }: DreamDialogProps) {
  const [prompt, setPrompt] = useState('');
  const [shape, setShape] = useState('landscape');
  const [turbo, setTurbo] = useState(false);
  const [model, setModel] = useState('');
  const models = useDreamModels(true);
  const chosen = effectiveDreamModel(models.catalog, model);
  // Turbo is MOSS-Image's distillation LoRA; other models run their own recipe (unknown = the default: offer it).
  const canTurbo = !chosen || chosen.turbo;
  // The run uses guidance (a negative prompt does something): not the turbo LoRA, not a distilled checkpoint.
  const guided = chosen ? !chosen.distilled && !(chosen.turbo && turbo) : !turbo;
  const [negative, setNegative] = useState('');
  // The model's sample prompt / negative, while the boxes are empty or hold another model's sample.
  const arch = chosen?.arch || (models.catalog.loaded ? 'moss' : '');
  useEffect(() => {
    if (!arch) return;
    const g = applyPromptGuide({ prompt, negative }, dreamPromptGuide(arch), true);
    if (g.prompt !== undefined) setPrompt(g.prompt);
    if (g.negative !== undefined) setNegative(g.negative);
    // only when the architecture changes
  }, [arch]);
  const [status, setStatus] = useState<DreamStatus | null>(null);
  const [error, setError] = useState('');
  const alive = useRef(true);
  useEffect(
    () => () => {
      alive.current = false;
    },
    []
  );

  const running = !!status && !['completed', 'error', 'cancelled'].includes(status.state);

  const start = async () => {
    setError('');
    try {
      let st = await sceneApi.dream({
        prompt,
        negative: guided ? negative.trim() : undefined,
        size: SHAPES.find((x) => x.id === shape)?.size,
        turbo: turbo && canTurbo,
        model,
      });
      setStatus(st);
      while (alive.current && !['completed', 'error', 'cancelled'].includes(st.state)) {
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => window.setTimeout(r, 1500));
        // eslint-disable-next-line no-await-in-loop
        st = await sceneApi.dreamStatus(st.jobId);
        if (alive.current) setStatus(st);
      }
      if (!alive.current) return;
      if (st.state !== 'completed' || !st.output) {
        setError(st.error || `The picture was not made (${st.state}).`);
        return;
      }
      onScene(await sceneApi.open(st.output, 0, true));
    } catch (e) {
      if (alive.current) setError(e instanceof Error ? e.message : String(e));
    }
  };

  const pct = status && status.total > 0 ? Math.round((status.done / status.total) * 100) : 0;
  return (
    <div className="sc-modal-back" role="presentation" onMouseDown={(e) => e.target === e.currentTarget && !running && onClose()}>
      <div className="sc-modal" role="dialog" aria-label="New scene from a prompt">
        <h3>New scene from a prompt</h3>
        <p className="sc-faint">{chosen?.label || 'Kromoss'} draws a new picture (it goes to the library’s Generated folder) and a scene starts from it.</p>
        <textarea
          className="sc-prompt"
          rows={4}
          autoFocus
          value={prompt}
          placeholder="Describe the picture: a quiet harbour town at dawn, fishing boats, mist over the water, soft pastel light…"
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && prompt.trim() && !running) start();
          }}
          disabled={running}
        />
        <div className="sc-grid">
          <span className="sc-label">Model</span>
          <ModelPicker catalog={models.catalog} value={model} onRefresh={models.refresh} onChange={(file) => setModel(file)} />
          {guided ? <span className="sc-label">Avoid</span> : null}
          {guided ? (
            <input className="sc-input" value={negative} disabled={running} placeholder="what the picture should not have" aria-label="Negative prompt" onChange={(e) => setNegative(e.target.value)} />
          ) : null}
          <span className="sc-label">Shape</span>
          <div className="sc-seg" role="radiogroup">
            {SHAPES.map((x) => (
              <button key={x.id} type="button" role="radio" aria-checked={shape === x.id} className={shape === x.id ? 'on' : ''} disabled={running} onClick={() => setShape(x.id)} title={x.size.replace('x', ' × ')}>
                {x.label}
              </button>
            ))}
          </div>
          {canTurbo ? <span className="sc-label">Speed</span> : null}
          {canTurbo ? (
            <div className="sc-seg" role="radiogroup">
              <button type="button" role="radio" aria-checked={!turbo} className={turbo ? '' : 'on'} disabled={running} onClick={() => setTurbo(false)}>
                Best
              </button>
              <button type="button" role="radio" aria-checked={turbo} className={turbo ? 'on' : ''} disabled={running} onClick={() => setTurbo(true)} title="8 steps with the turbo LoRA: several times faster">
                Turbo
              </button>
            </div>
          ) : null}
        </div>
        {running && (
          <div className="sc-generate">
            <div className="sc-progress">
              <div className="sc-progress-bar" style={{ width: `${Math.max(3, pct)}%` }} />
            </div>
            <div className="sc-job-line">{status?.state === 'pending' ? 'Waiting for the GPU…' : pct === 0 ? 'Loading the model…' : `Drawing… ${pct}%`}</div>
          </div>
        )}
        {error && <div className="sc-error">{error}</div>}
        <div className="sc-modal-actions">
          <button type="button" className="sc-btn sc-btn-quiet" onClick={onClose} disabled={running}>
            Cancel
          </button>
          <button type="button" className="sc-btn sc-btn-primary" disabled={!prompt.trim() || running} onClick={start}>
            {running ? 'Drawing…' : 'Create scene'}
          </button>
        </div>
      </div>
    </div>
  );
}
