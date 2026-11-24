import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { mediaServerBase } from '../../platform';
import ModelPicker from '../dream/model-picker';
import { useDreamModels } from '../dream/dream-models';
import {
  closeTrainLora,
  useTrainLoraRequest,
  type TrainLoraRequest,
} from './store';
import {
  DEFAULT_SETTINGS,
  KINDS,
  RANKS,
  RESOLUTIONS,
  buildTrainLoraJob,
  defaultSteps,
  estimateMinutes,
  isTrainImage,
  needsTrigger,
  runName,
  trainableModels,
  defaultTrainModel,
  type LoraCaptions,
  type TrainLoraSettings,
} from './plan';
import ContinueRun from './continue-run';
import { useLoraRuns } from './runs';
import type { TrainLoraJob } from './plan';
import './train-lora.css';

// "Train LoRA…" from the context palette: the selection's images become a MOSS-Image (Kromoss) LoRA, trained by the
// server's train-lora task (loki-tune). The dialog only gathers the settings; progress, checkpoints and previews
// follow in the job queue and the run folder.

const REMEMBER_KEY = 'loki.train-lora.settings.v1';

function remembered(): TrainLoraSettings {
  try {
    const raw = window.localStorage.getItem(REMEMBER_KEY);
    if (raw) {
      const v = JSON.parse(raw);
      // a trigger and a name belong to one LoRA: start each dialog without them
      return {
        ...DEFAULT_SETTINGS,
        ...v,
        name: '',
        trigger: '',
        className: v.className ?? '',
      };
    }
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_SETTINGS;
}

function remember(s: TrainLoraSettings) {
  try {
    const { kind, captions, rank, resolution, className, model } = s;
    window.localStorage.setItem(
      REMEMBER_KEY,
      JSON.stringify({ kind, captions, rank, resolution, className, model })
    );
  } catch {
    /* storage unavailable */
  }
}

const CAPTIONS: Array<{ id: LoraCaptions; label: string; title: string }> = [
  {
    id: 'auto',
    label: 'Written for the kind',
    title:
      'loki-scribe writes a long and a short caption per image, following the rules for the kind above (recommended)',
  },
  {
    id: 'library',
    label: 'Library descriptions',
    title:
      "Each item's description from the library where it has one (loki-scribe writes the rest); descriptions usually describe everything, which suits General and Style LoRAs less",
  },
  {
    id: 'none',
    label: 'Trigger only',
    title: 'No captions: every image trains on the trigger word alone',
  },
];

export function TrainLoraDialog({
  req,
  onClose,
}: {
  req: TrainLoraRequest;
  onClose: () => void;
}) {
  const [s, setS] = useState<TrainLoraSettings>(remembered);
  const [customSteps, setCustomSteps] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const images = useMemo(() => req.paths.filter(isTrainImage), [req.paths]);
  const skipped = req.paths.length - images.length;
  // a library query trains on everything it matches; the count is known when the view is the query
  const byQuery = !!req.query64;
  const n: number | null = byQuery ? req.count ?? null : images.length;
  const kind = KINDS.find((k) => k.id === s.kind) ?? KINDS[0];
  // New LoRA (from the images) or Continue a run (a checkpoint of an earlier run; no images needed)
  const [mode, setMode] = useState<'new' | 'continue'>(
    req.mode ?? (req.paths.length === 0 && !req.query64 ? 'continue' : 'new')
  );
  const runs = useLoraRuns(mode === 'continue', req.authToken);
  // the base model: MOSS-Image checkpoints on the server (loki-dream's catalog), default = loki-tune's
  const dream = useDreamModels(true, req.authToken);
  const catalog = useMemo(() => {
    const models = trainableModels(dream.catalog.models);
    return { ...dream.catalog, models, defaultFile: defaultTrainModel(models) };
  }, [dream.catalog]);
  const set = (patch: Partial<TrainLoraSettings>) =>
    setS((prev) => ({ ...prev, ...patch }));
  const missingTrigger = needsTrigger(s.kind) && !s.trigger.trim();
  const canStart = n !== 0 && !missingTrigger && !busy;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  const post = async (job: TrainLoraJob, after?: () => void) => {
    setBusy(true);
    setError('');
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (req.authToken) headers.Authorization = `Bearer ${req.authToken}`;
      const res = await fetch(`${mediaServerBase}/create`, {
        method: 'POST',
        headers,
        body: JSON.stringify(job),
        signal: AbortSignal.timeout(10000),
        redirect: 'error',
      });
      if (!res.ok) throw new Error(`the server answered ${res.status}`);
      after?.();
      onClose();
    } catch (e) {
      setError(
        `Could not start the training: ${
          e instanceof Error ? e.message : String(e)
        }`
      );
      setBusy(false);
    }
  };
  const start = () => {
    if (!canStart) return;
    post(
      buildTrainLoraJob(
        images,
        customSteps ? s : { ...s, steps: 0 },
        req.query64
      ),
      () => remember(s)
    );
  };

  const auto = n === null ? null : defaultSteps(n, s.kind === 'style');
  const steps = customSteps && s.steps > 0 ? s.steps : auto;
  const minutes =
    n === null || steps === null
      ? null
      : estimateMinutes(n, { steps, resolution: s.resolution });
  return (
    <div
      className="tl-backdrop"
      role="presentation"
      onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}
    >
      <div className="tl-panel" role="dialog" aria-label="Train a LoRA">
        <div className="tl-head">
          <div className="tl-tabs" role="tablist" aria-label="Train a LoRA">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'new'}
              className={mode === 'new' ? 'on' : ''}
              onClick={() => setMode('new')}
            >
              New LoRA
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'continue'}
              className={mode === 'continue' ? 'on' : ''}
              onClick={() => setMode('continue')}
            >
              Continue a run
            </button>
          </div>
          {mode === 'new' && (
            <span className="tl-count">
              {byQuery
                ? `${
                    n === null
                      ? 'All images'
                      : `${n} image${n === 1 ? '' : 's'}`
                  } in ${req.label || 'the library query'}`
                : `${images.length} image${images.length === 1 ? '' : 's'}`}
              {!byQuery && skipped > 0
                ? ` (${skipped} other file${skipped === 1 ? '' : 's'} skipped)`
                : ''}
            </span>
          )}
        </div>
        {mode === 'continue' ? (
          <>
            <p className="tl-faint">
              Train a run further from one of its checkpoints: more steps,
              optionally another learning rate.
            </p>
            {error && <div className="tl-error">{error}</div>}
            <ContinueRun
              runs={runs}
              busy={busy}
              initialRun={req.run}
              onCancel={onClose}
              onSubmit={(job) => post(job)}
            />
          </>
        ) : (
          <>
            <p className="tl-faint">
              A MOSS-Image LoRA for Kromoss and other MOSS models, trained on
              the server with loki-tune. It works with loki-dream’s bf16 and
              int8 models.
            </p>
            {n === 0 && !byQuery && (
              <div className="tl-warn">
                No images selected: select the training images first, then open
                Train LoRA again, or continue a run.
              </div>
            )}
            {n !== null && n > 0 && n < 8 && (
              <div className="tl-warn">
                Few images: LoRAs usually learn best from 15–50 varied pictures
                (different poses, places, light).
              </div>
            )}

            <div className="tl-grid">
              <span className="tl-label">Teaches</span>
              <div>
                <div
                  className="tl-seg"
                  role="radiogroup"
                  aria-label="What the LoRA teaches"
                >
                  {KINDS.map((k) => (
                    <button
                      key={k.id}
                      type="button"
                      role="radio"
                      aria-checked={s.kind === k.id}
                      className={s.kind === k.id ? 'on' : ''}
                      onClick={() => set({ kind: k.id })}
                    >
                      {k.label}
                    </button>
                  ))}
                </div>
                <div className="tl-hint">{kind.hint}</div>
              </div>

              <span className="tl-label">Trigger word</span>
              <div className="tl-row">
                <input
                  className="tl-input"
                  autoFocus
                  value={s.trigger}
                  placeholder={
                    needsTrigger(s.kind)
                      ? 'a rare word, e.g. zkx'
                      : 'optional, e.g. inkwash'
                  }
                  aria-label="Trigger word"
                  onChange={(e) => set({ trigger: e.target.value })}
                />
                {kind.hasClass && (
                  <input
                    className="tl-input"
                    value={s.className}
                    placeholder={`is a… ${kind.classPlaceholder ?? ''}`}
                    aria-label="What the trigger is"
                    onChange={(e) => set({ className: e.target.value })}
                  />
                )}
              </div>

              <span className="tl-label">Base model</span>
              <div>
                <ModelPicker
                  catalog={catalog}
                  value={s.model}
                  onRefresh={dream.refresh}
                  onChange={(file) => set({ model: file })}
                  label="Base model"
                />
                <div className="tl-hint">
                  A LoRA fits the model it was trained on best: pick the
                  checkpoint you generate with. Kromoss LoRAs work on its bf16
                  and int8 files alike.
                </div>
              </div>

              <span className="tl-label">Name</span>
              <input
                className="tl-input"
                value={s.name}
                placeholder={runName(s)}
                aria-label="LoRA name"
                onChange={(e) => set({ name: e.target.value })}
              />

              <span className="tl-label">Captions</span>
              <div className="tl-seg" role="radiogroup" aria-label="Captions">
                {CAPTIONS.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    role="radio"
                    aria-checked={s.captions === c.id}
                    className={s.captions === c.id ? 'on' : ''}
                    title={c.title}
                    onClick={() => set({ captions: c.id })}
                  >
                    {c.label}
                  </button>
                ))}
              </div>

              <span className="tl-label">Length</span>
              <div className="tl-row">
                <div
                  className="tl-seg"
                  role="radiogroup"
                  aria-label="Training length"
                >
                  <button
                    type="button"
                    role="radio"
                    aria-checked={!customSteps}
                    className={customSteps ? '' : 'on'}
                    onClick={() => setCustomSteps(false)}
                    title="About 50 passes over the images, 1000–3000 steps"
                  >
                    {auto === null
                      ? 'Auto (~50 passes)'
                      : `Auto (${auto} steps)`}
                  </button>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={customSteps}
                    className={customSteps ? 'on' : ''}
                    onClick={() => setCustomSteps(true)}
                  >
                    Steps
                  </button>
                </div>
                {customSteps && (
                  <input
                    className="tl-input tl-num"
                    type="number"
                    min={50}
                    step={250}
                    value={s.steps || auto || 2000}
                    aria-label="Steps"
                    onChange={(e) =>
                      set({ steps: Math.max(0, Number(e.target.value) || 0) })
                    }
                  />
                )}
              </div>

              <span className="tl-label">Detail</span>
              <div
                className="tl-seg"
                role="radiogroup"
                aria-label="Training resolution"
              >
                {RESOLUTIONS.map((r) => (
                  <button
                    key={r}
                    type="button"
                    role="radio"
                    aria-checked={s.resolution === r}
                    className={s.resolution === r ? 'on' : ''}
                    onClick={() => set({ resolution: r })}
                    title={
                      r === 1024
                        ? 'Full detail (MOSS-Image trains at 1 MP)'
                        : 'Faster, less fine detail'
                    }
                  >
                    {r}px
                  </button>
                ))}
              </div>

              <span className="tl-label">Rank</span>
              <div className="tl-seg" role="radiogroup" aria-label="LoRA rank">
                {RANKS.map((r) => (
                  <button
                    key={r}
                    type="button"
                    role="radio"
                    aria-checked={s.rank === r}
                    className={s.rank === r ? 'on' : ''}
                    onClick={() => set({ rank: r })}
                    title={
                      r === 16
                        ? 'Characters and objects (default)'
                        : 'More capacity: complex styles, many subjects; larger file'
                    }
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            <p className="tl-faint">
              {minutes !== null ? `About ${minutes} min on an RTX 4090. ` : ''}
              Checkpoints and preview images are saved every 250 steps in the
              server’s loras/{runName(s)} folder; pausing or cancelling the job
              saves one first, and resuming continues from it.
            </p>
            {error && <div className="tl-error">{error}</div>}
            <div className="tl-actions">
              <button
                type="button"
                className="tl-btn"
                onClick={onClose}
                disabled={busy}
              >
                Cancel
              </button>
              <button
                type="button"
                className="tl-btn tl-primary"
                disabled={!canStart}
                onClick={start}
                title={
                  missingTrigger
                    ? `A ${kind.label.toLowerCase()} LoRA needs a trigger word`
                    : undefined
                }
              >
                {busy ? 'Starting…' : 'Start training'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** Root-level host (mounted next to TransformFlow): renders the dialog while a request is open. */
export default function TrainLoraHost() {
  const req = useTrainLoraRequest();
  if (!req) return null;
  return createPortal(
    <TrainLoraDialog req={req} onClose={closeTrainLora} />,
    document.body
  );
}
