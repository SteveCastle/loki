import React, { useEffect, useMemo, useState } from 'react';
import { mediaUrl } from '../../platform';
import {
  branchesFrom,
  buildContinueJob,
  continuable,
  continueDefaults,
  type ContinueSettings,
  type LoraRun,
  type TrainLoraRunsState,
} from './runs';
import type { TrainLoraJob } from './plan';

// Train LoRA → "Continue a run": a finished or stopped training run, from any of its checkpoints, with more steps
// (and optionally another learning rate). The checkpoint's previews show where the run stands.

const LRS: Array<{ v: number; label: string }> = [
  { v: 5e-5, label: '5e-5' },
  { v: 1e-4, label: '1e-4' },
  { v: 2e-4, label: '2e-4' },
];

const KIND_LABEL: Record<string, string> = {
  character: 'Character',
  style: 'Style',
  object: 'Object',
  concept: 'Concept',
  general: 'General',
};

function runLine(r: LoraRun): string {
  const parts = [
    KIND_LABEL[r.kind || ''] || r.kind,
    `${r.images} images`,
    r.finished ? `finished at ${r.step}` : `stopped at ${r.step} of ${r.steps}`,
  ];
  if (r.loss !== undefined) parts.push(`loss ${r.loss.toFixed(3)}`);
  if (r.model) parts.push(r.model.replace(/\.safetensors$/i, ''));
  return parts.filter(Boolean).join(' · ');
}

export interface ContinueRunProps {
  runs: TrainLoraRunsState;
  /** the job is posted by the dialog (shared busy / error handling) */
  onSubmit: (job: TrainLoraJob) => void;
  busy: boolean;
  /** pre-select a run (by folder) */
  initialRun?: string;
  onCancel: () => void;
}

export default function ContinueRun({
  runs,
  onSubmit,
  busy,
  initialRun,
  onCancel,
}: ContinueRunProps) {
  const list = useMemo(() => continuable(runs.runs), [runs.runs]);
  const [dir, setDir] = useState<string>(initialRun || '');
  const run = list.find((r) => r.dir === dir) || list[0];
  const [c, setC] = useState<ContinueSettings | null>(null);
  // a new run picked: its defaults from its last checkpoint
  useEffect(() => {
    if (run) setC(continueDefaults(run));
  }, [run?.dir]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!runs.loaded)
    return <p className="tl-faint">Looking for training runs…</p>;
  if (runs.error) return <div className="tl-error">{runs.error}</div>;
  if (!run || !c)
    return (
      <p className="tl-faint">
        No training runs with a checkpoint yet. Start one with “New LoRA”.
      </p>
    );

  const cp =
    run.checkpoints.find((x) => x.dir === c.checkpoint) ||
    run.checkpoints[run.checkpoints.length - 1];
  const branch = branchesFrom(run, cp);
  const tooFew = c.steps <= cp.step;
  const runLr = run.lr ?? 1e-4;
  const set = (patch: Partial<ContinueSettings>) =>
    setC((prev) => (prev ? { ...prev, ...patch } : prev));
  return (
    <>
      <div className="tl-runs" role="listbox" aria-label="Training runs">
        {list.map((r) => (
          <button
            key={r.dir}
            type="button"
            role="option"
            aria-selected={r.dir === run.dir}
            className={`tl-run${r.dir === run.dir ? ' on' : ''}`}
            onClick={() => setDir(r.dir)}
            title={r.dir}
          >
            <span className="tl-run-name">
              {r.name}
              {r.trigger && r.trigger !== r.name ? (
                <span className="tl-run-trig"> {r.trigger}</span>
              ) : null}
            </span>
            <span className="tl-run-line">{runLine(r)}</span>
          </button>
        ))}
      </div>

      <div className="tl-grid">
        <span className="tl-label">From</span>
        <div className="tl-seg" role="radiogroup" aria-label="Checkpoint">
          {run.checkpoints.map((x) => (
            <button
              key={x.dir}
              type="button"
              role="radio"
              aria-checked={x.dir === cp.dir}
              className={x.dir === cp.dir ? 'on' : ''}
              onClick={() =>
                set({
                  checkpoint: x.dir,
                  steps: Math.max(c.steps, x.step + 250),
                })
              }
            >
              {x.step}
            </button>
          ))}
        </div>

        {cp.samples.length > 0 && (
          <>
            <span className="tl-label">Previews</span>
            <div className="tl-previews">
              {cp.samples.slice(0, 2).map((s, i) => (
                <figure key={s}>
                  <img
                    src={mediaUrl(s)}
                    alt={`Preview at step ${cp.step}`}
                    loading="lazy"
                  />
                  <figcaption>
                    {i === 0 ? 'a training caption' : 'a new scene'}
                  </figcaption>
                </figure>
              ))}
            </div>
          </>
        )}

        <span className="tl-label">Train until</span>
        <div className="tl-row">
          <input
            className="tl-input tl-num"
            type="number"
            min={cp.step + 50}
            step={250}
            value={c.steps}
            aria-label="Train until step"
            onChange={(e) =>
              set({ steps: Math.max(0, Number(e.target.value) || 0) })
            }
          />
          <span className="tl-faint-inline">
            steps in all ({Math.max(0, c.steps - cp.step)} more)
          </span>
        </div>

        <span className="tl-label">Learning rate</span>
        <div>
          <div className="tl-seg" role="radiogroup" aria-label="Learning rate">
            <button
              type="button"
              role="radio"
              aria-checked={c.lr === 0}
              className={c.lr === 0 ? 'on' : ''}
              onClick={() => set({ lr: 0 })}
            >
              As before ({runLr.toExponential(0)})
            </button>
            {LRS.filter((x) => Math.abs(x.v - runLr) > 1e-9).map((x) => (
              <button
                key={x.label}
                type="button"
                role="radio"
                aria-checked={c.lr === x.v}
                className={c.lr === x.v ? 'on' : ''}
                onClick={() => set({ lr: x.v })}
              >
                {x.label}
              </button>
            ))}
          </div>
          {run.kind === 'style' && runLr < 1e-4 && (
            <div className="tl-hint">
              This style trained at 5e-5, which is slow to carry over to new
              subjects; 1e-4 is the better rate.
            </div>
          )}
        </div>
      </div>

      <p className="tl-faint">
        {branch
          ? `Step ${cp.step} is not the run’s last checkpoint: training continues in a new run, ${branch}, and ${run.name} stays as it is.`
          : `Training continues in ${run.name} with its images, captions, model and rank.`}{' '}
        Watch the new-scene preview: once it takes on the look, that checkpoint
        is ready.
      </p>
      <div className="tl-actions">
        <button
          type="button"
          className="tl-btn"
          onClick={onCancel}
          disabled={busy}
        >
          Cancel
        </button>
        <button
          type="button"
          className="tl-btn tl-primary"
          disabled={busy || tooFew}
          onClick={() =>
            onSubmit(buildContinueJob({ ...c, checkpoint: cp.dir }))
          }
          title={tooFew ? `Train until a step after ${cp.step}` : undefined}
        >
          {busy ? 'Starting…' : 'Continue training'}
        </button>
      </div>
    </>
  );
}
