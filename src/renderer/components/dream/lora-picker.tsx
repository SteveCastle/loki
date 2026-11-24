// The LoRA picker of Generate / Reimagine: any number of LoRAs from the server's list (loras.ts), each with a
// strength, offered only when they fit the chosen model (MOSS-Image LoRAs for MOSS-Image models). Chosen LoRAs that
// do not fit the model stay listed but greyed (and are not sent) so switching models back keeps them.
import React, { useMemo, useState } from 'react';
import type { DreamModel } from './dream-models';
import { loraDetail, loraFits, type ChosenLora, type LoraFile, type LoraList } from './loras';
import './lora-picker.css';

export interface LoraPickerProps {
  list: LoraList;
  /** The chosen model (undefined: the server's default). */
  model: DreamModel | undefined;
  value: ChosenLora[];
  onChange: (next: ChosenLora[]) => void;
  onRefresh: () => void;
  /** The prompt, to offer each chosen LoRA's trigger word when it is missing from it. */
  prompt?: string;
  /** Put a trigger word into the prompt. */
  onTrigger?: (trigger: string) => void;
}

const same = (a: string, b: string) => a.replace(/\\/g, '/').toLowerCase() === b.replace(/\\/g, '/').toLowerCase();

function matches(l: LoraFile, q: string): boolean {
  const hay = [l.name, l.file, l.trigger, l.baseModel, l.run].join(' ').toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((t) => hay.includes(t));
}

export default function LoraPicker({ list, model, value, onChange, onRefresh, prompt, onTrigger }: LoraPickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [checkpoints, setCheckpoints] = useState(false);
  const find = (path: string) => list.loras.find((l) => same(l.path, path));
  const chosen = (path: string) => value.some((c) => same(c.path, path));
  const shown = useMemo(
    () => list.loras.filter((l) => (checkpoints || !l.checkpoint) && matches(l, query)),
    [list.loras, checkpoints, query]
  );
  const fitting = list.loras.filter((l) => loraFits(l, model).ok).length;
  const add = (l: LoraFile) => {
    if (chosen(l.path)) onChange(value.filter((c) => !same(c.path, l.path)));
    else onChange([...value, { path: l.path, scale: 1, name: l.name, trigger: l.trigger }]);
  };
  const setScale = (i: number, scale: number) => onChange(value.map((c, j) => (j === i ? { ...c, scale } : c)));
  const promptLower = (prompt || '').toLowerCase();

  return (
    <div className="lp">
      {value.length > 0 && (
        <ul className="lp-chosen" aria-label="Chosen LoRAs">
          {value.map((c, i) => {
            const l = find(c.path);
            const fit = l ? loraFits(l, model) : { ok: true };
            const trigger = l?.trigger || c.trigger;
            const name = l?.name || c.name || c.path.split(/[\\/]/).pop();
            return (
              <li key={c.path} className={`lp-row${fit.ok ? '' : ' off'}`}>
                <div className="lp-row-main">
                  <span className="lp-name" title={c.path}>
                    {name}
                  </span>
                  <span className="lp-detail">{!fit.ok ? `not used: ${fit.why}` : l ? loraDetail(l) : list.loaded ? 'not found on the server' : ''}</span>
                </div>
                {trigger && onTrigger && !promptLower.includes(trigger.toLowerCase()) && (
                  <button type="button" className="lp-trigger" onClick={() => onTrigger(trigger)} title={`Add the trigger word "${trigger}" to the prompt: the LoRA responds to it`}>
                    + {trigger}
                  </button>
                )}
                <input
                  className="lp-scale"
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={c.scale}
                  aria-label={`${name} strength`}
                  disabled={!fit.ok}
                  onChange={(e) => setScale(i, Number(e.target.value))}
                />
                <span className="lp-scale-val">{c.scale.toFixed(2)}</span>
                <button type="button" className="lp-x" aria-label={`Remove ${name}`} onClick={() => onChange(value.filter((_, j) => j !== i))}>
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      )}
      <div className="lp-bar">
        <button type="button" className="lp-add" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          {open ? 'Done' : value.length ? 'Add another LoRA' : 'Add a LoRA'}
        </button>
        <span className="lp-count">
          {list.loading ? 'Looking for LoRAs…' : list.error ? list.error : list.loaded ? `${fitting} fit this model` : ''}
        </span>
      </div>
      {open && (
        <div className="lp-pop">
          <div className="lp-tools">
            <input className="lp-search" autoFocus placeholder="Search LoRAs" aria-label="Search LoRAs" value={query} onChange={(e) => setQuery(e.target.value)} />
            <label className="lp-check">
              <input type="checkbox" checked={checkpoints} onChange={(e) => setCheckpoints(e.target.checked)} /> Checkpoints
            </label>
            <button type="button" className="lp-refresh" onClick={onRefresh} title="Look for new LoRAs (a training run that just finished)">
              Refresh
            </button>
          </div>
          {shown.length === 0 ? (
            <p className="lp-empty">
              {list.loras.length === 0
                ? `No LoRAs yet. Train one (Train LoRA… in the context palette) or put .safetensors LoRAs in ${list.dirs[1] || list.dirs[0] || 'the server’s loras folder'}.`
                : 'No LoRA matches.'}
            </p>
          ) : (
            <ul className="lp-list" role="listbox" aria-multiselectable="true" aria-label="LoRAs">
              {shown.map((l) => {
                const fit = loraFits(l, model);
                const on = chosen(l.path);
                return (
                  <li key={l.path}>
                    <button type="button" role="option" aria-selected={on} className={`lp-item${on ? ' on' : ''}`} disabled={!fit.ok && !on} onClick={() => add(l)} title={l.path}>
                      <span className="lp-tick">{on ? '✓' : ''}</span>
                      <span className="lp-item-main">
                        <span className="lp-name">
                          {l.name}
                          {l.trigger ? <span className="lp-trig"> {l.trigger}</span> : null}
                        </span>
                        <span className="lp-detail">{fit.ok ? [loraDetail(l), fit.note].filter(Boolean).join(' · ') : fit.why}</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
