import React, { useState } from 'react';
import { selectedObjects, type PickState } from './pick';
import type { SceneTrack, SegPrompt } from './scene-api';

// The bars of the Select tool: what to pick (a name, People / Foreground /
// Background, or clicks and boxes on the stage), what was found, and what to
// do with it; and for a roto, the frame being picked on, the prompts the clip
// will be tracked from and the tracking itself.

export interface PickBarProps {
  pick: PickState | null;
  /** Clip mode (a roto): the prompts are kept per frame and tracked instead of added to a mask. */
  roto: boolean;
  /** Whether a layer is selected to add to / cut from. */
  hasLayer: boolean;
  onFind: (text: string) => void;
  onPreset: (which: 'people' | 'foreground' | 'background') => void;
  onCycle: () => void;
  onToggle: (i: number) => void;
  onAdd: (cut: boolean) => void;
  onNewLayer: (kind: 'still' | 'motion') => void;
  /** Roto: keep this pick as prompts on its frame. */
  onKeep: () => void;
  onClear: () => void;
}

export function PickBar(props: PickBarProps) {
  const { pick, roto } = props;
  const [text, setText] = useState('');
  const objs = pick ? selectedObjects(pick) : [];
  const many = !!pick && (!!pick.text || !!pick.preset) && pick.objects.length > 0;
  const readings = pick && !many && pick.objects.length ? (pick.objects[0].candidates || []).length : 0;
  const has = objs.length > 0 || pick?.preset === 'background';
  return (
    <div className="sc-pickbar" role="toolbar" aria-label="Select objects">
      <form
        className="sc-pick-find"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) props.onFind(text);
        }}
      >
        <input className="sc-input" value={text} placeholder={roto ? 'Find in this frame… (the woman)' : 'Find… (person, the sky, the red car)'} onChange={(e) => setText(e.target.value)} aria-label="Find by name" />
      </form>
      {!roto && (
        <span className="sc-pick-presets">
          <button type="button" className={`sc-chip${pick?.preset === 'people' ? ' on' : ''}`} onClick={() => props.onPreset('people')} title="Every person in the picture">
            People
          </button>
          <button type="button" className={`sc-chip${pick?.preset === 'foreground' ? ' on' : ''}`} onClick={() => props.onPreset('foreground')} title="Every object, without the setting (sky, ground, walls)">
            Foreground
          </button>
          <button type="button" className={`sc-chip${pick?.preset === 'background' ? ' on' : ''}`} onClick={() => props.onPreset('background')} title="Everything but the objects: the setting">
            Background
          </button>
        </span>
      )}
      <span className="sc-pick-status">
        {!pick || (!pick.busy && !pick.objects.length && !pick.error) ? (
          <span className="sc-faint">Click an object (Alt-click: not this), or drag a box round it.</span>
        ) : pick.busy ? (
          <span className="sc-faint">Selecting…</span>
        ) : pick.error ? (
          <span className="sc-pick-error">{pick.error}</span>
        ) : many ? (
          <span className="sc-pick-objects">
            {pick.objects.map((o, i) => (
              <button
                // eslint-disable-next-line react/no-array-index-key
                key={i}
                type="button"
                className={`sc-chip${pick.chosen[i] ? ' on' : ''}`}
                onClick={() => props.onToggle(i)}
                title={`${Math.round(o.score * 100)}% sure · click to leave it out`}
              >
                {o.label} {i + 1}
              </button>
            ))}
          </span>
        ) : (
          <span className="sc-faint">
            {objs[0] ? `${!objs[0].label || objs[0].label === 'click' || objs[0].label === 'box' ? 'Object' : objs[0].label} ` : ''}
            {readings > 1 && (
              <button type="button" className="sc-btn sc-btn-tiny sc-pick-reading" onClick={props.onCycle} title="Tab: another reading of what you clicked (a part of it, the object, or the whole it belongs to), best first">
                reading {pick.candidate + 1}/{readings} · Tab
              </button>
            )}
            {pick.ms !== undefined ? ` · ${pick.ms} ms` : ''}
          </span>
        )}
      </span>
      <span className="sc-pick-actions">
        {roto ? (
          <button type="button" className="sc-btn sc-btn-primary" disabled={!has || !pick || pick.busy} onClick={props.onKeep} title="Keep this on this frame: the clip is tracked from it (Enter)">
            Keep on this frame ↵
          </button>
        ) : (
          <>
            <button type="button" className="sc-btn sc-btn-primary" disabled={!has || !props.hasLayer || !pick || pick.busy} onClick={() => props.onAdd(false)} title={props.hasLayer ? 'Add it to the layer’s mask (Enter)' : 'Select a layer first'}>
              Add ↵
            </button>
            <button type="button" className="sc-btn" disabled={!has || !props.hasLayer || !pick || pick.busy} onClick={() => props.onAdd(true)} title="Cut it out of the layer’s mask (Alt+Enter)">
              Cut
            </button>
            <button type="button" className="sc-btn sc-btn-quiet" disabled={!has || !pick || pick.busy} onClick={() => props.onNewLayer('still')} title="A new Edit layer masked to it">
              → Edit
            </button>
            <button type="button" className="sc-btn sc-btn-quiet" disabled={!has || !pick || pick.busy} onClick={() => props.onNewLayer('motion')} title="A new Motion layer masked to it">
              → Motion
            </button>
          </>
        )}
        <button type="button" className="sc-btn sc-btn-quiet" disabled={!pick} onClick={props.onClear} title="Esc">
          Clear
        </button>
      </span>
    </div>
  );
}

export interface RotoBarProps {
  frame: number;
  frames: number;
  fps: number;
  prompts: SegPrompt[];
  track: SceneTrack | null | undefined;
  showMatte: boolean;
  onFrame: (f: number) => void;
  onRemovePrompt: (i: number) => void;
  onTrack: () => void;
  onShowMatte: (v: boolean) => void;
  onDone: () => void;
}

function describe(p: SegPrompt): string {
  if (p.text) return `“${p.text}”`;
  const n = (p.points || []).length;
  const parts = [];
  if (n) parts.push(`${n} click${n > 1 ? 's' : ''}`);
  if (p.box) parts.push('a box');
  return parts.join(' + ') || 'a pick';
}

export function RotoBar(props: RotoBarProps) {
  const { frame, frames, fps, prompts, track } = props;
  const running = !!track && ['pending', 'in_progress', 'paused'].includes(track.state);
  const phase = running && track ? (track.state === 'pending' ? 'Waiting for the GPU…' : `Tracking ${track.total ? Math.round((track.done / track.total) * 100) : 0}%`) : '';
  return (
    <div className="sc-rotobar" role="toolbar" aria-label="Roto">
      <span className="sc-label">Roto</span>
      <button type="button" className="sc-btn sc-btn-tiny" disabled={frame <= 0} onClick={() => props.onFrame(frame - 1)} title="Previous frame (,)">
        ◀
      </button>
      <input type="range" className="sc-roto-scrub" min={0} max={Math.max(0, frames - 1)} value={frame} onChange={(e) => props.onFrame(Number(e.target.value))} aria-label="Frame" />
      <button type="button" className="sc-btn sc-btn-tiny" disabled={frame >= frames - 1} onClick={() => props.onFrame(frame + 1)} title="Next frame (.)">
        ▶
      </button>
      <span className="sc-roto-time">
        {frame + 1}/{frames} · {(frame / fps).toFixed(2)} s
      </span>
      <span className="sc-roto-prompts">
        {prompts.length ? (
          prompts.map((p, i) => (
            // eslint-disable-next-line react/no-array-index-key
            <span key={i} className="sc-chip on">
              <button type="button" className="sc-roto-go" onClick={() => props.onFrame(p.frame)} title="Go to its frame">
                f{p.frame + 1}: {describe(p)}
              </button>
              <button type="button" className="sc-row-x" onClick={() => props.onRemovePrompt(i)} title="Remove">
                ×
              </button>
            </span>
          ))
        ) : (
          <span className="sc-faint">Pick what to keep on any frame, then Keep it.</span>
        )}
      </span>
      {track?.state === 'completed' && (
        <label className="sc-roto-show" title="Tint the tracked matte on each frame">
          <input type="checkbox" checked={props.showMatte} onChange={(e) => props.onShowMatte(e.target.checked)} /> Matte
        </label>
      )}
      {track?.state === 'error' && <span className="sc-pick-error" title={track.error}>Tracking failed</span>}
      {running ? (
        <span className="sc-faint">{phase}</span>
      ) : (
        <button type="button" className="sc-btn sc-btn-primary" disabled={!prompts.length} onClick={props.onTrack} title="Track it through the whole clip (a GPU job)">
          {track?.state === 'completed' ? 'Track again' : 'Track'}
        </button>
      )}
      <button type="button" className="sc-btn sc-btn-quiet" onClick={props.onDone}>
        Done
      </button>
    </div>
  );
}

