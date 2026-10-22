import React, { useEffect, useState } from 'react';
import { mediaUrl } from '../../platform';
import { maskBounds, rectContains, type ContextSize } from './mask';
import { layerTiming, type LayerOptions, type LoopKind, type SceneLayer } from './scene-api';

// The Scene editor's right-hand panel: the layer stack (top of the list = top
// of the picture) and the selected layer's settings, generation and versions.

const STILL_IDEAS = ['make it night with warm window light', 'replace with lush green ivy', 'add soft falling snow', 'turn it into a marble statue', 'remove the object, fill with the background'];
const MOTION_IDEAS = [
  'gentle wind moves the hair and clothes',
  'water ripples and sparkles in the light',
  'clouds drift slowly across the sky',
  'flames flicker and embers rise',
  'leaves sway softly in a breeze',
  'the person blinks and breathes calmly',
];

const CONTEXTS: Array<{ value: ContextSize; label: string; title: string }> = [
  { value: 'tight', label: 'Tight', title: 'The engine sees little beyond what you painted: most detail, least awareness of the scene' },
  { value: 'normal', label: 'Normal', title: 'A margin around the painted area so the result blends with its surroundings' },
  { value: 'wide', label: 'Wide', title: 'A large margin: best blending and consistency, less detail per pixel' },
  { value: 'full', label: 'Full frame', title: 'The engine sees the whole picture; only the painted area is used' },
];

function Seg<T extends string | number>(props: { value: T; options: Array<{ value: T; label: string; title?: string }>; onChange: (v: T) => void; disabled?: boolean }) {
  return (
    <div className="sc-seg" role="radiogroup">
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={props.value === o.value}
          className={props.value === o.value ? 'on' : ''}
          title={o.title}
          disabled={props.disabled}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function KindIcon({ kind, size = 14 }: { kind: 'still' | 'motion'; size?: number }) {
  return kind === 'motion' ? (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
      <path d="M2 8c1.5-3 3-3 4.5 0s3 3 4.5 0 3-3 3-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M2 12c1.5-2 3-2 4.5 0s3 2 4.5 0" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" opacity="0.6" />
    </svg>
  ) : (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
      <path d="M3 13l2.5-.6L13 4.9 11.1 3 3.6 10.5z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M9.8 4.3l1.9 1.9" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function Eye({ off }: { off?: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden>
      <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" fill="none" stroke="currentColor" strokeWidth="1.3" />
      {off ? <path d="M3 13L13 3" stroke="currentColor" strokeWidth="1.4" /> : <circle cx="8" cy="8" r="2" fill="currentColor" />}
    </svg>
  );
}

function statusOf(l: SceneLayer): { text: string; tone: 'busy' | 'error' | 'ok' | 'idle' } {
  if (l.job) {
    if (l.job.step === 'scene-lock') return { text: 'locking edges', tone: 'busy' };
    if (l.job.state === 'pending') return { text: 'queued', tone: 'busy' };
    if (l.job.state === 'paused') return { text: 'paused', tone: 'busy' };
    return { text: l.job.total ? `${Math.round((l.job.done / l.job.total) * 100)}%` : 'starting', tone: 'busy' };
  }
  if (l.error) return { text: 'failed', tone: 'error' };
  if (l.versions.length) return { text: l.active >= 0 ? `v${l.active + 1}/${l.versions.length}` : 'off', tone: 'ok' };
  if (!maskBounds(l.shapes || [])) return { text: 'paint an area', tone: 'idle' };
  return { text: 'ready', tone: 'idle' };
}

export interface LayerListProps {
  layers: SceneLayer[];
  selectedId: string | null;
  baseName: string;
  onSelect: (id: string | null) => void;
  onAdd: (kind: 'still' | 'motion') => void;
  onToggleHidden: (id: string) => void;
  onMove: (id: string, dir: 1 | -1) => void;
}

export function LayerList(props: LayerListProps) {
  const { layers, selectedId } = props;
  const top = [...layers].reverse();
  return (
    <section className="sc-section sc-layers">
      <div className="sc-section-head">
        <h3>Layers</h3>
        <div className="sc-add">
          <button type="button" className="sc-btn" onClick={() => props.onAdd('still')} title="A still edit of one area (loki-retouch)">
            <KindIcon kind="still" /> Edit
          </button>
          <button type="button" className="sc-btn" onClick={() => props.onAdd('motion')} title="Bring one area to life (loki-reshoot)">
            <KindIcon kind="motion" /> Motion
          </button>
        </div>
      </div>
      <ol className="sc-layer-list">
        {top.map((l, i) => {
          const st = statusOf(l);
          return (
            <li key={l.id} className={`sc-layer-row${l.id === selectedId ? ' selected' : ''}${l.hidden ? ' hidden' : ''}`}>
              <button type="button" className="sc-layer-main" onClick={() => props.onSelect(l.id)}>
                <span className={`sc-kind ${l.kind}`}>
                  <KindIcon kind={l.kind} />
                </span>
                <span className="sc-layer-name">{l.name || (l.kind === 'motion' ? 'Motion' : 'Edit')}</span>
                <span className={`sc-status ${st.tone}`}>{st.text}</span>
              </button>
              <span className="sc-row-tools">
                <button type="button" title="Move up" disabled={i === 0} onClick={() => props.onMove(l.id, 1)}>
                  ▲
                </button>
                <button type="button" title="Move down" disabled={i === top.length - 1} onClick={() => props.onMove(l.id, -1)}>
                  ▼
                </button>
                <button type="button" title={l.hidden ? 'Show' : 'Hide'} onClick={() => props.onToggleHidden(l.id)}>
                  <Eye off={l.hidden} />
                </button>
              </span>
            </li>
          );
        })}
        <li className={`sc-layer-row base${selectedId === null ? ' selected' : ''}`}>
          <button type="button" className="sc-layer-main" onClick={() => props.onSelect(null)}>
            <span className="sc-kind base">▣</span>
            <span className="sc-layer-name">Base · {props.baseName}</span>
          </button>
        </li>
      </ol>
    </section>
  );
}

export interface InspectorProps {
  layer: SceneLayer;
  busy: boolean;
  onPatch: (patch: Partial<SceneLayer>) => void;
  onOptions: (patch: LayerOptions) => void;
  onGenerate: () => void;
  onCancel: () => void;
  onDelete: () => void;
  onClearMask: () => void;
  onDeleteVersion: (n: number) => void;
  /** Discard every version but n. */
  onKeepOnly: (n: number) => void;
  onLoop: (n: number, kind: LoopKind) => void;
  /** Whether the active version's motion map is shown; toggled by onShowMotion. */
  motionShown: boolean;
  onShowMotion: () => void;
  onGrowToMotion: () => void;
  onClearProtect: () => void;
  /** The active version's PTS multiplier in the scene (layerRate: 1 = as generated, 2 = half speed). */
  loopSpeed: number;
}

/** A version chip's discard button: the first click arms it, the second (within 3 s) discards. */
function DiscardButton({ label, onDiscard }: { label: string; onDiscard: () => void }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return undefined;
    const t = window.setTimeout(() => setArmed(false), 3000);
    return () => window.clearTimeout(t);
  }, [armed]);
  return (
    <button
      type="button"
      className={`sc-version-x${armed ? ' armed' : ''}`}
      title={armed ? `Click again to discard ${label} (deletes its file)` : `Discard ${label}`}
      onClick={(e) => {
        e.stopPropagation();
        if (armed) onDiscard();
        else setArmed(true);
      }}
    >
      {armed ? 'discard?' : '×'}
    </button>
  );
}

export function Inspector(props: InspectorProps) {
  const { layer: l, busy } = props;
  const o = l.options || {};
  const [confirmDelete, setConfirmDelete] = useState(false);
  const motion = l.kind === 'motion';
  const painted = maskBounds(l.shapes || []);
  const version = l.versions[l.active];
  const running = !!l.job;
  const needPrompt = !motion && !l.prompt.trim();
  const outside = version && painted && !rectContains(version.region, painted);
  const ideas = motion ? MOTION_IDEAS : STILL_IDEAS;
  const timing = layerTiming(o);
  // An offset of one loop is no offset: the slider spans one loop as the layer plays it.
  const offsetMax = Math.max(0.5, Math.round((version?.duration || o.duration || 5) * (props.loopSpeed || 1) * 20) / 20);
  const appendIdea = (idea: string) => {
    const p = l.prompt.trim();
    props.onPatch({ prompt: p ? `${p.replace(/[.,;]$/, '')}, ${idea}` : idea });
  };

  return (
    <section className="sc-section sc-inspector">
      <div className="sc-field-row">
        <input className="sc-name" value={l.name} placeholder={motion ? 'Motion' : 'Edit'} onChange={(e) => props.onPatch({ name: e.target.value })} aria-label="Layer name" />
        <Seg
          value={l.kind}
          disabled={l.versions.length > 0 || running}
          onChange={(kind) => props.onPatch({ kind })}
          options={[
            { value: 'still', label: 'Edit', title: 'Change the look of the area (a still image edit)' },
            { value: 'motion', label: 'Motion', title: 'Animate the area (a short looping video)' },
          ]}
        />
      </div>

      <label className="sc-label" htmlFor="sc-prompt">
        {motion ? 'How should it move?' : 'What should change here?'}
      </label>
      <textarea
        id="sc-prompt"
        className="sc-prompt"
        value={l.prompt}
        rows={4}
        placeholder={motion ? 'e.g. the waterfall flows and mist drifts up (leave empty for natural ambient motion)' : 'e.g. replace the sky with a stormy sunset'}
        onChange={(e) => props.onPatch({ prompt: e.target.value })}
      />
      <div className="sc-ideas">
        {ideas.map((idea) => (
          <button key={idea} type="button" className="sc-chip" onClick={() => appendIdea(idea)}>
            + {idea}
          </button>
        ))}
      </div>
      {motion && (
        <>
          <label className="sc-label" htmlFor="sc-describe">
            Subject <span className="sc-faint">(optional: anchors what may move)</span>
          </label>
          <input id="sc-describe" className="sc-input" value={o.describe || ''} placeholder="e.g. the waterfall behind the cabin" onChange={(e) => props.onOptions({ describe: e.target.value })} />
        </>
      )}

      <div className="sc-grid">
        <span className="sc-label">Context</span>
        <Seg value={o.context || 'normal'} options={CONTEXTS} onChange={(context) => props.onOptions({ context })} />
        <span className="sc-label" title="How softly the area blends into the picture around it">
          Soft edge
        </span>
        <div className="sc-slider">
          <input type="range" min={0} max={160} step={1} value={l.feather} onChange={(e) => props.onPatch({ feather: Number(e.target.value) })} />
          <span>{Math.round(l.feather)} px</span>
        </div>
        {motion ? (
          <>
            <span className="sc-label">Length</span>
            <Seg
              value={o.duration || 5}
              onChange={(duration) => props.onOptions({ duration })}
              options={[5, 8, 10, 15].map((d) => ({ value: d, label: `${d}s` }))}
            />
            <span className="sc-label">Engine</span>
            <Seg
              value={o.model || 'h3'}
              onChange={(model) => props.onOptions({ model })}
              options={[
                { value: 'h3', label: 'Best', title: 'MiniMax H3, 20 steps' },
                { value: 'fast', label: 'Fast', title: 'FastH3 Omni, 8 steps: about 2.5× faster, may lose some quality' },
              ]}
            />
            <span className="sc-label" title="How many pixels the engine renders the area at">
              Resolution
            </span>
            <Seg
              value={o.megapixels || 0.5}
              onChange={(megapixels) => props.onOptions({ megapixels })}
              options={[
                { value: 0.5, label: 'Standard', title: 'About 0.5 MP for the area' },
                { value: 0.9, label: 'High', title: 'About 0.9 MP for the area: sharper, slower' },
              ]}
            />
            <span className="sc-label" title="Keep the area's surroundings perfectly still">
              Edges
            </span>
            <Seg
              value={o.lock === false ? 0 : 1}
              onChange={(v) => props.onOptions({ lock: v === 1 })}
              options={[
                {
                  value: 1,
                  label: 'Locked',
                  title: 'Wherever the generated video does not really move, it becomes the still itself, and the moving part takes the still’s tones: the edges blend back into the picture with no shimmer, drift or halo',
                },
                { value: 0, label: 'Free', title: 'Use the engine’s video as it is (it may shimmer or shift in tone against the still)' },
              ]}
            />
            <span className="sc-label" title="Bring the motion out at the area's own resolution, with the still's fine detail">
              Detail
            </span>
            <Seg
              value={o.detail === false ? 0 : 1}
              onChange={(v) => props.onOptions({ detail: v === 1 })}
              options={[
                {
                  value: 1,
                  label: 'Match still',
                  title: 'The engine renders about 1 MP; the video is brought up to the area’s own size and the still’s fine detail is added back where things move gently',
                },
                { value: 0, label: 'Engine', title: 'The engine’s own resolution, scaled up to fit the area' },
              ]}
            />
            <span className="sc-label">Sound</span>
            <Seg
              value={o.sound ? 1 : 0}
              onChange={(v) => props.onOptions({ sound: v === 1 })}
              options={[
                { value: 0, label: 'Silent' },
                { value: 1, label: 'Generate', title: 'Generate ambient sound for this area; the export mixes every layer’s sound' },
              ]}
            />
            {o.sound && (
              <>
                <span className="sc-label" title="What this area should sound like; empty = the natural ambience of what moves">
                  Sounds like
                </span>
                <input
                  className="sc-input"
                  value={o.soundPrompt || ''}
                  placeholder="e.g. crackling fire, soft wind (empty: natural ambience)"
                  onChange={(e) => props.onOptions({ soundPrompt: e.target.value })}
                  aria-label="Sound description"
                />
                <span className="sc-label" title="Whether the sound may include speech or music">
                  Voices
                </span>
                <Seg
                  value={o.voices ? 1 : 0}
                  onChange={(v) => props.onOptions({ voices: v === 1 })}
                  options={[
                    { value: 0, label: 'No speech or music', title: 'Ambience and effects only (the default)' },
                    { value: 1, label: 'Allow', title: 'Speech or music may appear, e.g. when the description asks for it' },
                  ]}
                />
                <span className="sc-label" title="This layer's level in the mix (the preview stops at 100%; the export goes to 200%)">
                  Volume
                </span>
                <div className="sc-slider">
                  <input
                    type="range"
                    min={0}
                    max={200}
                    step={5}
                    value={Math.round(timing.volume * 100)}
                    onChange={(e) => props.onOptions({ volume: Number(e.target.value) / 100 })}
                    aria-label="Volume"
                  />
                  <span>{timing.volume === 0 ? 'muted' : `${Math.round(timing.volume * 100)}%`}</span>
                </div>
              </>
            )}
            <span className="sc-label" title="Play this layer's motion slower or faster">
              Speed
            </span>
            <Seg
              value={timing.speed}
              onChange={(speed) => props.onOptions({ speed })}
              options={[0.5, 0.75, 1, 1.25, 1.5, 2].map((v) => ({ value: v, label: `${v}×` }))}
            />
            <span className="sc-label" title="Start this layer's loop at a different point, so layers don't move in step">
              Offset
            </span>
            <div className="sc-slider">
              <input
                type="range"
                min={0}
                max={offsetMax}
                step={0.05}
                value={Math.min(timing.offset, offsetMax)}
                onChange={(e) => props.onOptions({ offset: Number(e.target.value) })}
                aria-label="Loop offset"
              />
              <span>{timing.offset.toFixed(2)} s</span>
              <button
                type="button"
                className="sc-btn sc-btn-quiet sc-btn-tiny"
                title="A random offset"
                onClick={() => props.onOptions({ offset: Math.round(Math.random() * offsetMax * 20) / 20 })}
              >
                ⚄
              </button>
            </div>
          </>
        ) : (
          <>
            <span className="sc-label">Quality</span>
            <Seg
              value={o.steps || 25}
              onChange={(steps) => props.onOptions({ steps })}
              options={[
                { value: 12, label: 'Draft' },
                { value: 25, label: 'Standard' },
                { value: 40, label: 'Best' },
              ]}
            />
          </>
        )}
        <span className="sc-label">Seed</span>
        <input
          className="sc-input sc-seed"
          inputMode="numeric"
          value={o.seed === undefined || o.seed < 0 ? '' : String(o.seed)}
          placeholder="random"
          onChange={(e) => {
            const v = e.target.value.trim();
            props.onOptions({ seed: v === '' || !/^\d+$/.test(v) ? -1 : Number(v) });
          }}
        />
      </div>

      <div className="sc-generate">
        {running ? (
          <>
            <div className="sc-progress">
              <div className="sc-progress-bar" style={{ width: `${l.job && l.job.total ? (l.job.done / l.job.total) * 100 : 3}%` }} />
            </div>
            <div className="sc-job-line">{l.job?.step === 'scene-lock'
                ? 'Locking the still parts to the picture so the edges stay static…'
                : l.job?.state === 'pending'
                ? 'Waiting for the GPU (one diffusion job at a time)…'
                : l.job?.line || 'Starting…'}</div>
            <button type="button" className="sc-btn" onClick={props.onCancel}>
              Cancel
            </button>
          </>
        ) : (
          <button
            type="button"
            className="sc-btn sc-btn-primary sc-btn-wide"
            disabled={busy || !painted || needPrompt}
            onClick={props.onGenerate}
            title={!painted ? 'Paint or box the area first' : needPrompt ? 'Describe the edit first' : 'Ctrl+Enter'}
          >
            {!painted ? 'Paint the area first' : needPrompt ? 'Describe the edit first' : `${l.versions.length ? 'Generate again' : motion ? 'Animate this area' : 'Generate'}  ⌃↵`}
          </button>
        )}
        {l.error && <div className="sc-error">{l.error}</div>}
        {outside && !running && <div className="sc-warn">The mask now reaches past the area this version was made for: generate again to fill it.</div>}
      </div>

      {l.versions.length > 0 && (
        <div className="sc-versions">
          <div className="sc-label">Versions</div>
          <div className="sc-version-chips">
            {l.versions.map((v, i) => (
              // eslint-disable-next-line react/no-array-index-key
              <div key={`${v.path}-${i}`} className={`sc-version${i === l.active ? ' on' : ''}`}>
                <button
                  type="button"
                  className="sc-version-pick"
                  title={`${v.note ? `${v.note} · ` : ''}${v.prompt || '(ambient motion)'}`}
                  onClick={() => props.onPatch({ active: i })}
                >
                  {motion ? <video src={mediaUrl(v.path)} muted preload="metadata" /> : <img src={mediaUrl(v.path)} alt="" loading="lazy" />}
                  <span>v{i + 1}</span>
                </button>
                <DiscardButton label={`v${i + 1}`} onDiscard={() => props.onDeleteVersion(i)} />
              </div>
            ))}
            <button type="button" className={`sc-version none${l.active < 0 ? ' on' : ''}`} onClick={() => props.onPatch({ active: -1 })} title="Show no version (the picture below)">
              <span>off</span>
            </button>
          </div>
          {version && (
            <>
              <div className="sc-version-actions">
                <span className="sc-faint sc-version-prompt">
                  {version.note || version.prompt || 'ambient motion'}
                  {motion && Math.abs(props.loopSpeed - 1) > 0.005 ? ` · plays at ${(100 / props.loopSpeed).toFixed(0)}% speed to loop with the scene` : ''}
                </span>
                {l.versions.length > 1 && (
                  <button type="button" className="sc-btn sc-btn-quiet" disabled={busy} onClick={() => props.onKeepOnly(l.active)} title="Discard every other version (deletes their files)">
                    Keep only v{l.active + 1}
                  </button>
                )}
              </div>
              {motion && (
                <>
                  <div className="sc-version-actions">
                    <span className="sc-label">Loop</span>
                    <button type="button" className="sc-btn" disabled={busy} onClick={() => props.onLoop(l.active, 'crossfade')} title="The end dissolves into the start: for things that move one way (water, smoke, fire, clouds)">
                      Crossfade
                    </button>
                    <button type="button" className="sc-btn" disabled={busy} onClick={() => props.onLoop(l.active, 'natural')} title="Find the two moments that look most alike and loop between them">
                      Natural
                    </button>
                    <button type="button" className="sc-btn" disabled={busy} onClick={() => props.onLoop(l.active, 'boomerang')} title="Forward then backward: for back-and-forth motion (swaying, breathing, flicker)">
                      Boomerang
                    </button>
                  </div>
                  <div className="sc-version-actions">
                    <span className="sc-label">Motion</span>
                    <button type="button" className={`sc-btn${props.motionShown ? ' on' : ''}`} disabled={busy} onClick={props.onShowMotion} title="Tint where this version actually moves">
                      Show
                    </button>
                    <button type="button" className="sc-btn" disabled={busy} onClick={props.onGrowToMotion} title="Add everywhere this version moves to the mask, so no motion is cut off at its edge">
                      Grow mask to motion
                    </button>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      <div className="sc-layer-foot">
        <span>
          <button type="button" className="sc-btn sc-btn-quiet" disabled={!painted} onClick={props.onClearMask}>
            Clear mask
          </button>
          {(l.shapes || []).some((op) => op.keep) && (
            <button type="button" className="sc-btn sc-btn-quiet" onClick={props.onClearProtect}>
              Clear protect
            </button>
          )}
        </span>
        {confirmDelete ? (
          <span className="sc-confirm">
            Delete layer{l.versions.length ? ` and ${l.versions.length} version${l.versions.length > 1 ? 's' : ''}` : ''}?
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
