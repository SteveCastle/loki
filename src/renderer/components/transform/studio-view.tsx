import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  INTENTS,
  MAX_DURATION,
  MAX_VARIATIONS,
  MIN_DURATION,
  availability,
  classify,
  estimateSeconds,
  formatDuration,
  intentById,
  mediaKind,
  parseSize,
  planFor,
  retouchPreset,
  type IntentId,
  type Quality,
  type Shake,
  type TransformSettings,
} from './intents';
import type { StudioPhase } from './store';
import { GlyphCheck, GlyphDie, GlyphMusic, IntentIcon } from './icons';
import './transform-studio.css';

// ---------------------------------------------------------------------------
// Small controls
// ---------------------------------------------------------------------------

interface SegOption<T extends string | number> {
  value: T;
  label: string;
  sub?: string;
}

function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: Array<SegOption<T>>;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="ts-seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={`ts-seg-opt${o.value === value ? ' active' : ''}`}
          onClick={() => onChange(o.value)}
        >
          <span>{o.label}</span>
          {o.sub ? <small>{o.sub}</small> : null}
        </button>
      ))}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="ts-field">
      <div className="ts-field-head">
        <span className="ts-field-label">{label}</span>
        {hint ? <span className="ts-field-hint">{hint}</span> : null}
      </div>
      {children}
    </div>
  );
}

// Style chips that append a phrase to the prompt (edit intents).
const STYLE_CHIPS: Array<{ label: string; text: string }> = [
  { label: 'Golden hour', text: 'lit by warm golden-hour sunlight' },
  { label: 'Night', text: 'at night with warm practical lights' },
  { label: 'Snow', text: 'in a snowy winter setting' },
  { label: 'Rain', text: 'in the rain with wet reflective surfaces' },
  { label: 'Fog', text: 'in soft fog and haze' },
  { label: 'Black & white', text: 'as a high-contrast black and white photograph' },
  { label: 'Film grain', text: 'with subtle analog film grain' },
  { label: 'Cinematic', text: 'with a cinematic color grade' },
  { label: 'Remove people', text: 'remove the people and fill in the background' },
];

function appendPhrase(prompt: string, phrase: string): string {
  const base = prompt.trim();
  if (!base) return phrase.charAt(0).toUpperCase() + phrase.slice(1);
  if (base.toLowerCase().includes(phrase.toLowerCase())) return prompt;
  return `${base.replace(/[.,;]+$/, '')}, ${phrase}`;
}

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

// ---------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------

function Thumb({
  path,
  role,
  thumbUrl,
  onRemove,
}: {
  path: string;
  role?: string;
  thumbUrl: (p: string) => string;
  onRemove?: () => void;
}) {
  const kind = mediaKind(path);
  return (
    <div className={`ts-thumb ts-thumb-${kind}`} title={baseName(path)}>
      {kind === 'image' && <img src={thumbUrl(path)} alt="" draggable={false} />}
      {kind === 'video' && (
        <video src={`${thumbUrl(path)}#t=0.1`} preload="metadata" muted playsInline />
      )}
      {kind === 'audio' && (
        <span className="ts-thumb-glyph">
          <GlyphMusic />
        </span>
      )}
      {kind === 'other' && <span className="ts-thumb-glyph">?</span>}
      {role ? <span className="ts-thumb-role">{role}</span> : null}
      {onRemove ? (
        <button type="button" className="ts-thumb-x" onClick={onRemove} aria-label={`Remove ${baseName(path)}`}>
          ×
        </button>
      ) : null}
    </div>
  );
}

/** Role tags the model sees: <image1>… for retouch, <Picture 1>/<Video 1>/<Audio 1> for reshoot. */
export function roleLabels(intent: IntentId | null, paths: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  if (!intent) return out;
  const inputs = classify(paths);
  if (intentById(intent).engine === 'retouch') {
    (inputs.images.length ? inputs.images : inputs.videos.slice(0, 1)).forEach((p, i) => {
      out[p] = intent === 'combine' ? `<image${i + 1}>` : '';
    });
    return out;
  }
  inputs.images.forEach((p, i) => {
    out[p] = intent === 'alive' ? 'photo' : `<Picture ${i + 1}>`;
  });
  inputs.videos.forEach((p, i) => {
    out[p] = `<Video ${i + 1}>`;
  });
  inputs.audios.forEach((p, i) => {
    out[p] = `<Audio ${i + 1}>`;
  });
  return out;
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

export interface StudioViewProps {
  paths: string[];
  intent: IntentId | null;
  phase: StudioPhase;
  settings: TransformSettings;
  source: { width: number; height: number } | null;
  videoTime?: number;
  queueAhead: number;
  status: 'idle' | 'submitting' | 'done' | 'error';
  error?: string;
  queuedCount?: number;
  thumbUrl: (p: string) => string;
  onSettings: (patch: Partial<TransformSettings>) => void;
  onIntent: (id: IntentId) => void;
  onPhase: (p: StudioPhase) => void;
  onRun: () => void;
  onClose: () => void;
  onRemovePath: (p: string) => void;
  onAgain: () => void;
}

const PHASES: Array<{ id: StudioPhase; label: string }> = [
  { id: 'choose', label: 'Choose' },
  { id: 'shape', label: 'Shape' },
  { id: 'review', label: 'Review' },
];

function summarizeInputs(paths: string[]): string {
  const c = classify(paths);
  const parts: string[] = [];
  if (c.images.length) parts.push(`${c.images.length} image${c.images.length > 1 ? 's' : ''}`);
  if (c.videos.length) parts.push(`${c.videos.length} video${c.videos.length > 1 ? 's' : ''}`);
  if (c.audios.length) parts.push(`${c.audios.length} audio`);
  return parts.join(' · ') || 'nothing selected';
}

export function TransformStudioView(props: StudioViewProps) {
  const { paths, intent, phase, status } = props;
  const inputs = useMemo(() => classify(paths), [paths]);
  const def = intent ? intentById(intent) : null;

  const phaseIndex = PHASES.findIndex((p) => p.id === phase);
  const canGoShape = !!intent;
  const canGoReview = !!intent && validateForRun(intent, props.settings, paths) === null;

  return (
    <div className="ts-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div
        className="ts-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Transform"
        data-phase={phase}
        onKeyDown={(e) => e.stopPropagation()}
        onKeyUp={(e) => e.stopPropagation()}
      >
        <header className="ts-header">
          <div className="ts-title">
            <span className="ts-title-main">Transform</span>
            <span className="ts-title-sub">{summarizeInputs(paths)}</span>
          </div>
          <ol className="ts-stepper" aria-label="Steps">
            {PHASES.map((p, i) => {
              const reachable = p.id === 'choose' || (p.id === 'shape' && canGoShape) || (p.id === 'review' && canGoReview);
              return (
                <li key={p.id} className={`ts-step${i === phaseIndex ? ' current' : ''}${i < phaseIndex ? ' done' : ''}`}>
                  <button
                    type="button"
                    disabled={!reachable || status === 'submitting' || status === 'done'}
                    onClick={() => props.onPhase(p.id)}
                  >
                    <span className="ts-step-n">{i < phaseIndex ? '✓' : i + 1}</span>
                    <span className="ts-step-label">{p.label}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          <button type="button" className="ts-close" onClick={props.onClose} aria-label="Close">
            ×
          </button>
        </header>

        <div className="ts-body">
          {status === 'done' ? (
            <DoneView {...props} />
          ) : phase === 'choose' ? (
            <ChooseView {...props} inputs={inputs} />
          ) : phase === 'shape' && def ? (
            <ShapeView {...props} def={def} />
          ) : def && phase === 'review' ? (
            <ReviewView {...props} def={def} />
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ---- Phase 1: choose -------------------------------------------------------

function ChooseView(props: StudioViewProps & { inputs: ReturnType<typeof classify> }) {
  const { inputs, intent } = props;
  const groups: Array<{ id: 'image' | 'video'; title: string; sub: string }> = [
    { id: 'image', title: 'Images', sub: 'loki-retouch · Qwen Image 2.1' },
    { id: 'video', title: 'Video', sub: 'loki-reshoot · MiniMax H3, with sound' },
  ];
  return (
    <div className="ts-choose">
      <p className="ts-lead">What do you want to do with {summarizeInputs(props.paths)}?</p>
      {groups.map((g) => (
        <section key={g.id} className="ts-group">
          <div className="ts-group-head">
            <h3>{g.title}</h3>
            <span>{g.sub}</span>
          </div>
          <div className="ts-cards">
            {INTENTS.filter((i) => i.group === g.id).map((i) => {
              const a = availability(i.id, inputs);
              return (
                <button
                  key={i.id}
                  type="button"
                  className={`ts-card${intent === i.id ? ' selected' : ''}${a.ok ? '' : ' disabled'}`}
                  disabled={!a.ok}
                  onClick={() => props.onIntent(i.id)}
                  data-intent={i.id}
                >
                  <span className="ts-card-icon">
                    <IntentIcon id={i.id} />
                  </span>
                  <span className="ts-card-text">
                    <strong>{i.title}</strong>
                    <small>{a.ok ? i.tagline : a.reason}</small>
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

// ---- Phase 2: shape --------------------------------------------------------

const SCALE_OPTIONS: Array<SegOption<number>> = [
  { value: 1.5, label: '1.5×' },
  { value: 2, label: '2×' },
  { value: 3, label: '3×' },
  { value: 4, label: '4×' },
];

const QUALITY_OPTIONS = (engine: 'retouch' | 'reshoot'): Array<SegOption<Quality>> => [
  { value: 'draft', label: 'Draft', sub: 'fast' },
  { value: 'balanced', label: 'Balanced', sub: engine === 'retouch' ? '25 steps' : '20 steps' },
  { value: 'fine', label: 'Fine', sub: 'slow' },
];

function ShapeView(props: StudioViewProps & { def: ReturnType<typeof intentById> }) {
  const { def, settings: s, onSettings, source, paths } = props;
  const inputs = classify(paths);
  const [more, setMore] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const t = window.setTimeout(() => textRef.current?.focus(), 60);
    return () => window.clearTimeout(t);
  }, [def.id]);

  const plan = planFor(def.id, s, paths, source);
  const eta = estimateSeconds(plan);
  const isRetouch = def.engine === 'retouch';
  const roles = roleLabels(def.id, paths);
  const shown = Object.keys(roles).length ? Object.keys(roles) : paths;
  const stills = inputs.images.length;
  const sizeOptions: Array<SegOption<string>> = [
    ...(def.id === 'upscale' ? [] : [{ value: 'same', label: 'Same' }]),
    ...SCALE_OPTIONS.map((o) => ({ value: String(o.value), label: o.label })),
    { value: 'custom', label: 'Custom' },
  ];
  const sizeValue = s.sizeMode === 'same' ? 'same' : s.sizeMode === 'custom' ? 'custom' : String(s.scale);
  const customParts = parseSize(s.customSize) || [0, 0];

  return (
    <div className="ts-shape">
      <div className="ts-controls">
        <div className="ts-intent-head">
          <span className="ts-card-icon">
            <IntentIcon id={def.id} />
          </span>
          <div>
            <h2>{def.title}</h2>
            <p>{def.blurb}</p>
          </div>
        </div>

        {def.needsText === 'prompt' && (
          <Field label={def.id === 'direct' ? 'Describe the scene' : def.id === 'combine' ? 'How should they combine?' : 'What should change?'} hint="Enter ▸ review">
            <textarea
              ref={textRef}
              className="ts-text"
              rows={3}
              value={s.prompt}
              placeholder={def.examples[0]}
              onChange={(e) => onSettings({ prompt: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
                  e.preventDefault();
                  props.onPhase('review');
                }
              }}
            />
            <div className="ts-chips">
              {def.examples.slice(0, 3).map((ex) => (
                <button key={ex} type="button" className="ts-chip ts-chip-idea" title={ex} onClick={() => onSettings({ prompt: ex })}>
                  {ex.length > 46 ? `${ex.slice(0, 44)}…` : ex}
                </button>
              ))}
              {def.id === 'edit' &&
                STYLE_CHIPS.map((c) => (
                  <button key={c.label} type="button" className="ts-chip" onClick={() => onSettings({ prompt: appendPhrase(s.prompt, c.text) })}>
                    + {c.label}
                  </button>
                ))}
            </div>
            {def.id === 'direct' && (
              <p className="ts-note">
                Refer to references with <code>&lt;Picture 1&gt;</code>, <code>&lt;Video 1&gt;</code>, <code>&lt;Audio 1&gt;</code>; say which one drives what.
              </p>
            )}
          </Field>
        )}

        {def.needsText === 'describe' && (
          <>
            <Field label="Who or what is in the photo?" hint="keeps identity stable">
              <textarea
                ref={textRef}
                className="ts-text"
                rows={2}
                value={s.describe}
                placeholder={def.examples[0]}
                onChange={(e) => onSettings({ describe: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    props.onPhase('review');
                  }
                }}
              />
              <div className="ts-chips">
                {def.examples.map((ex) => (
                  <button key={ex} type="button" className="ts-chip ts-chip-idea" onClick={() => onSettings({ describe: ex })}>
                    {ex.length > 40 ? `${ex.slice(0, 38)}…` : ex}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Camera">
              <Segmented<Shake>
                label="Camera shake"
                value={s.shake}
                onChange={(v) => onSettings({ shake: v })}
                options={[
                  { value: 'none', label: 'Locked', sub: 'tripod' },
                  { value: 'subtle', label: 'Subtle', sub: 'handheld sway' },
                  { value: 'handheld', label: 'Handheld', sub: 'visible' },
                ]}
              />
            </Field>
          </>
        )}

        {isRetouch && def.id !== 'wallpaper' && (
          <Field label="Output size" hint={plan.outWidth ? `${plan.outWidth}×${plan.outHeight}` : undefined}>
            <Segmented<string>
              label="Output size"
              value={sizeValue}
              options={sizeOptions}
              onChange={(v) => {
                if (v === 'same') onSettings({ sizeMode: 'same' });
                else if (v === 'custom') onSettings({ sizeMode: 'custom', customSize: s.customSize || (source ? `${source.width}x${source.height}` : '') });
                else onSettings({ sizeMode: 'scale', scale: parseFloat(v) });
              }}
            />
            {s.sizeMode === 'custom' && (
              <div className="ts-size-inputs">
                <input type="number" min={16} value={customParts[0] || ''} placeholder="width" onChange={(e) => onSettings({ customSize: `${e.target.value}x${customParts[1] || ''}` })} />
                <span>×</span>
                <input type="number" min={16} value={customParts[1] || ''} placeholder="height" onChange={(e) => onSettings({ customSize: `${customParts[0] || ''}x${e.target.value}` })} />
                <small>any size: the model works in multiples of 16 and the result is resampled to exactly this</small>
              </div>
            )}
          </Field>
        )}

        {def.id === 'wallpaper' && (
          <Field label="Screen">
            <Segmented<'desktop' | 'phone'>
              label="Wallpaper target"
              value={s.wallpaperTarget}
              onChange={(v) => onSettings({ wallpaperTarget: v })}
              options={[
                { value: 'desktop', label: 'Desktop', sub: '3840×2160' },
                { value: 'phone', label: 'Phone', sub: '1296×2800' },
              ]}
            />
          </Field>
        )}

        {!isRetouch && (
          <Field label="Length" hint={`${s.duration.toFixed(1)} s · ${plan.frames} frames`}>
            <input
              className="ts-range"
              type="range"
              min={MIN_DURATION}
              max={MAX_DURATION}
              step={0.5}
              value={s.duration}
              onChange={(e) => onSettings({ duration: parseFloat(e.target.value) })}
              style={{ ['--ts-fill' as string]: `${((s.duration - MIN_DURATION) / (MAX_DURATION - MIN_DURATION)) * 100}%` }}
              aria-label="Duration in seconds"
            />
            <div className="ts-range-ticks">
              <span>1 s</span>
              <span className={(plan.frames || 124) < 124 ? 'warn' : 'good'}>5 s+ is best</span>
              <span>15 s</span>
            </div>
            {(plan.frames || 124) < 124 && <p className="ts-note warn">Shorter than the model was trained on (5–15 s): it runs, quality may suffer.</p>}
          </Field>
        )}

        <Field label="Quality">
          <Segmented<Quality> label="Quality" value={s.quality} options={QUALITY_OPTIONS(def.engine)} onChange={(v) => onSettings({ quality: v })} />
        </Field>

        <div className="ts-row2">
          <Field label="Variations" hint={s.variations > 1 ? `${s.variations} results` : undefined}>
            <div className="ts-stepper-num">
              <button type="button" onClick={() => onSettings({ variations: Math.max(1, s.variations - 1) })} aria-label="Fewer variations">
                −
              </button>
              <span>{s.variations}</span>
              <button type="button" onClick={() => onSettings({ variations: Math.min(MAX_VARIATIONS, s.variations + 1) })} aria-label="More variations">
                +
              </button>
            </div>
          </Field>
          <Field label="Seed">
            <div className="ts-seed">
              <button
                type="button"
                className={`ts-seed-mode${s.seedMode === 'random' ? ' active' : ''}`}
                onClick={() => onSettings({ seedMode: 'random' })}
                title="A different take every time"
              >
                <GlyphDie /> Random
              </button>
              <input
                type="number"
                min={0}
                value={s.seedMode === 'fixed' ? s.seed : ''}
                placeholder="fixed #"
                onFocus={() => s.seedMode !== 'fixed' && onSettings({ seedMode: 'fixed' })}
                onChange={(e) => onSettings({ seedMode: 'fixed', seed: Math.max(0, parseInt(e.target.value || '0', 10) || 0) })}
                aria-label="Fixed seed"
              />
            </div>
          </Field>
        </div>

        <button type="button" className="ts-more-toggle" onClick={() => setMore((m) => !m)} aria-expanded={more}>
          {more ? '▾' : '▸'} More options
        </button>
        {more && (
          <div className="ts-more">
            {isRetouch && def.id !== 'edit' && def.id !== 'combine' && (
              <Field label="Extra direction" hint="appended to the built-in prompt">
                <textarea className="ts-text" rows={2} value={s.append} placeholder="e.g. keep the film grain" onChange={(e) => onSettings({ append: e.target.value })} />
              </Field>
            )}
            {!isRetouch && (
              <>
                <Field label="Reference image detail">
                  <Segmented<'match' | 'max'>
                    label="Reference image size"
                    value={s.refSize}
                    onChange={(v) => onSettings({ refSize: v })}
                    options={[
                      { value: 'match', label: 'Match output', sub: 'faster' },
                      { value: 'max', label: 'Maximum', sub: 'best identity, slower' },
                    ]}
                  />
                </Field>
                <label className="ts-check">
                  <input type="checkbox" checked={!s.noAudio} onChange={(e) => onSettings({ noAudio: !e.target.checked })} />
                  Generate sound
                </label>
                {inputs.videos.length > 0 && (
                  <label className="ts-check">
                    <input type="checkbox" checked={s.keepSoundtrack} onChange={(e) => onSettings({ keepSoundtrack: e.target.checked })} />
                    Use the soundtracks of reference clips
                  </label>
                )}
              </>
            )}
            {isRetouch && stills > 1 && def.id !== 'combine' && (
              <p className="ts-note">{stills} images → {stills} separate results (use “Combine” to merge them into one).</p>
            )}
          </div>
        )}
      </div>

      <aside className="ts-preview" aria-label="Preview">
        <div className="ts-preview-card">
          <div className="ts-preview-title">You’ll get</div>
          <div className="ts-preview-file">{plan.outputName}</div>
          <div className="ts-preview-meta">
            {plan.outWidth ? <span>{plan.outWidth}×{plan.outHeight}</span> : null}
            {source && isRetouch && plan.outWidth && (plan.outWidth !== source.width || plan.outHeight !== source.height) ? (
              <span className="muted">from {source.width}×{source.height}</span>
            ) : null}
            {!isRetouch ? <span>{plan.frames} frames · {s.duration.toFixed(1)} s</span> : null}
          </div>
          <div className="ts-preview-eta">
            <span>≈ {formatDuration(eta)}</span>
            <small>{plan.outputs > 1 ? `each · ${plan.outputs} results` : 'on the GPU'}</small>
          </div>
        </div>
        <div className="ts-preview-inputs">
          <div className="ts-preview-title">Using</div>
          <div className="ts-thumbs">
            {shown.slice(0, 8).map((p) => (
              <Thumb key={p} path={p} role={roles[p]} thumbUrl={props.thumbUrl} onRemove={paths.length > 1 ? () => props.onRemovePath(p) : undefined} />
            ))}
            {shown.length > 8 ? <div className="ts-thumb ts-thumb-more">+{shown.length - 8}</div> : null}
          </div>
        </div>
        <ShapeFooter {...props} />
      </aside>
    </div>
  );
}

function ShapeFooter(props: StudioViewProps) {
  const err = props.intent ? validateForRun(props.intent, props.settings, props.paths) : 'Pick something to do';
  return (
    <div className="ts-footer">
      <button type="button" className="ts-btn ghost" onClick={() => props.onPhase('choose')}>
        ← Back
      </button>
      <button type="button" className="ts-btn primary" disabled={!!err} onClick={() => props.onPhase('review')} title={err || undefined}>
        Review →
      </button>
      {err ? <span className="ts-footer-note">{err}</span> : null}
    </div>
  );
}

// ---- Phase 3: review -------------------------------------------------------

function ReviewView(props: StudioViewProps & { def: ReturnType<typeof intentById> }) {
  const { def, settings: s, paths, source, status } = props;
  const plan = planFor(def.id, s, paths, source);
  const eta = estimateSeconds(plan);
  const roles = roleLabels(def.id, paths);
  const shown = Object.keys(roles).length ? Object.keys(roles) : paths;
  const preset = def.engine === 'retouch' ? retouchPreset(def.id, s) : '';
  const instruction =
    def.id === 'alive'
      ? s.describe.trim() || '(no description — add one to help keep identity)'
      : s.prompt.trim() || (preset ? `Built-in “${preset}” prompt${s.append.trim() ? ` + “${s.append.trim()}”` : ''}` : '');
  const total = eta * plan.outputs;
  const busy = status === 'submitting';
  return (
    <div className="ts-review">
      <div className="ts-review-main">
        <div className="ts-intent-head">
          <span className="ts-card-icon">
            <IntentIcon id={def.id} />
          </span>
          <div>
            <h2>
              {def.title}
              {plan.outputs > 1 ? ` × ${plan.outputs}` : ''}
            </h2>
            <p>{def.tagline}</p>
          </div>
        </div>
        <dl className="ts-summary">
          <dt>{def.id === 'alive' ? 'Subject' : 'Instruction'}</dt>
          <dd className="ts-quote">{instruction}</dd>
          <dt>Result</dt>
          <dd>
            <strong>{plan.outputName}</strong>
            {plan.outWidth ? ` · ${plan.outWidth}×${plan.outHeight}` : ''}
            {def.engine === 'reshoot' ? ` · ${s.duration.toFixed(1)} s with sound` : ''}
            <span className="muted"> — saved beside the original{plan.outputs > 1 ? 's' : ''}</span>
          </dd>
          <dt>Settings</dt>
          <dd>
            {s.quality} quality ({plan.steps} steps) · {s.seedMode === 'fixed' ? `seed ${s.seed}` : 'random seed'}
            {s.variations > 1 ? ` · ${s.variations} variations` : ''}
            {def.id === 'alive' ? ` · camera ${s.shake === 'none' ? 'locked' : s.shake}` : ''}
          </dd>
          <dt>Time</dt>
          <dd>
            ≈ {formatDuration(total)}
            {plan.outputs > 1 ? ` in total (${plan.outputs} results, one at a time)` : ''}
            {props.queueAhead > 0 ? <span className="warn"> · {props.queueAhead} GPU job{props.queueAhead > 1 ? 's' : ''} already queued ahead</span> : null}
          </dd>
        </dl>
        {props.error ? <p className="ts-error">{props.error}</p> : null}
      </div>
      <aside className="ts-review-side">
        <div className="ts-preview-title">Inputs</div>
        <div className="ts-thumbs big">
          {shown.slice(0, 6).map((p) => (
            <Thumb key={p} path={p} role={roles[p]} thumbUrl={props.thumbUrl} />
          ))}
          {shown.length > 6 ? <div className="ts-thumb ts-thumb-more">+{shown.length - 6}</div> : null}
        </div>
        <p className="ts-note">Uses the local GPU engine <code>{def.engine === 'retouch' ? 'loki-retouch' : 'loki-reshoot'}</code>. Nothing leaves your machine.</p>
        <div className="ts-footer">
          <button type="button" className="ts-btn ghost" onClick={() => props.onPhase('shape')} disabled={busy}>
            ← Adjust
          </button>
          <button type="button" className="ts-btn primary big" onClick={props.onRun} disabled={busy} autoFocus>
            {busy ? 'Queuing…' : `Queue ${plan.jobCount > 1 ? `${plan.jobCount} jobs` : 'it'}`}
            <kbd>Ctrl ⏎</kbd>
          </button>
        </div>
      </aside>
    </div>
  );
}

function DoneView(props: StudioViewProps) {
  const n = props.queuedCount || 1;
  return (
    <div className="ts-done">
      <span className="ts-done-badge">
        <GlyphCheck />
      </span>
      <h2>{n > 1 ? `${n} jobs queued` : 'Queued'}</h2>
      <p>
        The results appear next to the originals when they finish, and the library jumps to the first new file. Follow progress in the job queue.
      </p>
      <div className="ts-footer center">
        <button type="button" className="ts-btn ghost" onClick={props.onAgain}>
          Tweak and run again
        </button>
        <button type="button" className="ts-btn primary" onClick={props.onClose} autoFocus>
          Done
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export function validateForRun(id: IntentId, s: TransformSettings, paths: string[]): string | null {
  const a = availability(id, classify(paths));
  if (!a.ok) return a.reason || 'Not available for this selection';
  const def = intentById(id);
  if (def.needsText === 'prompt' && !s.prompt.trim()) return def.id === 'direct' ? 'Describe the scene to continue' : def.id === 'combine' ? 'Describe how to combine them' : 'Describe what should change';
  if (s.sizeMode === 'custom' && def.engine === 'retouch' && def.id !== 'wallpaper' && !parseSize(s.customSize)) return 'Enter a width and height';
  return null;
}
