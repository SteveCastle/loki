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
  promptIsVerbatim,
  referenceTokens,
  retouchPreset,
  tokenIssues,
  usesReferences,
  type IntentId,
  type Quality,
  type Shake,
  type TransformSettings,
} from './intents';
import type { FlowPhase } from './store';
import { GlyphCheck, GlyphDie, GlyphMusic, IntentIcon } from './icons';
import './transform-flow.css';

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
  onPick,
}: {
  path: string;
  role?: string;
  thumbUrl: (p: string) => string;
  onRemove?: () => void;
  /** Makes the thumbnail a button (used to put its prompt token into the prompt). */
  onPick?: () => void;
}) {
  const kind = mediaKind(path);
  const pick = onPick
    ? {
        role: 'button' as const,
        tabIndex: 0,
        onClick: onPick,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onPick();
          }
        },
      }
    : {};
  return (
    <div
      className={`ts-thumb ts-thumb-${kind}${onPick ? ' ts-thumb-pick' : ''}`}
      title={onPick && role ? `${baseName(path)}: click to put ${role} in the prompt` : baseName(path)}
      {...pick}
    >
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
        <button
          type="button"
          className="ts-thumb-x"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove ${baseName(path)}`}
        >
          ×
        </button>
      ) : null}
    </div>
  );
}

/** Put `token` into `text` at the caret / over the selection, with a space on each side where it would touch a word. */
export function insertToken(text: string, token: string, start: number, end: number): { text: string; caret: number } {
  const s = Math.max(0, Math.min(start, text.length));
  const e = Math.max(s, Math.min(end, text.length));
  const before = text.slice(0, s);
  const after = text.slice(e);
  const lead = before && !/\s$/.test(before) ? ' ' : '';
  const trail = after && !/^\s/.test(after) ? ' ' : after ? '' : ' ';
  const out = `${before}${lead}${token}${trail}${after}`;
  return { text: out, caret: before.length + lead.length + token.length + trail.length };
}

/** A prompt textarea's "insert this token at the caret" action (keeps focus and puts the caret after it). */
function useTokenInsert(ref: React.RefObject<HTMLTextAreaElement>, value: string, setValue: (v: string) => void) {
  return (token: string) => {
    const ta = ref.current;
    const at = ta && document.activeElement === ta ? [ta.selectionStart, ta.selectionEnd] : [value.length, value.length];
    const r = insertToken(value, token, at[0] ?? value.length, at[1] ?? value.length);
    setValue(r.text);
    window.setTimeout(() => {
      ta?.focus();
      ta?.setSelectionRange(r.caret, r.caret);
    }, 0);
  };
}

export interface RefItem {
  token: string;
  path: string;
}

/** The reference tokens of a prompt as chips (thumbnail · token · file name); clicking one inserts its token. */
function ReferenceChips({
  refs,
  thumbUrl,
  onPick,
  onMakeFirst,
}: {
  refs: RefItem[];
  thumbUrl: (p: string) => string;
  onPick?: (token: string) => void;
  /** Offered on every chip but the first: make that file <image1>. */
  onMakeFirst?: (path: string) => void;
}) {
  return (
    <div className="ts-refs">
      {refs.map((r, i) => {
        const kind = mediaKind(r.path);
        return (
          <div key={`${r.token}|${r.path}`} className="ts-ref">
            <button
              type="button"
              className="ts-ref-token"
              disabled={!onPick}
              onClick={() => onPick?.(r.token)}
              title={onPick ? `Insert ${r.token} (${baseName(r.path)}) into the prompt` : `${r.token} is ${baseName(r.path)}`}
            >
              <span className="ts-ref-thumb">
                {kind === 'image' && <img src={thumbUrl(r.path)} alt="" draggable={false} />}
                {kind === 'video' && <video src={`${thumbUrl(r.path)}#t=0.1`} preload="metadata" muted playsInline />}
                {kind === 'audio' && <GlyphMusic />}
              </span>
              <code>{r.token}</code>
              <span className="ts-ref-name">{baseName(r.path)}</span>
            </button>
            {onMakeFirst && i > 0 ? (
              <button type="button" className="ts-ref-main" onClick={() => onMakeFirst(r.path)} title={`Edit ${baseName(r.path)} instead (it becomes ${refs[0].token}; the prompt's tokens follow their files)`}>
                make {refs[0].token}
              </button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

/** Token chips for the selection (deduplicated: a per-image batch shows its one <image1>). */
function refItems(intent: IntentId, paths: string[], manual: boolean): RefItem[] {
  const tokens = referenceTokens(intent, paths, manual);
  const seen = new Set<string>();
  const out: RefItem[] = [];
  Object.keys(tokens).forEach((p) => {
    const t = tokens[p];
    if (!t.startsWith('<') || seen.has(t)) return;
    seen.add(t);
    out.push({ token: t, path: p });
  });
  return out;
}

/** Warnings about the tokens a prompt uses (unknown = blocks the run, unused = a hint). */
function TokenWarnings({ prompt, refs, shared }: { prompt: string; refs: RefItem[]; shared: boolean }) {
  const issues = tokenIssues(prompt, refs.map((r) => r.token));
  return (
    <>
      {issues.unknown.length > 0 && (
        <p className="ts-note warn">
          {issues.unknown.join(', ')} {issues.unknown.length > 1 ? 'match' : 'matches'} none of the selected files.
        </p>
      )}
      {shared && prompt.trim() && issues.unused.length > 0 && (
        <p className="ts-note">
          The prompt doesn’t mention {issues.unused.join(', ')} yet. The model still sees {issues.unused.length > 1 ? 'them' : 'it'}, but naming each reference
          says what to take from it.
        </p>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

export interface FlowViewProps {
  paths: string[];
  intent: IntentId | null;
  phase: FlowPhase;
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
  onPhase: (p: FlowPhase) => void;
  onRun: () => void;
  onClose: () => void;
  onRemovePath: (p: string) => void;
  /** Make this file the first input (<image1>, the one an edit changes). */
  onMakeFirst?: (p: string) => void;
  onAgain: () => void;
  /** The engine-expanded prompt, fetched for the review step. */
  preview?: PromptPreviewState;
}

/** POST /api/transform/prompt's answer (media-server/tasks/prompt_preview.go). */
export interface PromptPreviewData {
  engine: string;
  prompt: string;
  modelInput?: string;
  references: Array<RefItem & { kind: string }>;
  runs: number;
  fromEngine: boolean;
  note?: string;
}

export interface PromptPreviewState {
  status: 'idle' | 'loading' | 'ok' | 'error';
  data?: PromptPreviewData;
  error?: string;
}

const PHASES: Array<{ id: FlowPhase; label: string }> = [
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

export function TransformFlowView(props: FlowViewProps) {
  const { paths, intent, phase, status } = props;
  const inputs = useMemo(() => classify(paths), [paths]);
  const def = intent ? intentById(intent) : null;

  const phaseIndex = PHASES.findIndex((p) => p.id === phase);
  const canGoShape = !!intent;
  const canGoReview = !!intent && validateShape(intent, props.settings, paths) === null;

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

function ChooseView(props: FlowViewProps & { inputs: ReturnType<typeof classify> }) {
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
                    <small>
                      {!a.ok ? a.reason : usesReferences(i.id, inputs) ? `Uses all ${inputs.images.length} images as references` : i.tagline}
                    </small>
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

function ShapeView(props: FlowViewProps & { def: ReturnType<typeof intentById> }) {
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
  const manual = s.manualPrompt !== null;
  const roles = referenceTokens(def.id, paths, manual);
  const shown = Object.keys(roles).length ? Object.keys(roles) : paths;
  const stills = inputs.images.length;
  const shared = usesReferences(def.id, inputs);
  const refs = refItems(def.id, paths, manual);
  const examples = shared && def.referenceExamples ? def.referenceExamples : def.examples;
  const sizeOptions: Array<SegOption<string>> = [
    ...(def.id === 'upscale' ? [] : [{ value: 'same', label: 'Same' }]),
    ...SCALE_OPTIONS.map((o) => ({ value: String(o.value), label: o.label })),
    { value: 'custom', label: 'Custom' },
  ];
  const sizeValue = s.sizeMode === 'same' ? 'same' : s.sizeMode === 'custom' ? 'custom' : String(s.scale);
  // The raw "WxH" text, not parseSize(): a half-typed "1x" is not a size yet, and re-deriving the inputs from
  // it would blank the field under the user's fingers.
  const [customW = '', customH = ''] = (s.customSize || '').split('x');
  const promptField: 'prompt' | null = def.needsText === 'prompt' ? 'prompt' : null;
  const insertPromptToken = useTokenInsert(textRef, s.prompt, (v) => onSettings({ prompt: v }));
  const pickToken = (token: string) => {
    if (promptField) insertPromptToken(token);
  };

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
          <Field
            label={def.id === 'direct' ? 'Describe the scene' : shared ? `What should change? Refer to the ${stills} images by their tokens` : 'What should change?'}
            hint="sent as written · Enter ▸ review"
          >
            <textarea
              ref={textRef}
              className="ts-text"
              rows={3}
              value={s.prompt}
              placeholder={examples[0]}
              onChange={(e) => onSettings({ prompt: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || !e.shiftKey)) {
                  e.preventDefault();
                  props.onPhase('review');
                }
              }}
            />
            <div className="ts-chips">
              {examples.slice(0, 3).map((ex) => (
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
            {refs.length > 0 && (
              <div className="ts-refblock">
                <div className="ts-refblock-head">
                  {refs.length > 1 ? 'What the model calls each reference' : 'What the model calls the image'} · click a token to insert it
                </div>
                <ReferenceChips refs={refs} thumbUrl={props.thumbUrl} onPick={pickToken} onMakeFirst={shared ? props.onMakeFirst : undefined} />
                {shared && (
                  <p className="ts-note">
                    <code>&lt;image1&gt;</code> is the image being edited (the result keeps its size); the others are references the model can take
                    people, objects or style from.
                  </p>
                )}
                {def.id === 'direct' && <p className="ts-note">Say which reference drives what: who appears, whose motion, which music.</p>}
                <TokenWarnings prompt={s.prompt} refs={refs} shared={shared || def.id === 'direct'} />
              </div>
            )}
          </Field>
        )}

        {manual && (
          <p className="ts-note warn ts-manual-note">
            The full prompt is hand-written (in Review), so it replaces what these settings would generate.{' '}
            <button type="button" className="ts-linkbtn" onClick={() => onSettings({ manualPrompt: null })}>
              Discard it
            </button>
          </p>
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
                <input type="number" inputMode="numeric" min={16} value={customW} placeholder="width" onChange={(e) => onSettings({ customSize: `${e.target.value}x${customH}` })} />
                <span>×</span>
                <input type="number" inputMode="numeric" min={16} value={customH} placeholder="height" onChange={(e) => onSettings({ customSize: `${customW}x${e.target.value}` })} />
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
            {isRetouch && def.id !== 'edit' && (
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
            {isRetouch && stills > 1 && !shared && (
              <p className="ts-note">{stills} images → {stills} separate results, each processed on its own with these settings.</p>
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
          <div className="ts-preview-title">Using{promptField && Object.values(roles).some((r) => r.startsWith('<')) ? ' · click one to put its tag in the prompt' : ''}</div>
          <div className="ts-thumbs">
            {shown.slice(0, 8).map((p) => (
              <Thumb
                key={p}
                path={p}
                role={roles[p]}
                thumbUrl={props.thumbUrl}
                onRemove={paths.length > 1 ? () => props.onRemovePath(p) : undefined}
                onPick={promptField && roles[p]?.startsWith('<') ? () => pickToken(roles[p]) : undefined}
              />
            ))}
            {shown.length > 8 ? <div className="ts-thumb ts-thumb-more">+{shown.length - 8}</div> : null}
          </div>
        </div>
        <ShapeFooter {...props} />
      </aside>
    </div>
  );
}

function ShapeFooter(props: FlowViewProps) {
  const err = props.intent ? validateShape(props.intent, props.settings, props.paths) : 'Pick something to do';
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

/**
 * "Prompt sent to the model": the engine's own expansion of the prompt (presets, living photo, --append), the
 * reference tokens it can use, and a full manual override. Edit and Direct prompts are already sent verbatim, so
 * editing them edits the prompt itself; anything else switches to a hand-written manualPrompt.
 */
function PromptPanel(props: FlowViewProps & { def: ReturnType<typeof intentById> }) {
  const { def, settings: s, onSettings, paths, preview } = props;
  const verbatim = promptIsVerbatim(def.id);
  const manual = s.manualPrompt !== null;
  const [editingVerbatim, setEditingVerbatim] = useState(false);
  const [raw, setRaw] = useState(false);
  const editing = manual || editingVerbatim;
  const textRef = useRef<HTMLTextAreaElement>(null);
  const value = manual ? (s.manualPrompt as string) : s.prompt;
  const setValue = (v: string) => onSettings(manual ? { manualPrompt: v } : { prompt: v });
  const insert = useTokenInsert(textRef, value, setValue);

  const data = preview?.status === 'ok' ? preview.data : undefined;
  const engine = def.engine === 'retouch' ? 'loki-retouch' : 'loki-reshoot';
  const local = refItems(def.id, paths, true);
  // The engine's list when we have it (it is what the run will use); with a manual prompt the photo of Bring to
  // life is a plain <Picture 1> either way.
  const refs: RefItem[] = data && data.references.length ? data.references.map((r) => ({ token: r.token, path: r.path })) : local;
  const shared = usesReferences(def.id, classify(paths)) || def.id === 'direct';

  const preset = def.engine === 'retouch' ? retouchPreset(def.id, s) : '';
  let shownText = '';
  if (data?.prompt) shownText = data.prompt;
  else if (verbatim) shownText = s.prompt.trim();
  else if (preview?.status === 'loading') shownText = '';
  else shownText = preset ? `(the built-in “${preset}” prompt${s.append.trim() ? `, then: ${s.append.trim()}` : ''})` : def.id === 'alive' ? '(the living-photo prompt)' : '';

  const startEdit = () => {
    if (verbatim) setEditingVerbatim(true);
    else onSettings({ manualPrompt: data?.prompt ?? '' });
    window.setTimeout(() => textRef.current?.focus(), 0);
  };

  let source = '';
  if (manual) source = 'written by you · replaces the generated prompt';
  else if (verbatim) source = 'your words, sent as written';
  else if (data?.fromEngine) source = `as ${engine} expands it`;

  return (
    <section className="ts-prompt" aria-label="Prompt sent to the model">
      <div className="ts-prompt-head">
        <span className="ts-field-label">Prompt sent to the model</span>
        {source ? <span className="ts-field-hint">{source}</span> : null}
        <span className="ts-prompt-actions">
          {!editing && (
            <button type="button" className="ts-chip" onClick={startEdit} disabled={preview?.status === 'loading' && !verbatim}>
              Edit full prompt
            </button>
          )}
          {editingVerbatim && (
            <button type="button" className="ts-chip" onClick={() => setEditingVerbatim(false)}>
              Done
            </button>
          )}
          {manual && (
            <button type="button" className="ts-chip" onClick={() => onSettings({ manualPrompt: null })}>
              Reset to generated
            </button>
          )}
          {!editing && data?.modelInput ? (
            <button type="button" className="ts-chip" aria-pressed={raw} onClick={() => setRaw((r) => !r)}>
              {raw ? 'Hide raw model input' : 'Raw model input'}
            </button>
          ) : null}
        </span>
      </div>

      {refs.length > 0 && (
        <>
          <ReferenceChips refs={refs} thumbUrl={props.thumbUrl} onPick={editing ? insert : undefined} />
          {editing ? <p className="ts-note">Click a token to insert it at the cursor.</p> : null}
        </>
      )}

      {editing ? (
        <textarea
          ref={textRef}
          className="ts-text ts-prompt-edit"
          rows={manual ? 12 : 5}
          value={value}
          placeholder="Write the whole prompt, using the tokens above for the references"
          onChange={(e) => setValue(e.target.value)}
          aria-label="Full prompt"
        />
      ) : raw && data?.modelInput ? (
        <pre className="ts-prompt-text raw">{data.modelInput}</pre>
      ) : (
        <pre className="ts-prompt-text">{preview?.status === 'loading' && !shownText ? `Asking ${engine}…` : shownText}</pre>
      )}

      {editing && <TokenWarnings prompt={value} refs={refs} shared={shared || manual} />}
      {manual && def.id === 'alive' ? (
        <p className="ts-note">Runs without the living-photo expansion: the subject and camera settings no longer apply; the photo is <code>&lt;Picture 1&gt;</code>.</p>
      ) : null}
      {manual && def.engine === 'retouch' && preset ? (
        <p className="ts-note">The “{preset}” preset still sets the output size; its built-in prompt and the extra direction are replaced.</p>
      ) : null}
      {data?.note ? <p className="ts-note">{data.note}</p> : null}
      {preview?.status === 'error' ? <p className="ts-note warn">Could not ask {engine} for its prompt ({preview.error}); showing what is known here.</p> : null}
    </section>
  );
}

function ReviewView(props: FlowViewProps & { def: ReturnType<typeof intentById> }) {
  const { def, settings: s, paths, source, status } = props;
  const plan = planFor(def.id, s, paths, source);
  const eta = estimateSeconds(plan);
  const manual = s.manualPrompt !== null;
  const roles = referenceTokens(def.id, paths, manual);
  const shown = Object.keys(roles).length ? Object.keys(roles) : paths;
  const total = eta * plan.outputs;
  const busy = status === 'submitting';
  const problem = validateForRun(def.id, s, paths);
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
            <p>{usesReferences(def.id, classify(paths)) ? `One result from ${classify(paths).images.length} images` : def.tagline}</p>
          </div>
        </div>
        <PromptPanel {...props} />
        <dl className="ts-summary">
          {def.id === 'alive' && !manual ? (
            <>
              <dt>Subject</dt>
              <dd className="ts-quote">{s.describe.trim() || '(no description — add one to help keep identity)'}</dd>
            </>
          ) : null}
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
            {def.id === 'alive' && !manual ? ` · camera ${s.shake === 'none' ? 'locked' : s.shake}` : ''}
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
          <button type="button" className="ts-btn primary big" onClick={props.onRun} disabled={busy || !!problem} title={problem || undefined} autoFocus>
            {busy ? 'Queuing…' : `Queue ${plan.jobCount > 1 ? `${plan.jobCount} jobs` : 'it'}`}
            <kbd>Ctrl ⏎</kbd>
          </button>
          {problem ? <span className="ts-footer-note">{problem}</span> : null}
        </div>
      </aside>
    </div>
  );
}

function DoneView(props: FlowViewProps) {
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

/** A token in the prompt that names none of the inputs (the model would see the bare text). */
function unknownTokenProblem(prompt: string, id: IntentId, paths: string[], manual: boolean): string | null {
  const tokens = Object.values(referenceTokens(id, paths, manual));
  const { unknown } = tokenIssues(prompt, tokens);
  if (unknown.length === 0) return null;
  return `${unknown.join(', ')} ${unknown.length > 1 ? 'match' : 'matches'} none of the selected files`;
}

/** Whether the Shape step is complete (what Review needs before it opens). */
export function validateShape(id: IntentId, s: TransformSettings, paths: string[]): string | null {
  const a = availability(id, classify(paths));
  if (!a.ok) return a.reason || 'Not available for this selection';
  const def = intentById(id);
  if (s.manualPrompt === null) {
    if (def.needsText === 'prompt' && !s.prompt.trim()) return def.id === 'direct' ? 'Describe the scene to continue' : 'Describe what should change';
    if (promptIsVerbatim(id)) {
      const bad = unknownTokenProblem(s.prompt, id, paths, false);
      if (bad) return bad;
    }
  }
  if (s.sizeMode === 'custom' && def.engine === 'retouch' && def.id !== 'wallpaper' && !parseSize(s.customSize)) return 'Enter a width and height';
  return null;
}

/** Everything a run needs: the Shape checks plus the review step's hand-written prompt, if any. */
export function validateForRun(id: IntentId, s: TransformSettings, paths: string[]): string | null {
  const shape = validateShape(id, s, paths);
  if (shape) return shape;
  if (s.manualPrompt !== null) {
    if (!s.manualPrompt.trim()) return 'Write the prompt, or reset it to the generated one';
    return unknownTokenProblem(s.manualPrompt, id, paths, true);
  }
  return null;
}
