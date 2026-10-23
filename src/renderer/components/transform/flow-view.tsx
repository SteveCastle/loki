import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  DREAM_ASPECTS,
  FAST_STEPS,
  INTENTS,
  MAX_DURATION,
  MIN_DURATION,
  STRENGTH_PRESETS,
  availability,
  dreamSize,
  maxVariations,
  qualityLabel,
  strengthOf,
  usesSelection,
  classify,
  estimateSeconds,
  formatDuration,
  intentById,
  intentKind,
  mediaKind,
  parseSize,
  planFor,
  promptIsVerbatim,
  activePins,
  NO_TARGET,
  pinnable,
  referenceTokens,
  runGroups,
  retouchPreset,
  runSummary,
  SAME_VIEW,
  suggestTemplateName,
  tokenIssues,
  usesReferences,
  type IntentId,
  type DreamAspect,
  type EngineId,
  type IntentKind,
  type Quality,
  type ReshootModel,
  type Shake,
  type TransformSettings,
} from './intents';
import { deleteTemplate, saveTemplate, templateSettings, templateSummary, useAllTemplates, useLastRun, useTemplates, type FlowPhase, type LastRun, type PromptTemplate } from './store';
import { GlyphCheck, GlyphDie, GlyphMusic, GlyphPin, GlyphScene, IntentIcon } from './icons';
import { markUsed, savedTransformKey } from '../../saved-recent';
import CameraRig from './camera-rig';
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
  action,
  children,
}: {
  label: string;
  hint?: string;
  /** A control at the right of the header (e.g. Save prompt). */
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="ts-field">
      <div className="ts-field-head">
        <span className="ts-field-label">{label}</span>
        {hint ? <span className="ts-field-hint">{hint}</span> : null}
        {action ? <span className="ts-field-action">{action}</span> : null}
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

// Style chips for a new picture (Generate, Reimagine).
const DREAM_STYLE_CHIPS: Array<{ label: string; text: string }> = [
  { label: 'Photo', text: 'a photograph, natural light, sharp focus' },
  { label: 'Cinematic', text: 'cinematic still, dramatic lighting, film grain' },
  { label: 'Golden hour', text: 'lit by warm golden-hour sunlight' },
  { label: 'Oil painting', text: 'as an oil painting with visible brush strokes' },
  { label: 'Watercolor', text: 'as a soft watercolor illustration' },
  { label: 'Anime', text: 'in a clean anime illustration style' },
  { label: '3D render', text: 'as a detailed 3D render, soft studio lighting' },
];

/** The local engine an intent runs on, by its executable's name. */
export function engineName(engine: EngineId): string {
  if (engine === 'retouch') return 'loki-retouch';
  if (engine === 'reshoot') return 'loki-reshoot';
  return 'loki-dream';
}

function appendPhrase(prompt: string, phrase: string): string {
  const base = prompt.trim();
  if (!base) return phrase.charAt(0).toUpperCase() + phrase.slice(1);
  if (base.toLowerCase().includes(phrase.toLowerCase())) return prompt;
  return `${base.replace(/[.,;]+$/, '')}, ${phrase}`;
}

// What each kind of starting point is called, everywhere in the flow and the palette.
const KIND: Record<IntentKind | 'saved', { label: string; title: string }> = {
  preset: { label: 'Preset', title: 'Preset: runs a built-in prompt. You can add to it, or replace it in Review.' },
  custom: { label: 'Custom', title: 'Custom: runs the prompt you write.' },
  saved: { label: 'Saved', title: 'Saved: a custom prompt you saved to start from again.' },
};

export function KindBadge({ kind }: { kind: IntentKind | 'saved' }) {
  return (
    <span className={`ts-kind ts-kind-${kind}`} title={KIND[kind].title}>
      {KIND[kind].label}
    </span>
  );
}

/** A labelled row of chips under a prompt (Saved / Ideas / Add), so each row says what its chips do. */
function ChipRow({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="ts-chiprow" role="group" aria-label={label}>
      <span className="ts-chiprow-label" title={hint}>
        {label}
      </span>
      <div className="ts-chips">{children}</div>
    </div>
  );
}

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

export { suggestTemplateName };

/**
 * "☆ Save prompt": always shown where a custom prompt is written or reviewed (greyed out while empty, "★ Saved"
 * once it is), opening an inline name field. Esc there cancels only the naming (see data-ts-local-esc). The prompt
 * is saved with the rest of its configuration (output size, quality…), so bringing it back restores both.
 */
export function SavePrompt({ intent, settings }: { intent: IntentId; settings: TransformSettings }) {
  const list = useTemplates(intent);
  const [naming, setNaming] = useState<string | null>(null);
  const trimmed = settings.prompt.trim();
  const current = list.find((t) => t.prompt === trimmed);
  const commit = () => {
    if (naming !== null && saveTemplate(intent, naming, trimmed, settings)) setNaming(null);
  };
  if (naming !== null) {
    return (
      <span className="ts-template-name" data-ts-local-esc>
        <input
          autoFocus
          value={naming}
          placeholder="Name"
          aria-label="Saved prompt name"
          onChange={(e) => setNaming(e.target.value)}
          onFocus={(e) => e.target.select()}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              e.stopPropagation();
              commit();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              setNaming(null);
            }
          }}
        />
        <button type="button" className="ts-chip" disabled={!naming.trim()} onClick={commit}>
          Save
        </button>
        <button type="button" className="ts-chip" onClick={() => setNaming(null)}>
          Cancel
        </button>
      </span>
    );
  }
  if (current) {
    return (
      <span className="ts-chip ts-template-saved" title={`Saved as “${current.name}”`}>
        ★ Saved
      </span>
    );
  }
  return (
    <button
      type="button"
      className="ts-chip ts-template-save"
      disabled={!trimmed}
      onClick={() => setNaming(suggestTemplateName(trimmed))}
      title={trimmed ? 'Keep this prompt, and its size and quality settings, to start from next time' : 'Write a prompt to save it'}
    >
      ☆ Save prompt
    </button>
  );
}

// The user's saved tasks for an intent, as a labelled chip row: click one to start from it (and its settings). A
// custom intent's are prompts (the one in the box is marked); a preset's are settings saved from a result.
function SavedPrompts({ intent, prompt, onUse }: { intent: IntentId; prompt: string; onUse: (t: PromptTemplate) => void }) {
  const list = useTemplates(intent);
  const trimmed = prompt.trim();
  if (!list.length) return null;
  return (
    <ChipRow label="Saved" hint="Your saved tasks: click one to start from it">
      {list.map((t) => (
        <span key={t.id} className={`ts-template${trimmed && t.prompt === trimmed ? ' active' : ''}`}>
          <button type="button" className="ts-template-use" title={`Start from your saved task:\n${templateSummary(intent, t)}`} onClick={() => {
              markUsed(savedTransformKey(intent, t.id));
              onUse(t);
            }}>
            ☆ {t.name}
          </button>
          <button type="button" className="ts-template-x" onClick={() => deleteTemplate(intent, t.id)} aria-label={`Delete saved task ${t.name}`} title="Delete">
            ×
          </button>
        </span>
      ))}
    </ChipRow>
  );
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
  pinned,
  onPin,
}: {
  path: string;
  role?: string;
  thumbUrl: (p: string) => string;
  onRemove?: () => void;
  pinned?: boolean;
  /** Offers the pin toggle (pinnable intents). */
  onPin?: (on: boolean) => void;
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
      className={`ts-thumb ts-thumb-${kind}${onPick ? ' ts-thumb-pick' : ''}${pinned ? ' pinned' : ''}`}
      title={onPick && role ? `${baseName(path)}: click to insert ${role} into the prompt` : baseName(path)}
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
      {onPin ? (
        <button
          type="button"
          className={`ts-thumb-pin${pinned ? ' on' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            onPin(!pinned);
          }}
          aria-pressed={!!pinned}
          aria-label={pinned ? `Unpin ${baseName(path)}` : `Pin ${baseName(path)} as a reference`}
          title={
            pinned
              ? 'Pinned: comes along every time, after whatever you select. Click to unpin.'
              : 'Pin as a reference: it comes along every time, after whatever you select'
          }
        >
          <GlyphPin />
        </button>
      ) : null}
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

function wordCount(text: string): number {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

/**
 * A prompt textarea with a toggle that expands it over the whole Transform panel for long, focused writing (the
 * reference chips in `extra` stay usable there). Esc or the toggle collapses it. Only one textarea is mounted at a
 * time, on `textRef`, so token insertion keeps working; focus and caret carry over. Expanded, Enter is a newline and
 * only Ctrl/Cmd+Enter submits.
 */
export function ExpandablePrompt({
  textRef,
  value,
  onChange,
  label,
  placeholder,
  rows,
  className = 'ts-text',
  ariaLabel,
  onSubmit,
  submitOnEnter = false,
  extra,
}: {
  textRef: React.RefObject<HTMLTextAreaElement>;
  value: string;
  onChange: (v: string) => void;
  /** Heading of the expanded view. */
  label: string;
  placeholder?: string;
  rows?: number;
  className?: string;
  ariaLabel?: string;
  onSubmit?: () => void;
  /** Collapsed: plain Enter submits (Shift+Enter is a newline). */
  submitOnEnter?: boolean;
  /** Shown above the text in the expanded view (e.g. the reference chips). */
  extra?: React.ReactNode;
}) {
  // the panel the expanded view covers; null = collapsed
  const [panel, setPanel] = useState<HTMLElement | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const carry = useRef<{ start: number; end: number } | null>(null);

  const toggle = (open: boolean) => {
    const ta = textRef.current;
    carry.current = ta ? { start: ta.selectionStart, end: ta.selectionEnd } : { start: value.length, end: value.length };
    setPanel(open ? (wrapRef.current?.closest('.ts-panel') as HTMLElement | null) : null);
  };
  useEffect(() => {
    const ta = textRef.current;
    const c = carry.current;
    if (!ta || !c) return;
    carry.current = null;
    ta.focus();
    ta.setSelectionRange(c.start, c.end);
  }, [panel, textRef]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (panel && e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      toggle(false);
      return;
    }
    if (e.key === 'Enter' && onSubmit && (e.ctrlKey || e.metaKey || (!panel && submitOnEnter && !e.shiftKey))) {
      e.preventDefault();
      onSubmit();
    }
  };
  const area = (expanded: boolean) => (
    <textarea
      ref={textRef}
      className={expanded ? 'ts-text ts-xprompt-text' : className}
      rows={expanded ? undefined : rows}
      value={value}
      placeholder={placeholder}
      aria-label={ariaLabel ?? label}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
    />
  );
  const words = wordCount(value);

  return (
    <>
      <div ref={wrapRef} className="ts-xprompt-inline">
        {panel ? <div className="ts-xprompt-away">Editing in the expanded view</div> : area(false)}
        {!panel && (
          <button type="button" className="ts-xprompt-toggle" title="Expand the prompt to the whole window" aria-label="Expand the prompt editor" onClick={() => toggle(true)}>
            <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9" />
            </svg>
          </button>
        )}
      </div>
      {panel &&
        createPortal(
          <div className="ts-xprompt" data-ts-local-esc role="group" aria-label={`${label} (expanded)`}>
            <div className="ts-xprompt-head">
              <span className="ts-field-label">{label}</span>
              <span className="ts-field-hint">
                {words} {words === 1 ? 'word' : 'words'}
                {onSubmit ? ' · Ctrl+Enter to continue' : ''} · Esc to collapse
              </span>
              <button type="button" className="ts-chip ts-xprompt-collapse" aria-label="Collapse the prompt editor" onClick={() => toggle(false)}>
                <svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M13.5 6.5h-4v-4M9.5 6.5l4.5-4.5M2.5 9.5h4v4M6.5 9.5 2 14" />
                </svg>
                Collapse
              </button>
            </div>
            {extra ? <div className="ts-xprompt-extra">{extra}</div> : null}
            {area(true)}
          </div>,
          panel
        )}
    </>
  );
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
  pinned = [],
}: {
  refs: RefItem[];
  /** Pinned files, marked on their chips. */
  pinned?: string[];
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
              <span className="ts-ref-name">
                {pinned.includes(r.path) ? (
                  <span className="ts-ref-pinned" title="Pinned reference">
                    <GlyphPin />
                  </span>
                ) : null}
                {baseName(r.path)}
              </span>
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
function refItems(intent: IntentId, paths: string[], manual: boolean, pinned: string[] = []): RefItem[] {
  const tokens = referenceTokens(intent, paths, manual, pinned);
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
  /** Start a custom intent from one of the user's saved prompts. */
  onTemplate?: (id: IntentId, t: PromptTemplate) => void;
  /** Load the last run (its intent and every setting) to adjust and run again. */
  onLoadRun?: (run: LastRun) => void;
  /** Open the Scene editor on the single image / video (absent when the selection can't start a scene). */
  onOpenScene?: () => void;
  onPhase: (p: FlowPhase) => void;
  onRun: () => void;
  onClose: () => void;
  onRemovePath: (p: string) => void;
  /** Make this file the first input (<image1>, the one an edit changes). */
  onMakeFirst?: (p: string) => void;
  /** Pin / unpin a reference (Edit, Direct). */
  onPin?: (p: string, on: boolean) => void;
  /** Pick more references from the library in a picker window (Edit, Direct). */
  onAddReferences?: () => void;
  /** A picker window is open for onAddReferences. */
  pickingReferences?: boolean;
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
  { id: 'shape', label: 'Set up' },
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
            <span className="ts-title-sub">
              {intent && !usesSelection(intent) ? 'a new picture' : summarizeInputs(paths)}
              {intent && activePins(intent, props.settings.pinned).length ? ` · ${activePins(intent, props.settings.pinned).length} pinned` : ''}
            </span>
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
  const last = useLastRun();
  const lastOk = last ? availability(last.intent, inputs) : null;
  const saved = useAllTemplates().filter(({ intent: id }) => availability(id, inputs).ok);
  const create = { id: 'create' as const, title: 'New pictures', sub: 'loki-dream · Kromoss · from a description' };
  const groups: Array<{ id: 'image' | 'video' | 'create'; title: string; sub: string }> = [
    { id: 'image', title: 'Images', sub: 'loki-retouch · Qwen Image 2.1' },
    { id: 'video', title: 'Video', sub: 'loki-reshoot · MiniMax H3, with sound' },
  ];
  // With nothing selected, making something new is all there is to do: it comes first.
  if (props.paths.length === 0) groups.unshift(create);
  else groups.push(create);
  return (
    <div className="ts-choose">
      <p className="ts-lead">
        {props.paths.length === 0 ? 'Nothing is selected: make a new picture from a description.' : `What do you want to do with ${summarizeInputs(props.paths)}?`}
      </p>
      <p className="ts-legend">
        <span>
          <KindBadge kind="preset" /> built-in prompt, ready to run
        </span>
        <span>
          <KindBadge kind="custom" /> you write the prompt
        </span>
        {saved.length ? (
          <span>
            <KindBadge kind="saved" /> a task you saved
          </span>
        ) : null}
      </p>
      {last && lastOk && (
        <section className="ts-group">
          <div className="ts-group-head">
            <h3>Last run</h3>
            <span>the same action and settings, on this selection · adjust anything before it runs</span>
          </div>
          <div className="ts-cards">
            <button
              type="button"
              className={`ts-card ts-card-last${lastOk.ok ? '' : ' disabled'}`}
              disabled={!lastOk.ok}
              onClick={() => props.onLoadRun?.(last)}
              data-intent="last"
            >
              <span className="ts-card-icon">
                <IntentIcon id={last.intent} />
              </span>
              <span className="ts-card-text">
                <strong>
                  {intentById(last.intent).title} again <KindBadge kind={last.settings.manualPrompt !== null ? 'custom' : intentKind(last.intent)} />
                </strong>
                <small>{lastOk.ok ? runSummary(last.intent, last.settings) : lastOk.reason}</small>
              </span>
            </button>
          </div>
        </section>
      )}
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
                    <strong>
                      {i.title} <KindBadge kind={intentKind(i.id)} />
                    </strong>
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
      {props.onOpenScene && (
        <section className="ts-group">
          <div className="ts-group-head">
            <h3>Scene</h3>
            <span>opens in its own window · both engines, one region at a time</span>
          </div>
          <div className="ts-cards">
            <button type="button" className="ts-card ts-card-scene" onClick={props.onOpenScene} data-intent="scene">
              <span className="ts-card-icon">
                <GlyphScene />
              </span>
              <span className="ts-card-text">
                <strong>Scene editor</strong>
                <small>Mask parts of the picture, edit them, then animate them one by one into a living scene</small>
              </span>
            </button>
          </div>
        </section>
      )}
      {saved.length > 0 && (
        <section className="ts-group">
          <div className="ts-group-head">
            <h3>Saved tasks</h3>
            <span>save one from the prompt box, or from a result&apos;s Generation info</span>
          </div>
          <div className="ts-cards">
            {saved.map(({ intent: id, t }) => (
              <button key={t.id} type="button" className="ts-card ts-card-saved" onClick={() => props.onTemplate?.(id, t)} title={templateSummary(id, t)}>
                <span className="ts-card-icon">
                  <IntentIcon id={id} />
                </span>
                <span className="ts-card-text">
                  <strong>
                    ☆ {t.name} <KindBadge kind="saved" />
                  </strong>
                  <small>
                    {intentById(id).title} · {templateSummary(id, t)}
                  </small>
                </span>
              </button>
            ))}
          </div>
        </section>
      )}
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

const QUALITY_OPTIONS = (engine: EngineId): Array<SegOption<Quality>> =>
  engine === 'dream'
    ? [
        { value: 'draft', label: 'Turbo', sub: '8 steps, fast' },
        { value: 'balanced', label: 'Balanced', sub: '30 steps' },
        { value: 'fine', label: 'Fine', sub: '50 steps, slow' },
      ]
    : [
        { value: 'draft', label: 'Draft', sub: 'fast' },
        { value: 'balanced', label: 'Balanced', sub: engine === 'retouch' ? '25 steps' : '20 steps' },
        { value: 'fine', label: 'Fine', sub: 'slow' },
      ];

const ASPECT_OPTIONS: Array<SegOption<DreamAspect>> = [
  ...DREAM_ASPECTS.map((a) => ({ value: a.id, label: a.label, sub: `${a.width}×${a.height}` })),
  { value: 'custom', label: 'Custom' },
];

const MODEL_OPTIONS: Array<SegOption<ReshootModel>> = [
  { value: 'h3', label: 'H3', sub: 'best quality' },
  { value: 'fast', label: 'Fast', sub: `${FAST_STEPS} steps, ~4x faster` },
];

function promptLabel(id: IntentId, shared: boolean, stills: number): string {
  if (id === 'direct') return 'Describe the scene';
  if (id === 'generate') return 'Describe the picture';
  if (id === 'reimagine') return 'Describe the result';
  return shared ? `What should change? Refer to the ${stills} images by their tokens` : 'What should change?';
}

/** Under Generate's custom size: what the model will actually make (snapped to its grid and pixel budget). */
function customSizeNote(raw: string, snapped: [number, number] | null): string {
  const rule = 'the model works in multiples of 16, up to about a megapixel';
  const typed = parseSize(raw);
  if (!snapped || !typed) return `width × height: ${rule}`;
  if (typed[0] === snapped[0] && typed[1] === snapped[1]) return rule;
  return `makes ${snapped[0]}×${snapped[1]}: ${rule}`;
}

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
  const isReshoot = def.engine === 'reshoot';
  const isDream = def.engine === 'dream';
  const manual = s.manualPrompt !== null;
  const pins = activePins(def.id, s.pinned);
  const groups = runGroups(def.id, paths, s.pinned);
  const roles = referenceTokens(def.id, paths, manual, s.pinned);
  // The selection, then pins that are not part of it.
  const shown = usesSelection(def.id) ? [...(Object.keys(roles).length ? paths.filter((p) => p in roles) : paths), ...pins.filter((p) => !paths.includes(p))] : [];
  const stills = inputs.images.length;
  const shared = usesReferences(def.id, inputs) || !!groups;
  const refs = refItems(def.id, paths, manual, s.pinned);
  // What may be pinned: Edit pins images, Direct anything it takes.
  const canPin = (p: string) => pinnable(def.id) && (def.id === 'edit' ? mediaKind(p) === 'image' : mediaKind(p) !== 'other');
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
            <h2>
              {def.title} <KindBadge kind={intentKind(def.id)} />
            </h2>
            <p>{def.blurb}</p>
          </div>
        </div>

        {def.needsText !== 'prompt' && <SavedPrompts intent={def.id} prompt="" onUse={(t) => onSettings(templateSettings(def.id, t))} />}

        {def.needsText === 'prompt' && (
          <Field
            label={promptLabel(def.id, shared, stills)}
            hint="sent as written · Enter ▸ review"
            action={<SavePrompt intent={def.id} settings={s} />}
          >
            <ExpandablePrompt
              textRef={textRef}
              rows={3}
              value={s.prompt}
              placeholder={examples[0]}
              label={promptLabel(def.id, false, stills)}
              onChange={(v) => onSettings({ prompt: v })}
              submitOnEnter
              onSubmit={() => props.onPhase('review')}
              extra={
                refs.length > 0 ? (
                  <>
                    <ReferenceChips refs={refs} thumbUrl={props.thumbUrl} onPick={pickToken} pinned={pins} />
                    <TokenWarnings prompt={s.prompt} refs={refs} shared={shared || def.id === 'direct'} />
                  </>
                ) : null
              }
            />
            <SavedPrompts
              intent={def.id}
              prompt={s.prompt}
              onUse={(t) => {
                const { prompt } = t;
                onSettings({ ...(t.settings || {}), prompt, manualPrompt: null });
                window.setTimeout(() => {
                  const ta = textRef.current;
                  ta?.focus();
                  ta?.setSelectionRange(prompt.length, prompt.length);
                }, 0);
              }}
            />
            {examples.length > 0 && (
              <ChipRow label="Ideas" hint="Example prompts: click one to replace the prompt with it">
                {examples.slice(0, 3).map((ex) => (
                  <button key={ex} type="button" className="ts-chip ts-chip-idea" title={`Replace the prompt with:\n${ex}`} onClick={() => onSettings({ prompt: ex })}>
                    {ex.length > 46 ? `${ex.slice(0, 44)}…` : ex}
                  </button>
                ))}
              </ChipRow>
            )}
            {(def.id === 'edit' || isDream) && (
              <ChipRow label="Add" hint="Click to add the phrase to the end of the prompt">
                {(isDream ? DREAM_STYLE_CHIPS : STYLE_CHIPS).map((c) => (
                  <button key={c.label} type="button" className="ts-chip" title={`Add “${c.text}”`} onClick={() => onSettings({ prompt: appendPhrase(s.prompt, c.text) })}>
                    + {c.label}
                  </button>
                ))}
              </ChipRow>
            )}
            {refs.length > 0 && (
              <div className="ts-refblock">
                <div className="ts-refblock-head">
                  {refs.length > 1 ? 'Tokens for each image' : 'Token for the image'} · click one to insert it into the prompt
                </div>
                <ReferenceChips refs={refs} thumbUrl={props.thumbUrl} onPick={pickToken} onMakeFirst={shared && !groups ? props.onMakeFirst : undefined} pinned={pins} />
                {groups && groups.length > 1 && (
                  <p className="ts-note">
                    {groups.length} selected → {groups.length} separate runs, each with the {pins.length} pinned reference{pins.length > 1 ? 's' : ''} after it.
                  </p>
                )}
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
            You wrote your own prompt in Review: it replaces the {def.title} preset prompt and anything added to it here.{' '}
            <button type="button" className="ts-linkbtn" onClick={() => onSettings({ manualPrompt: null })}>
              Use the preset prompt again
            </button>
          </p>
        )}

        {isDream && s.quality !== 'draft' && (
          <Field label="Avoid" hint="optional · what the picture should not have">
            <textarea
              className="ts-text"
              rows={1}
              value={s.negative}
              placeholder="e.g. blurry, watermark, extra fingers"
              aria-label="Negative prompt"
              onChange={(e) => onSettings({ negative: e.target.value })}
            />
          </Field>
        )}

        {def.id === 'generate' && (
          <Field label="Shape" hint={plan.outWidth ? `${plan.outWidth}×${plan.outHeight}` : undefined}>
            <Segmented<DreamAspect>
              label="Shape"
              value={s.aspect}
              options={ASPECT_OPTIONS}
              onChange={(v) => onSettings(v === 'custom' ? { aspect: v, customSize: parseSize(s.customSize) ? s.customSize : '1152x896' } : { aspect: v })}
            />
            {s.aspect === 'custom' && (
              <div className="ts-size-inputs">
                <input type="number" inputMode="numeric" min={256} value={customW} placeholder="width" onChange={(e) => onSettings({ customSize: `${e.target.value}x${customH}` })} />
                <span>×</span>
                <input type="number" inputMode="numeric" min={256} value={customH} placeholder="height" onChange={(e) => onSettings({ customSize: `${customW}x${e.target.value}` })} />
                <small>{customSizeNote(s.customSize, dreamSize(s))}</small>
              </div>
            )}
          </Field>
        )}

        {def.id === 'reimagine' && (
          <Field label="How much to change" hint={`strength ${strengthOf(s).toFixed(2)}`}>
            <Segmented<number> label="How much to change" value={strengthOf(s)} options={STRENGTH_PRESETS} onChange={(v) => onSettings({ strength: v })} />
            <input
              className="ts-range"
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={strengthOf(s)}
              onChange={(e) => onSettings({ strength: parseFloat(e.target.value) })}
              style={{ ['--ts-fill' as string]: `${((strengthOf(s) - 0.05) / 0.95) * 100}%` }}
              aria-label="Strength"
            />
            <div className="ts-range-ticks">
              <span>keep more</span>
              <span>redraw more</span>
            </div>
          </Field>
        )}

        {def.id === 'angle' && (
          <Field label="Camera position" hint={stills > 1 ? `same angle for all ${stills} images` : undefined}>
            <CameraRig
              azimuth={s.cameraAzimuth}
              elevation={s.cameraElevation}
              onChange={onSettings}
              thumbUrl={inputs.images[0] ? props.thumbUrl(inputs.images[0]) : undefined}
            />
          </Field>
        )}

        {isRetouch && def.id !== 'edit' && !manual && (
          <Field label="Add to the preset" hint="optional · follows the built-in prompt">
            <textarea className="ts-text" rows={2} value={s.append} placeholder="e.g. keep the film grain" onChange={(e) => onSettings({ append: e.target.value })} />
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

        {isReshoot && (
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

        {isReshoot && (
          <Field label="Model">
            <Segmented<ReshootModel> label="Model" value={s.reshootModel} options={MODEL_OPTIONS} onChange={(v) => onSettings({ reshootModel: v })} />
          </Field>
        )}

        {!isReshoot || s.reshootModel !== 'fast' ? (
          <Field label="Quality">
            <Segmented<Quality> label="Quality" value={s.quality} options={QUALITY_OPTIONS(def.engine)} onChange={(v) => onSettings({ quality: v })} />
            {isDream && s.quality === 'draft' ? (
              <p className="ts-note">Turbo runs the distilled 8-step model without guidance: several times faster, a little less detail, and no “Avoid” prompt.</p>
            ) : null}
          </Field>
        ) : (
          <p className="ts-note">The fast model always runs {FAST_STEPS} steps, the count it was distilled for: there is no quality setting. It may look less refined than H3.</p>
        )}

        <div className="ts-row2">
          <Field label="Variations" hint={s.variations > 1 ? `${s.variations} results${def.id === 'generate' ? ', consecutive seeds' : ''}` : undefined}>
            <div className="ts-stepper-num">
              <button type="button" onClick={() => onSettings({ variations: Math.max(1, s.variations - 1) })} aria-label="Fewer variations">
                −
              </button>
              <span>{s.variations}</span>
              <button type="button" onClick={() => onSettings({ variations: Math.min(maxVariations(def.id), s.variations + 1) })} aria-label="More variations">
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
            {isReshoot && (
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
            {(isRetouch || def.id === 'reimagine') && stills > 1 && !shared && !groups && (
              <p className="ts-note">{stills} images → {stills} separate results, each processed on its own with these settings.</p>
            )}
            {isDream && <p className="ts-note">A fixed seed with the same prompt and settings makes the same picture again.</p>}
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
            {isReshoot ? <span>{plan.frames} frames · {s.duration.toFixed(1)} s</span> : null}
          </div>
          <div className="ts-preview-eta">
            <span>≈ {formatDuration(eta)}</span>
            <small>{plan.outputs > 1 ? `each · ${plan.outputs} results` : 'on the GPU'}</small>
          </div>
        </div>
        <div className="ts-preview-inputs">
          <div className="ts-preview-title">Inputs{promptField && Object.values(roles).some((r) => r.startsWith('<')) ? ' · click one to insert its token' : ''}</div>
          {!usesSelection(def.id) ? (
            <p className="ts-note">
              None: a new picture from your description{paths.length ? ', the selection is left alone' : ''}. It goes into the Generated folder of your library.
            </p>
          ) : null}
          <div className="ts-thumbs">
            {shown.slice(0, 8).map((p) => {
              const isPin = pins.includes(p);
              const selected = paths.includes(p);
              return (
                <Thumb
                  key={p}
                  path={p}
                  role={roles[p]}
                  thumbUrl={props.thumbUrl}
                  // A pin that is not part of the selection goes away by unpinning it.
                  onRemove={!selected ? undefined : paths.length > 1 ? () => props.onRemovePath(p) : undefined}
                  onPick={promptField && roles[p]?.startsWith('<') ? () => pickToken(roles[p]) : undefined}
                  pinned={isPin}
                  onPin={props.onPin && canPin(p) ? (on) => props.onPin?.(p, on) : undefined}
                />
              );
            })}
            {shown.length > 8 ? <div className="ts-thumb ts-thumb-more">+{shown.length - 8}</div> : null}
          </div>
          {pinnable(def.id) && props.onAddReferences ? (
            <button
              type="button"
              className="ts-btn ts-add-refs"
              onClick={props.onAddReferences}
              title={`Browse and search the library for ${def.id === 'edit' ? 'images' : 'images, clips or audio'} to add as references, in a window of its own that closes when you're done`}
            >
              {props.pickingReferences ? 'Choosing in the picker window…' : '+ Add references…'}
            </button>
          ) : null}
          {pinnable(def.id) && props.onPin ? (
            <p className="ts-pin-hint">
              {pins.length
                ? 'Pinned references come along every time, after whatever you select. They are saved with your prompts.'
                : 'Pin a reference (top-left of its thumbnail) to keep it for next time, even with one image selected.'}
            </p>
          ) : null}
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
  const engine = engineName(def.engine);
  const local = refItems(def.id, paths, true, s.pinned);
  // The engine's list when we have it (it is what the run will use); with a manual prompt the photo of Bring to
  // life is a plain <Picture 1> either way.
  const refs: RefItem[] = data && data.references.length ? data.references.map((r) => ({ token: r.token, path: r.path })) : local;
  const shared = usesReferences(def.id, classify(paths)) || def.id === 'direct' || !!runGroups(def.id, paths, s.pinned);

  const preset = def.engine === 'retouch' ? retouchPreset(def.id, s) : '';
  let shownText = '';
  if (data?.prompt) shownText = data.prompt;
  else if (verbatim) shownText = s.prompt.trim();
  else if (preview?.status === 'loading') shownText = '';
  else shownText = preset || def.id === 'alive' ? `(the ${def.title} preset prompt${s.append.trim() ? `, then: ${s.append.trim()}` : ''})` : '';

  const startEdit = () => {
    if (verbatim) setEditingVerbatim(true);
    else onSettings({ manualPrompt: data?.prompt ?? '' });
    window.setTimeout(() => textRef.current?.focus(), 0);
  };

  let source = '';
  if (manual) source = `custom · yours, replaces the ${def.title} preset prompt`;
  else if (verbatim) source = 'custom · your words, sent as written';
  else if (data?.fromEngine) source = `preset · as ${engine} expands it`;
  else source = 'preset';

  return (
    <section className="ts-prompt" aria-label="Prompt sent to the model">
      <div className="ts-prompt-head">
        <span className="ts-field-label">Prompt sent to the model</span>
        {source ? <span className="ts-field-hint">{source}</span> : null}
        <span className="ts-prompt-actions">
          {!editing && (
            <button
              type="button"
              className="ts-chip"
              onClick={startEdit}
              disabled={preview?.status === 'loading' && !verbatim}
              title={verbatim ? 'Edit your prompt' : 'Start from the preset prompt and write your own; it replaces the preset prompt for this run'}
            >
              {verbatim ? 'Edit prompt' : 'Write my own'}
            </button>
          )}
          {editingVerbatim && (
            <button type="button" className="ts-chip" onClick={() => setEditingVerbatim(false)}>
              Done
            </button>
          )}
          {verbatim && <SavePrompt intent={def.id} settings={s} />}
          {manual && (
            <button type="button" className="ts-chip" onClick={() => onSettings({ manualPrompt: null })}>
              Use the preset prompt
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
        <ExpandablePrompt
          textRef={textRef}
          className="ts-text ts-prompt-edit"
          rows={manual ? 12 : 5}
          value={value}
          placeholder="Write the whole prompt, using the tokens above for the references"
          onChange={setValue}
          label="Prompt sent to the model"
          ariaLabel="Full prompt"
          extra={
            refs.length > 0 ? (
              <>
                <ReferenceChips refs={refs} thumbUrl={props.thumbUrl} onPick={insert} />
                <TokenWarnings prompt={value} refs={refs} shared={shared || manual} />
              </>
            ) : null
          }
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
        <p className="ts-note">The {def.title} preset still sets the output size; your prompt replaces its built-in prompt and anything added to it.</p>
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
  const roles = referenceTokens(def.id, paths, manual, s.pinned);
  const shown = !usesSelection(def.id) ? [] : Object.keys(roles).length ? Object.keys(roles) : paths;
  const total = eta * plan.outputs;
  const groups = runGroups(def.id, paths, s.pinned);
  const pins = activePins(def.id, s.pinned).length;
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
              {plan.outputs > 1 ? ` × ${plan.outputs}` : ''} <KindBadge kind={manual ? 'custom' : intentKind(def.id)} />
            </h2>
            <p>
              {groups
                ? `${groups.length > 1 ? `Each of ${groups.length} selected` : 'The selection'}, with ${pins} pinned reference${pins > 1 ? 's' : ''} after it`
                : usesReferences(def.id, classify(paths))
                ? `One result from ${classify(paths).images.length} images`
                : def.tagline}
            </p>
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
            <span className="muted">
              {def.id === 'generate' ? ' — added to your library, in its Generated folder' : ` — saved beside the original${plan.outputs > 1 ? 's' : ''}`}
            </span>
          </dd>
          {def.engine === 'dream' && s.quality !== 'draft' && s.negative.trim() ? (
            <>
              <dt>Avoid</dt>
              <dd className="ts-quote">{s.negative.trim()}</dd>
            </>
          ) : null}
          <dt>Settings</dt>
          <dd>
            {def.engine === 'reshoot' && s.reshootModel === 'fast' ? 'fast model' : `${qualityLabel(def.engine, s.quality)} quality`} ({plan.steps} steps) ·{' '}
            {s.seedMode === 'fixed' ? `seed ${s.seed}` : 'random seed'}
            {def.id === 'reimagine' ? ` · strength ${strengthOf(s).toFixed(2)}` : ''}
            {s.variations > 1 ? ` · ${Math.min(s.variations, maxVariations(def.id))} variations` : ''}
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
        {!usesSelection(def.id) ? <p className="ts-note">None: a new picture from your description.</p> : null}
        <div className="ts-thumbs big">
          {shown.slice(0, 6).map((p) => (
            <Thumb key={p} path={p} role={roles[p]} thumbUrl={props.thumbUrl} />
          ))}
          {shown.length > 6 ? <div className="ts-thumb ts-thumb-more">+{shown.length - 6}</div> : null}
        </div>
        <p className="ts-note">Uses the local GPU engine <code>{engineName(def.engine)}</code>. Nothing leaves your machine.</p>
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
  const custom = props.intent && promptIsVerbatim(props.intent) && props.settings.prompt.trim();
  return (
    <div className="ts-done">
      <span className="ts-done-badge">
        <GlyphCheck />
      </span>
      <h2>{n > 1 ? `${n} jobs queued` : 'Queued'}</h2>
      <p>
        {props.intent && !usesSelection(props.intent)
          ? 'The new pictures are added to your library (in its Generated folder) when they finish, and the library jumps to the first one.'
          : 'The results appear next to the originals when they finish, and the library jumps to the first new file.'}{' '}
        Follow progress in the job queue.
        This run is remembered as your <strong>last run</strong>: re-run it on other files from the palette’s “{props.intent ? intentById(props.intent).title : 'Last run'} again” chip.
      </p>
      {custom && props.intent ? (
        <div className="ts-done-save">
          <SavePrompt intent={props.intent} settings={props.settings} />
        </div>
      ) : null}
      <div className="ts-footer center">
        <button type="button" className="ts-btn ghost" onClick={props.onAgain}>
          Adjust and run again
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
function unknownTokenProblem(prompt: string, id: IntentId, paths: string[], manual: boolean, pinned: string[] = []): string | null {
  const tokens = Object.values(referenceTokens(id, paths, manual, pinned));
  const { unknown } = tokenIssues(prompt, tokens);
  if (unknown.length === 0) return null;
  return `${unknown.join(', ')} ${unknown.length > 1 ? 'match' : 'matches'} none of the selected files`;
}

/** Whether the Shape step is complete (what Review needs before it opens). */
export function validateShape(id: IntentId, s: TransformSettings, paths: string[]): string | null {
  const a = availability(id, classify(paths));
  if (!a.ok) return a.reason || 'Not available for this selection';
  const groups = runGroups(id, paths, s.pinned);
  if (groups && !groups.length) return NO_TARGET;
  const def = intentById(id);
  if (s.manualPrompt === null) {
    if (def.needsText === 'prompt' && !s.prompt.trim()) {
      if (def.id === 'direct') return 'Describe the scene to continue';
      if (def.id === 'generate') return 'Describe the picture';
      if (def.id === 'reimagine') return 'Describe the result';
      return 'Describe what should change';
    }
    if (promptIsVerbatim(id)) {
      const bad = unknownTokenProblem(s.prompt, id, paths, false, s.pinned);
      if (bad) return bad;
    }
  }
  if (id === 'angle' && s.manualPrompt === null && !retouchPreset(id, s)) return SAME_VIEW;
  if (s.sizeMode === 'custom' && def.engine === 'retouch' && def.id !== 'wallpaper' && !parseSize(s.customSize)) return 'Enter a width and height';
  if (id === 'generate' && !dreamSize(s)) return 'Enter a width and height';
  return null;
}

/** Everything a run needs: the Shape checks plus the review step's hand-written prompt, if any. */
export function validateForRun(id: IntentId, s: TransformSettings, paths: string[]): string | null {
  const shape = validateShape(id, s, paths);
  if (shape) return shape;
  if (s.manualPrompt !== null) {
    if (!s.manualPrompt.trim()) return 'Write the prompt, or reset it to the generated one';
    return unknownTokenProblem(s.manualPrompt, id, paths, true, s.pinned);
  }
  return null;
}
