import React, { useEffect, useRef, useState } from 'react';
import { DREAM_ENABLED } from '../../features';
import { mediaUrl } from '../../platform';
import { maskBounds, rectContains, type ContextSize } from './mask';
import { PromptLibrary } from './prompt-library';
import References from './references';
import { refTokens, type RefKind } from './refs';
import { composeFinish, FINISH_GROUNDING, FINISH_MODES, finishStrength, sketchWhere, DIALOGUE_LANGUAGES, MAX_GENERATIONS, dialogueSeconds, generationsInFlight, jobPhase, layerTiming, type DialogueLine, type FinishMode, type LayerOptions, type LoopKind, type SceneLayer } from './scene-api';

// Shortcut labels: ⌃/↵ have no glyph in Windows' UI fonts and render as junk.
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
export const KEY_UNDO = IS_MAC ? '⌘Z' : 'Ctrl+Z';
export const KEY_GENERATE = IS_MAC ? '⌘↵' : 'Ctrl+Enter';

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

export function KindIcon({ kind, size = 14 }: { kind: 'still' | 'motion' | 'paint'; size?: number }) {
  if (kind === 'paint') {
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
        <path d="M8 2.5c3.3 0 5.5 2.2 5.5 4.8 0 1.8-1.3 2.7-2.6 2.7H9.6c-.8 0-1.2.9-.7 1.5.6.7.1 2-1 2C4.6 13.5 2.5 11 2.5 8S4.8 2.5 8 2.5z" fill="none" stroke="currentColor" strokeWidth="1.3" />
        <circle cx="5.5" cy="7" r="1" fill="currentColor" />
        <circle cx="8" cy="5" r="1" fill="currentColor" />
        <circle cx="10.6" cy="6.6" r="1" fill="currentColor" />
      </svg>
    );
  }
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

function statusOf(l: SceneLayer, stale: boolean): { text: string; tone: 'busy' | 'error' | 'ok' | 'idle' | 'warn' } {
  if (l.job) {
    const { label, percent } = jobPhase(l.job);
    const more = l.queued?.length ? ` +${l.queued.length}` : '';
    return { text: `${label.toLowerCase()}${percent !== null ? ` ${percent}%` : ''}${more}`, tone: 'busy' };
  }
  if (l.kind === 'paint') {
    if (!l.mask) return { text: 'paint something', tone: 'idle' };
    return l.options?.guide ? { text: 'guide', tone: 'idle' } : { text: 'in picture', tone: 'ok' };
  }
  if (l.error) return { text: 'failed', tone: 'error' };
  if (l.versions.length && stale) return { text: `v${l.active + 1} · below changed`, tone: 'warn' };
  if (l.versions.length) return { text: l.active >= 0 ? `v${l.active + 1}/${l.versions.length}` : 'off', tone: 'ok' };
  if (!l.mask) return { text: 'paint an area', tone: 'idle' };
  return { text: 'ready', tone: 'idle' };
}

export interface LayerListProps {
  layers: SceneLayer[];
  selectedId: string | null;
  baseName: string;
  onSelect: (id: string | null) => void;
  onAdd: (kind: 'still' | 'motion' | 'paint') => void;
  onToggleHidden: (id: string) => void;
  onMove: (id: string, dir: 1 | -1) => void;
  /** Move a layer to index `to` of the stack (bottom = 0). */
  onMoveTo: (id: string, to: number) => void;
  /** Layers whose shown version was made from stills that have since changed. */
  stale: Set<string>;
}

export function LayerList(props: LayerListProps) {
  const { layers, selectedId } = props;
  const top = [...layers].reverse();
  // Drag to reorder: where the dragged row would land (a row of the top-first list, and above or below it).
  const [dragId, setDragId] = useState<string | null>(null);
  const [drop, setDrop] = useState<{ id: string; above: boolean } | null>(null);
  const endDrag = () => {
    setDragId(null);
    setDrop(null);
  };
  const dropAt = () => {
    if (!dragId || !drop || drop.id === dragId) return;
    // Above a row in the top-first list = just over it in the stack.
    const rest = layers.filter((l) => l.id !== dragId);
    const j = rest.findIndex((l) => l.id === drop.id);
    if (j >= 0) props.onMoveTo(dragId, drop.above ? j + 1 : j);
  };
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
          <button type="button" className="sc-btn" onClick={() => props.onAdd('paint')} title="Paint in colour: rough colour or a sketch, in the picture or as a guide the AI follows">
            <KindIcon kind="paint" /> Paint
          </button>
        </div>
      </div>
      <ol className="sc-layer-list">
        {top.map((l, i) => {
          const st = statusOf(l, props.stale.has(l.id));
          const mark = drop && drop.id === l.id && dragId !== l.id ? (drop.above ? ' drop-above' : ' drop-below') : '';
          return (
            <li
              key={l.id}
              className={`sc-layer-row${l.id === selectedId ? ' selected' : ''}${l.hidden ? ' hidden' : ''}${dragId === l.id ? ' dragging' : ''}${mark}`}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', l.id);
                setDragId(l.id);
              }}
              onDragOver={(e) => {
                if (!dragId) return;
                e.preventDefault();
                const r = e.currentTarget.getBoundingClientRect();
                const above = e.clientY < r.top + r.height / 2;
                if (!drop || drop.id !== l.id || drop.above !== above) setDrop({ id: l.id, above });
              }}
              onDrop={(e) => {
                e.preventDefault();
                dropAt();
                endDrag();
              }}
              onDragEnd={endDrag}
              title={st.tone === 'warn' ? 'A layer below was moved, hidden or changed since this version was made: generate again to make it from the picture as it is now' : undefined}
            >
              <button type="button" className="sc-layer-main" onClick={() => props.onSelect(l.id)}>
                <span className={`sc-kind ${l.kind}`}>
                  <KindIcon kind={l.kind} />
                </span>
                <span className="sc-layer-name">{l.name || (l.kind === 'motion' ? 'Motion' : l.kind === 'paint' ? 'Paint' : 'Edit')}</span>
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
  /** The shown version was made from stills below that have since changed. */
  stale: boolean;
  /** The Mask section (mask-panel.tsx). */
  maskPanel: React.ReactNode;
  /** The scene's paint layers (references). */
  paintLayers: SceneLayer[];
  onPatch: (patch: Partial<SceneLayer>) => void;
  onOptions: (patch: LayerOptions) => void;
  onGenerate: () => void;
  /** Cancel one generation (its job id), or all of the layer's when omitted. */
  onCancel: (job?: string) => void;
  onDelete: () => void;
  onClearMask: () => void;
  onDeleteVersion: (n: number) => void;
  /** Discard every version but n. */
  onKeepOnly: (n: number) => void;
  onLoop: (n: number, kind: LoopKind) => void;
  /** Redo version n's finish (edges, detail) with the current settings. */
  onRefinish: (n: number) => void;
  /** A finish layer: switch what it does with its paint (add, recolour, repose). */
  onFinishMode: (mode: FinishMode) => void;
  /** Whether the active version's motion map is shown; toggled by onShowMotion. */
  motionShown: boolean;
  onShowMotion: () => void;
  onGrowToMotion: () => void;
  onClearProtect: () => void;
  /** The active version's PTS multiplier in the scene (layerRate: 1 = as generated, 2 = half speed). */
  loopSpeed: number;
  /** Pick an audio or video file whose voice the first speaker uses; resolves to its path, or null. */
  onChooseVoice: () => Promise<string | null>;
  /** References: pick in a picker window, or with a file dialog. */
  onPickLibrary: (kinds: RefKind[]) => { result: Promise<string[] | null>; cancel: () => void };
  onPickFiles: (kinds: RefKind[]) => Promise<string[]>;
  /** Motion: open the roto of version n (pick what to keep, track it through the clip). */
  onRoto: (n: number) => void;
  /** Whether this layer's roto is open now. */
  rotoActive: boolean;
  /** Motion: drop version n's roto. */
  onUntrack: (n: number) => void;
  /** Motion: an Edit layer below that removes version n's rotoscoped subject from the picture under it. */
  onCleanPlate: (n: number) => void;
}

const LENGTHS = [5, 8, 10, 15];

/** The Dialogue section of a Motion layer: lines in order, an optional voice, and whether they fit the length. */
function DialogueEditor({ o, onOptions, onChooseVoice }: { o: LayerOptions; onOptions: (p: LayerOptions) => void; onChooseVoice: () => Promise<string | null> }) {
  const lines = o.dialogue || [];
  const set = (next: DialogueLine[]) => onOptions(next.length ? { dialogue: next, sound: true } : { dialogue: next });
  const patch = (i: number, p: Partial<DialogueLine>) => set(lines.map((l, j) => (j === i ? { ...l, ...p } : l)));
  const need = dialogueSeconds(lines);
  const length = o.duration || 5;
  const fits = need <= length;
  const longer = LENGTHS.find((d) => d >= need);
  const voiceName = o.voiceRef ? o.voiceRef.split(/[\\/]/).pop() : '';
  return (
    <div className="sc-dialogue">
      <div className="sc-dialogue-head">
        <span className="sc-label">Dialogue</span>
        <button type="button" className="sc-btn sc-btn-quiet sc-btn-tiny" onClick={() => set([...lines, { text: '', lang: lines[lines.length - 1]?.lang || 'English' }])}>
          + Line
        </button>
      </div>
      {lines.map((l, i) => (
        // eslint-disable-next-line react/no-array-index-key
        <div key={i} className="sc-dialogue-line">
          <input className="sc-input sc-speaker" value={l.speaker || ''} placeholder="the person" title="Who says it (empty: the person in the picture)" onChange={(e) => patch(i, { speaker: e.target.value })} aria-label="Speaker" />
          <select className="sc-select sc-lang" value={l.lang || 'English'} onChange={(e) => patch(i, { lang: e.target.value })} aria-label="Language">
            {DIALOGUE_LANGUAGES.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
          <button type="button" className="sc-row-x" title="Remove this line" onClick={() => set(lines.filter((_, j) => j !== i))}>
            ×
          </button>
          <textarea className="sc-prompt sc-line-text" rows={2} value={l.text} placeholder="What they say, in their language" onChange={(e) => patch(i, { text: e.target.value })} aria-label="Words" />
        </div>
      ))}
      {lines.length > 0 && (
        <>
          <div className="sc-voice">
            <span className="sc-label" title="An audio or video file of the voice the first speaker should have (its first 15 s are used)">
              Voice
            </span>
            {o.voiceRef ? (
              <span className="sc-voice-file" title={o.voiceRef}>
                {voiceName}
                <button type="button" className="sc-row-x" title="Use a generated voice instead" onClick={() => onOptions({ voiceRef: '' })}>
                  ×
                </button>
              </span>
            ) : (
              <span className="sc-faint">generated to suit the speaker</span>
            )}
            <button
              type="button"
              className="sc-btn sc-btn-tiny"
              onClick={async () => {
                const p = await onChooseVoice();
                if (p) onOptions({ voiceRef: p });
              }}
            >
              {o.voiceRef ? 'Change…' : 'Choose…'}
            </button>
          </div>
          {need > 0 && (
            <div className={fits ? 'sc-faint sc-fit' : 'sc-warn'}>
              About {need.toFixed(1)} s of speech in a {length} s layer
              {!fits && longer && (
                <>
                  {' '}
                  — it won’t fit.{' '}
                  <button type="button" className="sc-btn sc-btn-tiny" onClick={() => onOptions({ duration: longer })}>
                    Use {longer} s
                  </button>
                </>
              )}
              {!fits && !longer && ' — too long even for 15 s: split it across two layers or shorten it.'}
            </div>
          )}
        </>
      )}
    </div>
  );
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
  const painted = !!l.mask;
  // Whether the shapes reach past the version's area (only measurable for plain shapes: no links, no invert).
  const mo = l.maskOpts || {};
  const plain = !mo.invert && !(l.shapes || []).some((op) => op.t === 'layer') ? maskBounds(l.shapes || []) : null;
  const grown = plain && mo.expand && mo.expand > 0 ? { x: plain.x - mo.expand, y: plain.y - mo.expand, w: plain.w + 2 * mo.expand, h: plain.h + 2 * mo.expand } : plain;
  const version = l.versions[l.active];
  const running = !!l.job;
  const queued = l.queued || [];
  const full = generationsInFlight(l) >= MAX_GENERATIONS;
  const finish = !motion ? o.finish : undefined;
  // Kromoss redraws from a description: no instruction grounding, no reference images.
  const dream = DREAM_ENABLED && !motion && o.engine === 'dream';
  const needPrompt = !motion && !finish && !l.prompt.trim();
  // A finish layer: the grounding its words are wrapped in, and what the engine will be given.
  const grounding = finish ? o.grounding || FINISH_GROUNDING[finish] : '';
  const guideToken = (() => {
    const refs = o.refs || [];
    const i = refs.findIndex((r) => r.layer && r.layer === o.paint);
    return (i >= 0 && refTokens('still', refs, false)[i]) || '<image2>';
  })();
  // A Retouch sketch finish is also shown its paint alone (the server adds it as the last image) and told where it is.
  const sketchTarget = (() => {
    if (finish !== 'sketch' || dream) return { where: '', target: '' };
    const paint = (props.paintLayers || []).find((p) => p.id === o.paint);
    const images = refTokens('still', o.refs || [], false).filter(Boolean).length;
    return { where: paint ? sketchWhere(paint.region, l.region) : '', target: `<image${images + 2}>` };
  })();
  // A Retouch Repose is also given the crop again (the server adds it last) and, for a mannequin, its pose in words.
  const repose = (() => {
    if (finish !== 'repose' || dream) return { pose: '', self: '' };
    const paint = (props.paintLayers || []).find((p) => p.id === o.paint);
    const images = refTokens('still', o.refs || [], false).filter(Boolean).length;
    const mannequin = (paint?.shapes || []).some((op) => op.t === 'pose' && !op.off);
    return { pose: mannequin ? '[the mannequin’s pose, in words]' : '', self: `<image${images + 2}>` };
  })();
  const fullPrompt = finish ? composeFinish(grounding, l.prompt, guideToken, sketchTarget.where, sketchTarget.target, repose.pose, repose.self) : '';
  const outside = version && grown && !rectContains(version.region, grown);
  const ideas = motion ? MOTION_IDEAS : STILL_IDEAS;
  const promptRef = useRef<HTMLTextAreaElement>(null);
  // Where the cursor was last in the prompt. Clicking a token or thumbnail moves
  // focus off the textarea first, so its live selection can't be trusted then.
  const promptCaret = useRef<number | null>(null);
  useEffect(() => {
    promptCaret.current = null;
  }, [l.id]);
  const rememberCaret = () => {
    const el = promptRef.current;
    if (el) promptCaret.current = el.selectionStart;
  };
  // A reference token goes in at the prompt's cursor (or the end), spaced from its neighbours.
  const insertToken = (token: string) => {
    const el = promptRef.current;
    const text = l.prompt;
    const at = Math.min(promptCaret.current ?? text.length, text.length);
    const before = text.slice(0, at);
    const after = text.slice(at);
    const pre = before && !/\s$/.test(before) ? ' ' : '';
    const post = after && !/^\s/.test(after) ? ' ' : '';
    props.onPatch({ prompt: before + pre + token + post + after });
    const caret = (before + pre + token).length;
    promptCaret.current = caret;
    window.setTimeout(() => {
      el?.focus();
      el?.setSelectionRange(caret, caret);
    }, 0);
  };
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

      {finish && (
        <div className="sc-field-row">
          <span className="sc-label" title="What this layer does with its paint layer">
            Finish
          </span>
          <Seg
            value={finish}
            disabled={running}
            onChange={(mode) => props.onFinishMode(mode)}
            options={(Object.keys(FINISH_MODES) as FinishMode[]).map((mode) => ({ value: mode, label: FINISH_MODES[mode].label, title: FINISH_MODES[mode].help }))}
          />
        </div>
      )}
      <PromptLibrary
        kind={l.kind}
        label={
          motion
            ? 'How should it move?'
            : finish
            ? FINISH_MODES[finish].ask
            : dream
            ? 'What should be there?'
            : 'What should change here?'
        }
        prompt={l.prompt}
        onUse={(prompt) => {
          props.onPatch({ prompt });
          window.setTimeout(() => promptRef.current?.focus(), 0);
        }}
      />
      <textarea
        id="sc-prompt"
        ref={promptRef}
        className="sc-prompt"
        value={l.prompt}
        rows={4}
        placeholder={
          motion
            ? 'e.g. the waterfall flows and mist drifts up (leave empty for natural ambient motion)'
            : finish
            ? FINISH_MODES[finish].placeholder
            : dream
            ? 'e.g. a stormy sunset sky with towering clouds (describe it, don’t instruct)'
            : 'e.g. replace the sky with a stormy sunset'
        }
        onChange={(e) => {
          promptCaret.current = e.target.selectionStart;
          props.onPatch({ prompt: e.target.value });
        }}
        onSelect={rememberCaret}
        onBlur={rememberCaret}
      />
      {finish && !dream && (
        <details className="sc-grounding">
          <summary>
            Grounding <span className="sc-faint">· how the paint is read{o.grounding ? ' (yours)' : ''}</span>
          </summary>
          <textarea
            className="sc-prompt sc-grounding-text"
            rows={5}
            value={grounding}
            onChange={(e) => props.onOptions({ grounding: e.target.value === FINISH_GROUNDING[finish] ? '' : e.target.value })}
            aria-label="Grounding"
          />
          <div className="sc-faint sc-grounding-help">
            {'{what}'} is replaced by your words above{FINISH_MODES[finish].guide ? `, {guide} by the paint layer’s token (${guideToken})` : ''}
            {finish === 'sketch' && !dream ? `, {where} by where the sketch is, {target} by the sketch shown alone (${sketchTarget.target})` : ''}.
          </div>
          <div className="sc-grounding-full">
            <span className="sc-label">Sent to the engine</span>
            <p>{fullPrompt}</p>
          </div>
          <div className="sc-version-actions">
            {o.grounding && (
              <button type="button" className="sc-btn sc-btn-quiet sc-btn-tiny" onClick={() => props.onOptions({ grounding: '' })}>
                Reset grounding
              </button>
            )}
            <button
              type="button"
              className="sc-btn sc-btn-tiny"
              onClick={() => {
                props.onPatch({ prompt: fullPrompt });
                props.onOptions({ finish: undefined, grounding: '' });
              }}
              title="Put the whole prompt in the prompt box to edit freely (the layer stops wrapping your words in grounding)"
            >
              Edit full prompt
            </button>
          </div>
        </details>
      )}
      {!dream && (
      <References
        layer={l}
        onInsert={insertToken}
        onChange={(refs, prompt) => {
          props.onOptions({ refs });
          if (prompt !== l.prompt) props.onPatch({ prompt });
        }}
        onPickLibrary={props.onPickLibrary}
        onPickFiles={props.onPickFiles}
        paintLayers={props.paintLayers}
      />
      )}
      <div className="sc-ideas" hidden={!!finish}>
        {ideas.map((idea) => (
          <button key={idea} type="button" className="sc-chip" onClick={() => appendIdea(idea)}>
            + {idea}
          </button>
        ))}
      </div>
      {motion && <DialogueEditor o={o} onOptions={props.onOptions} onChooseVoice={props.onChooseVoice} />}
      {props.maskPanel}
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
            {(o.sound || (o.dialogue || []).length > 0) && (
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
            {DREAM_ENABLED && (
              <>
                <span className="sc-label" title="Which engine draws the area">
                  Engine
                </span>
                <Seg
                  value={o.engine || 'retouch'}
                  onChange={(engine) => props.onOptions({ engine })}
                  options={[
                    { value: 'retouch', label: 'Retouch', title: 'Qwen Image: edits the area by instruction (“make it night”), can use reference images' },
                    { value: 'dream', label: 'Kromoss', title: 'MOSS-Image (Kromoss): redraws the masked area from a description of what should be there, in the picture’s context' },
                  ]}
                />
              </>
            )}
            {dream && (
              <>
                <span className="sc-label" title="How much of the area is redrawn: low keeps its shapes and colours (a sketch), 100% draws it anew">
                  Strength
                </span>
                <div className="sc-slider">
                  <input
                    type="range"
                    min={10}
                    max={100}
                    step={5}
                    value={Math.round((o.strength ?? finishStrength(finish)) * 100)}
                    onChange={(e) => props.onOptions({ strength: Number(e.target.value) / 100 })}
                    aria-label="Strength"
                  />
                  <span>{Math.round((o.strength ?? finishStrength(finish)) * 100)}%</span>
                </div>
              </>
            )}
            <span className="sc-label">Quality</span>
            <Seg
              value={o.steps || 25}
              onChange={(steps) => props.onOptions({ steps })}
              options={[
                { value: 12, label: 'Draft', title: dream ? 'The 8-step turbo LoRA: several times faster' : undefined },
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
        {l.job && (
          <>
            <div className="sc-progress">
              <div className="sc-progress-bar" style={{ width: `${l.job.total ? (l.job.done / l.job.total) * 100 : 3}%` }} />
            </div>
            <div className="sc-job-line">{l.job.step === 'scene-lock'
                ? 'Finishing: blending the motion into the picture (static edges, detail)…'
                : l.job.state === 'pending'
                ? 'Waiting for the GPU (one diffusion job at a time)…'
                : l.job.line || 'Starting…'}</div>
            <button type="button" className="sc-btn" onClick={() => props.onCancel(l.job?.id)}>
              Cancel
            </button>
          </>
        )}
        {queued.length > 0 && (
          <div className="sc-queue">
            <div className="sc-queue-head">
              <span className="sc-label">Up next</span>
              <button type="button" className="sc-btn sc-btn-quiet" onClick={() => props.onCancel()} title="Cancel the running generation and every queued one">
                Cancel all
              </button>
            </div>
            {queued.map((j, i) => (
              <div key={j.id} className="sc-queue-row">
                <span className="sc-queue-n">{i + 1}</span>
                <span className="sc-queue-text" title={j.prompt}>{j.note || j.prompt || (motion ? 'Ambient motion' : 'Edit')}</span>
                <button type="button" className="sc-btn sc-btn-quiet sc-queue-x" onClick={() => props.onCancel(j.id)} title="Remove from the queue">
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        <button
          type="button"
          className={`sc-btn ${running ? '' : 'sc-btn-primary '}sc-btn-wide`}
          disabled={busy || !painted || needPrompt || full}
          onClick={props.onGenerate}
          title={
            !painted
              ? 'Paint or box the area first'
              : needPrompt
              ? 'Describe the edit first'
              : full
              ? `At most ${MAX_GENERATIONS} generations per layer`
              : running
              ? `Runs after the ones before it, with the prompt and settings as they are now (${KEY_GENERATE})`
              : KEY_GENERATE
          }
        >
          {!painted ? 'Paint the area first' : needPrompt ? 'Describe the edit first' : `${running ? 'Queue another' : l.versions.length ? 'Generate again' : motion ? 'Animate this area' : 'Generate'}  ${KEY_GENERATE}`}
        </button>
        {l.error && <div className="sc-error">{l.error}</div>}
        {outside && !running && <div className="sc-warn">The mask now reaches past the area this version was made for: generate again to fill it.</div>}
        {props.stale && !running && (
          <div className="sc-warn">A layer below was moved, hidden or changed since this version was made, so it may not match the picture under it: generate again to make it from the picture as it is now.</div>
        )}
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
              {version.inputs && version.inputs.length > 0 && (
                <div className="sc-version-inputs">
                  <span className="sc-label" title="Exactly what the engine was given for this version (click to open full size)">
                    Inputs
                  </span>
                  {version.inputs.map((p, i) => {
                    const label = i === 0 ? (motion ? '<Picture 1>' : '<image1>') : motion ? `ref ${i}` : `<image${i + 1}>`;
                    return (
                      <a key={p} className="sc-input-thumb" href={mediaUrl(p)} target="_blank" rel="noreferrer" title={`${label} · ${p}`}>
                        {/\.(mp4|mov|webm|mkv|m4v)$/i.test(p) ? <video src={mediaUrl(p)} muted preload="metadata" /> : /\.(wav|mp3|m4a|aac|flac|ogg|opus)$/i.test(p) ? <span>♪</span> : <img src={mediaUrl(p)} alt="" loading="lazy" />}
                        <span>{label}</span>
                      </a>
                    );
                  })}
                </div>
              )}
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
              {!motion && (
                <div className="sc-version-actions">
                  <span className="sc-label">Mask</span>
                  <button
                    type="button"
                    className={`sc-btn${o.fit ? ' on' : ''}`}
                    onClick={() => props.onOptions({ fit: !o.fit })}
                    title="Grow the mask to wherever the shown version changed the picture near it, so a result that corrects a rough shape (or casts a shadow) isn't cut off at the edge you drew"
                  >
                    {o.fit ? 'Fitted to result' : 'Fit to result'}
                  </button>
                </div>
              )}
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
                    <span className="sc-label">Finish</span>
                    <button
                      type="button"
                      className="sc-btn"
                      disabled={busy || full}
                      onClick={() => props.onRefinish(l.active)}
                      title="Redo this version's finish (static edges, detail) from the engine's original output with the current Edges/Detail settings, as a new version. No regeneration: a few seconds on the CPU."
                    >
                      Re-finish
                    </button>
                  </div>
                  <div className="sc-version-actions">
                    <span className="sc-label" title="Rotoscope: what of this clip is kept, frame by frame (tracked with SAM 3.1)">
                      Roto
                    </span>
                    {(() => {
                      const t = version.track;
                      const live = !!t && ['pending', 'in_progress', 'paused'].includes(t.state);
                      return (
                        <>
                          <button type="button" className={`sc-btn${props.rotoActive ? ' on' : ''}`} disabled={busy} onClick={() => props.onRoto(l.active)} title="Pick what to keep on any frame of the clip, then track it through the clip">
                            {t ? 'Edit roto' : 'Roto…'}
                          </button>
                          {live && t && <span className="sc-faint">{t.state === 'pending' ? 'waiting for the GPU' : `tracking ${t.total ? Math.round((t.done / t.total) * 100) : 0}%`}</span>}
                          {t?.state === 'error' && (
                            <span className="sc-pick-error" title={t.error}>
                              failed
                            </span>
                          )}
                          {t?.state === 'completed' && (
                            <button
                              type="button"
                              className={`sc-btn${o.matte === false ? '' : ' on'}`}
                              onClick={() => props.onOptions({ matte: o.matte === false })}
                              title="Composite this layer through its roto matte (preview and export)"
                            >
                              {o.matte === false ? 'Matte off' : 'Matte on'}
                            </button>
                          )}
                          {t?.state === 'completed' && (
                            <button
                              type="button"
                              className="sc-btn"
                              disabled={busy}
                              onClick={() => props.onCleanPlate(l.active)}
                              title="Add an Edit layer below that removes this subject from the picture under it, so it can move away without its still copy showing through"
                            >
                              Clean plate
                            </button>
                          )}
                          {t && (
                            <button type="button" className="sc-btn sc-btn-quiet" disabled={busy} onClick={() => props.onUntrack(l.active)} title="Drop this version’s roto">
                              Remove
                            </button>
                          )}
                        </>
                      );
                    })()}
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
