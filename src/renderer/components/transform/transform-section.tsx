import React, { useState } from 'react';
import { mediaServerBase } from '../../platform';
import {
  INTENTS,
  availability,
  buildJobs,
  classify,
  intentById,
  intentKind,
  runSummary,
  settingsFor,
  type IntentId,
  type TransformSettings,
} from './intents';
import { EngineSetupDeferred } from './engine-setup';
import {
  loadRemembered,
  openTransformFlow,
  rememberTemplateRun,
  saveLastRun,
  submitJobs,
  useLastRun,
} from './store';
import { validateForRun } from './flow-view';
import { GlyphReplay, IntentIcon } from './icons';
import './transform-section.css';

// The palette's "Transform" section, in two visually separate groups:
// - Full control: the custom intents that run the prompt you write, with references and every setting (Edit a
//   photo, Direct a video). They open the flow at their prompt.
// - Presets: built-in treatments that run with the settings you used last time (Shift-click to adjust first; the
//   ones ending in "…" open a short settings step), and the last run (“… again”).
// The Scene editor is an icon in the palette header (context-palette.tsx), beside Open in Studio.
// "All options" opens the full flow at its Choose step.
// Generate needs no selection, so the section is there (with just Generate) even when nothing is selected.
const CUSTOM_ORDER: IntentId[] = ['edit', 'direct'];
const PRESET_ORDER: IntentId[] = ['restore', 'upscale', 'wallpaper', 'angle', 'reimagine', 'alive', 'generate'];

export interface TransformSectionProps {
  /** The working set: the right-clicked file plus any multi-selection ([] = nothing: only Generate is offered). */
  paths: string[];
  authToken: string | null;
  /** Seconds into the on-screen video when the single target is the one playing. */
  getVideoTime: () => number | undefined;
  /** Close the palette (after a job is queued or the flow opened). */
  onDone: () => void;
  notify: (type: 'success' | 'error' | 'info', title: string, message: string) => void;
}

export default function TransformSection(props: TransformSectionProps) {
  const { paths, authToken, getVideoTime, onDone, notify } = props;
  const [busy, setBusy] = useState<IntentId | 'last' | null>(null);
  const last = useLastRun();
  const lastOk = !!last && availability(last.intent, classify(paths)).ok;
  const inputs = classify(paths);
  const fits = (id: IntentId) => availability(id, inputs).ok;
  const custom = CUSTOM_ORDER.filter(fits);
  const chips = PRESET_ORDER.map(intentById).filter((i) => fits(i.id));
  const hasVideoOnly = inputs.images.length === 0 && inputs.videos.length === 1;

  const open = (intent: IntentId | undefined, start?: { prompt?: string; settings?: TransformSettings }) => {
    openTransformFlow({
      paths,
      videoTime: getVideoTime(),
      intent,
      phase: intent ? 'shape' : 'choose',
      prompt: start?.prompt,
      settings: start?.settings,
    });
    onDone();
  };

  const quickRun = async (id: IntentId, settings: TransformSettings = settingsFor(id, loadRemembered(id)), key: IntentId | 'last' = id) => {
    const built = buildJobs(id, settings, paths, { videoTime: getVideoTime(), recipe: true });
    if (built.error || built.jobs.length === 0) {
      notify('error', 'Cannot run', built.error || 'Nothing to run');
      return;
    }
    setBusy(key);
    try {
      await submitJobs(built.jobs, { mediaServerBase, authToken });
      saveLastRun(id, settings, paths);
      rememberTemplateRun(id, settings);
      onDone();
    } catch (e) {
      if (e instanceof EngineSetupDeferred) {
        if (e.result === 'background') notify('info', 'Downloading', 'The download continues in the background. Run this again when it has finished.');
      } else {
        notify('error', 'Failed to Create Job', 'Could not communicate with job service');
      }
      onDone();
    } finally {
      setBusy(null);
    }
  };

  const onChip = (e: React.MouseEvent, id: IntentId) => {
    const def = intentById(id);
    if (def.oneClick && !e.shiftKey) quickRun(id);
    else open(id);
  };

  // The last run: re-run as it was, or (Shift-click, or when it no longer fits this selection, e.g. a prompt naming
  // <image3> on two images) load it into the flow to adjust.
  const onLast = (e: React.MouseEvent) => {
    if (!last) return;
    const settings = { ...settingsFor(last.intent), ...last.settings };
    if (e.shiftKey || validateForRun(last.intent, settings, paths)) open(last.intent, { settings });
    else quickRun(last.intent, settings, 'last');
  };

  if (custom.length === 0 && chips.length === 0) return null;
  return (
    <div className="context-palette-merge transform-section">
      <div className="workflow-picker-header">
        <span className="action-group-title">Transform</span>
        <button
          type="button"
          className="person-rename-btn"
          onClick={() => open(undefined)}
          title="Every preset, custom prompt and option, step by step"
        >
          All options ▸
        </button>
      </div>
      {custom.length > 0 && (
        <div className="transform-custom" role="group" aria-label="Full control">
          <span className="transform-group-label">
            Full control <span className="transform-group-hint">your prompt, references, every setting</span>
          </span>
          <div className="transform-custom-cards">
            {custom.map((id) => (
              <button
                key={id}
                type="button"
                className="transform-custom-card"
                disabled={busy !== null}
                onClick={() => open(id)}
                title={chipTitle(id)}
              >
                <IntentIcon id={id} size={18} />
                <span className="transform-custom-text">
                  <span className="transform-custom-title">{customTitle(id, hasVideoOnly)}</span>
                  <span className="transform-custom-sub">{customSub(id, inputs.images.length > 1)}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
      {(chips.length > 0 || (last && lastOk)) && (
        <>
          <span className="transform-group-label">
            Presets <span className="transform-group-hint">built-in, with your last settings</span>
          </span>
          <div className="type-chips" role="group" aria-label="Presets">
            {last && lastOk && (
              <button
                type="button"
                className="type-chip transform-chip transform-chip-last"
                disabled={busy !== null}
                onClick={onLast}
                title={`Last run: ${intentById(last.intent).title} · ${runSummary(last.intent, last.settings)}
Click to run it again on this selection, Shift-click to adjust first.`}
              >
                <GlyphReplay size={13} />
                {busy === 'last' ? 'Queuing…' : `${intentById(last.intent).title} again`}
              </button>
            )}
            {chips.map((c) => (
              <button
                key={c.id}
                type="button"
                className="type-chip transform-chip"
                disabled={busy !== null}
                onClick={(e) => onChip(e, c.id)}
                title={chipTitle(c.id)}
              >
                <IntentIcon id={c.id} size={13} />
                {busy === c.id ? 'Queuing…' : chipLabel(c.id, hasVideoOnly)}
              </button>
            ))}
          </div>
        </>
      )}
      <span className="merge-selection-note">
        {paths.length === 0 ? 'nothing selected: Generate makes a new picture from a description · ' : ''}
        {hasVideoOnly ? 'acts on the frame you are looking at · ' : ''}
        {last && lastOk ? '“again” repeats your last run · ' : ''}Shift-click a preset to adjust it first · “…” opens its settings
      </span>
    </div>
  );
}

function chipTitle(id: IntentId): string {
  const def = intentById(id);
  if (id === 'generate') return `Custom: ${def.tagline}. Opens the prompt box; the selection is left alone.`;
  if (intentKind(id) === 'custom') return `Custom: ${def.tagline}. Opens the prompt box.`;
  if (def.oneClick) return `Preset: ${def.tagline}. Click to run with your last settings, Shift-click to adjust first.`;
  return `Preset: ${def.tagline}. Opens its settings.`;
}

function customTitle(id: IntentId, videoFrame: boolean): string {
  if (id === 'edit') return videoFrame ? 'Edit frame…' : 'Edit photo…';
  if (id === 'direct') return 'Direct video…';
  return chipLabel(id, videoFrame);
}

function customSub(id: IntentId, severalImages: boolean): string {
  if (id === 'edit') return severalImages ? 'Any change, using the others as references' : 'Describe any change';
  if (id === 'direct') return 'Images, clips & music → video';
  return intentById(id).tagline;
}

function chipLabel(id: IntentId, videoFrame: boolean): string {
  if (id === 'upscale') {
    const s = settingsFor('upscale', loadRemembered('upscale'));
    return `Upscale ${s.scale}×`;
  }
  if (id === 'edit') return videoFrame ? 'Edit frame…' : 'Edit…';
  if (id === 'alive') return 'Animate…';
  if (id === 'angle') return 'New angle…';
  if (id === 'direct') return 'Direct…';
  if (id === 'generate') return 'Generate…';
  if (id === 'reimagine') return 'Reimagine…';
  if (id === 'wallpaper') {
    const s = settingsFor('wallpaper', loadRemembered('wallpaper'));
    return s.wallpaperTarget === 'phone' ? 'Phone wallpaper' : 'Wallpaper';
  }
  return INTENTS.find((i) => i.id === id)?.title || id;
}
