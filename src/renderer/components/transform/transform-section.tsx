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
import { GlyphReplay, GlyphScene, IntentIcon } from './icons';
import { canStartScene, openSceneEditor } from '../scene/launch';
import './transform-section.css';

// The palette's "Transform" row: one-click presets (run with the settings you
// used last time), the custom intents (open the flow at their prompt), the last run (“… again”: same action and settings on this
// selection), and "All options" for the full flow. Shift-click a preset or
// the last run to adjust it first.
// Generate needs no selection, so the row is there (with just Generate) even when nothing is selected.
const CHIP_ORDER: IntentId[] = ['restore', 'upscale', 'wallpaper', 'angle', 'edit', 'reimagine', 'alive', 'generate'];

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
  const chips = CHIP_ORDER.map(intentById).filter((i) => availability(i.id, inputs).ok);
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

  if (chips.length === 0) return null;
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
      <div className="type-chips" role="group" aria-label="Transform">
        {last && lastOk && (
          <button
            type="button"
            className="type-chip transform-chip transform-chip-last"
            disabled={busy !== null}
            onClick={onLast}
            title={`Last run: ${intentById(last.intent).title} · ${runSummary(last.intent, last.settings)}\nClick to run it again on this selection, Shift-click to adjust first.`}
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
        {paths.length === 1 && canStartScene(paths[0]) && (
          <button
            type="button"
            className="type-chip transform-chip"
            disabled={busy !== null}
            onClick={() => {
              openSceneEditor(paths[0], getVideoTime());
              onDone();
            }}
            title="Scene editor (opens a window): mask parts of the picture, edit them, then animate them one by one"
          >
            <GlyphScene size={13} />
            Scene…
          </button>
        )}
      </div>
      <span className="merge-selection-note">
        {paths.length === 0 ? 'nothing selected: Generate makes a new picture from a description · ' : ''}
        {hasVideoOnly ? 'acts on the frame you are looking at · ' : ''}
        {last && lastOk ? '“again” repeats your last run · ' : ''}presets run with your last settings · Shift-click to adjust first
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
