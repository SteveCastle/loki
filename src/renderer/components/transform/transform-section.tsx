import React, { useState } from 'react';
import { mediaServerBase } from '../../platform';
import {
  INTENTS,
  availability,
  buildJobs,
  classify,
  intentById,
  settingsFor,
  type IntentId,
} from './intents';
import { EngineSetupDeferred } from './engine-setup';
import { loadRemembered, openTransformFlow, submitJobs } from './store';
import { IntentIcon } from './icons';
import './transform-section.css';

// The palette's "Transform" row: one-click presets (run with the settings you
// used last time) and an entry to the Transform flow for everything that
// needs a prompt or finer control. Shift-click any chip to tune it first.
const CHIP_ORDER: IntentId[] = ['restore', 'upscale', 'wallpaper', 'edit', 'alive'];

export interface TransformSectionProps {
  /** The working set: the right-clicked file plus any multi-selection. */
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
  const [busy, setBusy] = useState<IntentId | null>(null);
  const inputs = classify(paths);
  const chips = CHIP_ORDER.map(intentById).filter((i) => availability(i.id, inputs).ok);
  const hasVideoOnly = inputs.images.length === 0 && inputs.videos.length === 1;

  const open = (intent: IntentId | undefined) => {
    openTransformFlow({
      paths,
      videoTime: getVideoTime(),
      intent,
      phase: intent ? 'shape' : 'choose',
    });
    onDone();
  };

  const quickRun = async (id: IntentId) => {
    const settings = settingsFor(id, loadRemembered(id));
    const built = buildJobs(id, settings, paths, { videoTime: getVideoTime() });
    if (built.error || built.jobs.length === 0) {
      notify('error', 'Cannot run', built.error || 'Nothing to run');
      return;
    }
    setBusy(id);
    try {
      await submitJobs(built.jobs, { mediaServerBase, authToken });
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

  if (chips.length === 0) return null;
  return (
    <div className="context-palette-merge transform-section">
      <div className="workflow-picker-header">
        <span className="action-group-title">Transform</span>
        <button
          type="button"
          className="person-rename-btn"
          onClick={() => open(undefined)}
          title="Customize: every option, step by step"
        >
          Customize ▸
        </button>
      </div>
      <div className="type-chips" role="group" aria-label="Transform presets">
        {chips.map((c) => (
          <button
            key={c.id}
            type="button"
            className="type-chip transform-chip"
            disabled={busy !== null}
            onClick={(e) => onChip(e, c.id)}
            title={`${c.tagline}${c.oneClick ? ' — click to run, Shift-click to tune' : ' — opens the full options'}`}
          >
            <IntentIcon id={c.id} size={13} />
            {busy === c.id ? 'Queuing…' : chipLabel(c.id, hasVideoOnly)}
          </button>
        ))}
      </div>
      <span className="merge-selection-note">
        {hasVideoOnly ? 'acts on the frame you are looking at · ' : ''}
        click runs with your last settings · Shift-click to tune
      </span>
    </div>
  );
}

function chipLabel(id: IntentId, videoFrame: boolean): string {
  if (id === 'upscale') {
    const s = settingsFor('upscale', loadRemembered('upscale'));
    return `Upscale ${s.scale}×`;
  }
  if (id === 'edit') return videoFrame ? 'Edit frame…' : 'Edit…';
  if (id === 'alive') return 'Bring to life…';
  if (id === 'wallpaper') {
    const s = settingsFor('wallpaper', loadRemembered('wallpaper'));
    return s.wallpaperTarget === 'phone' ? 'Phone wallpaper' : 'Wallpaper';
  }
  return INTENTS.find((i) => i.id === id)?.title || id;
}
