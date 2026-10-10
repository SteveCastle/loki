import React, { useState } from 'react';
import { mediaServerBase } from '../../platform';
import {
  INTENTS,
  availability,
  buildJobs,
  classify,
  intentById,
  intentKind,
  settingsFor,
  type IntentId,
} from './intents';
import { EngineSetupDeferred } from './engine-setup';
import { loadRemembered, openTransformFlow, submitJobs, useTemplates } from './store';
import { IntentIcon } from './icons';
import './transform-section.css';

// The palette's "Transform" row: one-click presets (run with the settings you
// used last time), the custom intents (open the flow at their prompt), the
// user's saved prompts, and "All options" for the full flow. Shift-click a
// preset to adjust it first.
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
  const savedEdit = useTemplates('edit');
  const savedDirect = useTemplates('direct');
  const saved = [
    ...(availability('edit', inputs).ok ? savedEdit.map((t) => ({ intent: 'edit' as IntentId, t })) : []),
    ...(availability('direct', inputs).ok ? savedDirect.map((t) => ({ intent: 'direct' as IntentId, t })) : []),
  ];

  const open = (intent: IntentId | undefined, prompt?: string) => {
    openTransformFlow({
      paths,
      videoTime: getVideoTime(),
      intent,
      phase: intent ? 'shape' : 'choose',
      prompt,
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
          title="Every preset, custom prompt and option, step by step"
        >
          All options ▸
        </button>
      </div>
      <div className="type-chips" role="group" aria-label="Transform">
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
        {saved.map(({ intent, t }) => (
          <button
            key={t.id}
            type="button"
            className="type-chip transform-chip transform-chip-saved"
            disabled={busy !== null}
            onClick={() => open(intent, t.prompt)}
            title={`Saved prompt (${intentById(intent).title}): opens it to check and run\n${t.prompt}`}
          >
            ☆ {t.name}
          </button>
        ))}
      </div>
      <span className="merge-selection-note">
        {hasVideoOnly ? 'acts on the frame you are looking at · ' : ''}
        presets run with your last settings · Shift-click to adjust first{saved.length ? ' · ☆ your saved prompts' : ''}
      </span>
    </div>
  );
}

function chipTitle(id: IntentId): string {
  const def = intentById(id);
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
  if (id === 'alive') return 'Bring to life…';
  if (id === 'wallpaper') {
    const s = settingsFor('wallpaper', loadRemembered('wallpaper'));
    return s.wallpaperTarget === 'phone' ? 'Phone wallpaper' : 'Wallpaper';
  }
  return INTENTS.find((i) => i.id === id)?.title || id;
}
