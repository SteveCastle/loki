import React, { useMemo, useState } from 'react';
import { parseGenerationMetadata } from '../../generation-metadata';
import { mediaServerBase } from '../../platform';
import { rerunFromRecord, suggestTaskName, taskFromRecord, type RebuiltTask, type Rerun } from '../transform/recipe';
import { buildJobs, intentById } from '../transform/intents';
import { EngineSetupDeferred } from '../transform/engine-setup';
import { saveTemplate, submitJobs } from '../transform/store';
import copyIcon from '../../../../assets/copy.svg';

import './generation-info.css';

/* The panel's "Generation" section: what the AI tool that made this file
 * wrote into it. The prompt is the thing people read; the workflow JSON is
 * the thing they take with them (drop it onto ComfyUI to rebuild the
 * graph), so it gets a one-click copy rather than a wall of text. */

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy: ', err);
    return false;
  }
}

function CopyButton({ text, label, title }: { text: string; label?: string; title: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className={'gen-copy' + (label ? ' labelled' : '')}
      title={title}
      onClick={async (e) => {
        e.stopPropagation();
        if (await copyText(text)) {
          setDone(true);
          setTimeout(() => setDone(false), 1200);
        }
      }}
    >
      <img src={copyIcon} alt="" />
      {label && <span>{done ? 'Copied' : label}</span>}
    </button>
  );
}

/* "Save as Transform task": the run that made this file, as a saved task to
 * run on other images (transform/recipe.ts). It lands with the prompts saved
 * from the Transform flow itself, in the palette's ☆ chips. */
function SaveTask({ task }: { task: RebuiltTask }) {
  const [naming, setNaming] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const title = intentById(task.intent).title;
  const commit = () => {
    if (naming === null) return;
    const t = saveTemplate(task.intent, naming, task.settings.prompt, task.settings);
    if (t) {
      setSaved(t.name);
      setNaming(null);
    }
  };
  const note = task.approximate
    ? 'Rebuilt from the engine settings (this file predates stamped recipes): check quality, size and seed before running.'
    : null;
  if (naming !== null) {
    return (
      <div className="gen-task">
        <input
          autoFocus
          value={naming}
          placeholder="Name"
          aria-label="Saved task name"
          onChange={(e) => setNaming(e.target.value)}
          onFocus={(e) => e.target.select()}
          onKeyDown={(e) => {
            // typing here must not reach the viewer's own shortcuts
            e.stopPropagation();
            if (e.key === 'Enter') {
              e.preventDefault();
              commit();
            } else if (e.key === 'Escape') {
              e.preventDefault();
              setNaming(null);
            }
          }}
        />
        <button type="button" className="gen-copy labelled" disabled={!naming.trim()} onClick={commit}>
          Save
        </button>
        <button type="button" className="gen-copy labelled" onClick={() => setNaming(null)}>
          Cancel
        </button>
        {note && <span className="gen-task-note">{note}</span>}
      </div>
    );
  }
  return (
    <div className="gen-task">
      <button
        type="button"
        className="gen-copy labelled"
        title={`Save this ${title} run (its prompt and settings) as a Transform task, to run on other images from the palette's ☆ chips`}
        onClick={() => {
          setSaved(null);
          setNaming(suggestTaskName(task));
        }}
      >
        <span>☆ Save as {title} task</span>
      </button>
      {saved && <span className="gen-dim">★ Saved as “{saved}” · under Transform in the palette</span>}
    </div>
  );
}

/** What queuing a job from the panel needs (the viewer's job service); absent = no Regenerate (read-only, no server). */
export interface JobAccess {
  authToken: string | null;
  notify: (type: 'error' | 'info', title: string, message: string) => void;
}

/* "Regenerate": the same run queued again in one click, with a fresh random seed. The result arrives like any other
 * Transform job's (job toast, then the library). */
function Regenerate({ rerun, jobs }: { rerun: Rerun; jobs: JobAccess }) {
  const { authToken, notify: toast } = jobs;
  const [state, setState] = useState<'idle' | 'busy' | 'queued'>('idle');
  const { task, paths } = rerun;
  const title = intentById(task.intent).title;
  const run = async () => {
    const built = buildJobs(task.intent, task.settings, paths, { recipe: true });
    if (built.error || built.jobs.length === 0) {
      toast('error', 'Cannot regenerate', built.error || 'Nothing to run');
      return;
    }
    setState('busy');
    try {
      await submitJobs(built.jobs, { mediaServerBase, authToken });
      setState('queued');
      setTimeout(() => setState('idle'), 2500);
    } catch (e) {
      setState('idle');
      if (e instanceof EngineSetupDeferred) {
        if (e.result === 'background') toast('info', 'Downloading', 'The download continues in the background. Regenerate again when it has finished.');
      } else {
        toast('error', 'Failed to Create Job', 'Could not communicate with job service');
      }
    }
  };
  return (
    <button
      type="button"
      className="gen-copy labelled"
      disabled={state === 'busy' || !authToken}
      title={
        `Queue this ${title} run again with the same prompt, inputs and settings and a new random seed` +
        (task.approximate ? ' (rebuilt from the engine settings: this file predates stamped recipes, so it may differ a little)' : '')
      }
      onClick={run}
    >
      <span>{state === 'queued' ? '✓ Queued' : state === 'busy' ? 'Queuing…' : '↻ Regenerate'}</span>
    </button>
  );
}

const sizeOf = (text: string) => {
  const bytes = new Blob([text]).size;
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

export default function GenerationInfo({ chunks, jobs }: { chunks?: Record<string, string> | null; jobs?: JobAccess }) {
  const info = useMemo(() => parseGenerationMetadata(chunks), [chunks]);
  const task = useMemo(() => (info?.lokiRecord ? taskFromRecord(info.lokiRecord) : null), [info]);
  const rerun = useMemo(() => (info?.lokiRecord ? rerunFromRecord(info.lokiRecord) : null), [info]);
  const [showRaw, setShowRaw] = useState(false);
  if (!info) return null;

  const copyables: Array<{ key: string; label: string; title: string; text: string }> = [];
  if (info.workflowJson)
    copyables.push({ key: 'workflow', label: 'Copy workflow JSON', title: `the full ComfyUI workflow (${sizeOf(info.workflowJson)}) — paste or drop it into ComfyUI to rebuild the graph`, text: info.workflowJson });
  if (info.promptJson)
    copyables.push({ key: 'prompt', label: 'Copy API prompt JSON', title: `the ComfyUI API-format prompt (${sizeOf(info.promptJson)}) — what the server actually ran`, text: info.promptJson });
  if (info.parametersText)
    copyables.push({ key: 'parameters', label: 'Copy parameters', title: 'the whole WebUI parameters block — paste it into the PNG Info tab', text: info.parametersText });
  if (info.commandLine)
    copyables.push({ key: 'command', label: 'Copy command', title: 'the command line that made this file — run it again, or change a flag', text: info.commandLine });
  if (info.recordJson)
    copyables.push({ key: 'record', label: 'Copy generation JSON', title: `the full generation record (${sizeOf(info.recordJson)}) as the tool wrote it`, text: info.recordJson });

  const rawEntries = Object.entries(info.raw);

  return (
    <div className="section GenerationInfo">
      <h2>Generation · {info.generator}{info.tool ? ` · ${info.tool}` : ''}</h2>

      {info.prompt !== undefined && (
        <div className="gen-block">
          <div className="gen-label">
            Prompt
            <CopyButton text={info.prompt} title="copy the prompt" />
          </div>
          <div className="gen-text">{info.prompt || <span className="gen-empty">(empty)</span>}</div>
        </div>
      )}
      {info.negativePrompt && (
        <div className="gen-block">
          <div className="gen-label">
            Negative prompt
            <CopyButton text={info.negativePrompt} title="copy the negative prompt" />
          </div>
          <div className="gen-text">{info.negativePrompt}</div>
        </div>
      )}

      {info.params.length > 0 && (
        <div className="gen-params">
          {info.params.map(([label, value]) => (
            <div className="gen-param" key={label} title={`${label}: ${value}`}>
              <span className="gen-param-key">{label}</span>
              <span className="gen-param-val">{value}</span>
            </div>
          ))}
        </div>
      )}

      {info.commandLine && (
        <div className="gen-block">
          <div className="gen-label">
            Command
            <CopyButton text={info.commandLine} title="copy the command line" />
          </div>
          <div className="gen-text gen-mono">{info.commandLine}</div>
        </div>
      )}

      {(task || (rerun && jobs)) && (
        <div className="gen-run-actions">
          {rerun && jobs && <Regenerate key={`regen:${info.recordJson}`} rerun={rerun} jobs={jobs} />}
          {task && <SaveTask key={`save:${info.recordJson}`} task={task} />}
        </div>
      )}

      {copyables.length > 0 && (
        <div className="gen-actions">
          {copyables.map((c) => (
            <CopyButton key={c.key} text={c.text} label={c.label} title={c.title} />
          ))}
        </div>
      )}

      {rawEntries.length > 0 && (
        <div className="gen-raw">
          <button type="button" className="gen-raw-toggle" onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? '▾' : '▸'} embedded text ({rawEntries.length})
          </button>
          {showRaw &&
            rawEntries.map(([key, text]) => (
              <div className="gen-block" key={key}>
                <div className="gen-label">
                  {key} <span className="gen-dim">{sizeOf(text)}</span>
                  <CopyButton text={text} title={`copy the raw "${key}" text`} />
                </div>
                <div className="gen-text gen-mono">
                  {text.length > 2000 ? `${text.slice(0, 2000)}…` : text}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
