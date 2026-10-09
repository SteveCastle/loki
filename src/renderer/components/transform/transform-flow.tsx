import React, { useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { mediaServerBase, mediaUrl } from '../../platform';
import {
  buildJobs,
  classify,
  intentById,
  mediaKind,
  settingsFor,
  type IntentId,
  type TransformSettings,
} from './intents';
import {
  closeTransformFlow,
  loadRemembered,
  saveRemembered,
  submitJobs,
  useFlowRequest,
  type FlowPhase,
} from './store';
import { EngineSetupDeferred } from './engine-setup';
import { TransformFlowView, validateForRun } from './flow-view';

// ---------------------------------------------------------------------------
// Host (container): owns phase/intent/settings state, talks to the server
// ---------------------------------------------------------------------------

function useSourceSize(path: string | undefined, thumbUrl: (p: string) => string) {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  useEffect(() => {
    setSize(null);
    if (!path) return undefined;
    const kind = mediaKind(path);
    if (kind === 'image') {
      let alive = true;
      const img = new Image();
      img.onload = () => alive && setSize({ width: img.naturalWidth, height: img.naturalHeight });
      img.src = thumbUrl(path);
      return () => {
        alive = false;
      };
    }
    if (kind === 'video') {
      let alive = true;
      const v = document.createElement('video');
      v.preload = 'metadata';
      v.onloadedmetadata = () => alive && v.videoWidth && setSize({ width: v.videoWidth, height: v.videoHeight });
      v.src = thumbUrl(path);
      return () => {
        alive = false;
        v.removeAttribute('src');
      };
    }
    return undefined;
  }, [path, thumbUrl]);
  return size;
}

export default function TransformFlow() {
  const req = useFlowRequest();
  if (!req) return null;
  return <FlowSession key={req.paths.join('|') + (req.intent || '')} req={req} />;
}

function FlowSession({ req }: { req: NonNullable<ReturnType<typeof useFlowRequest>> }) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(libraryService, (state) => state.context.authToken);
  const [paths, setPaths] = useState<string[]>(req.paths);
  const [intent, setIntent] = useState<IntentId | null>(req.intent || null);
  const [phase, setPhase] = useState<FlowPhase>(req.phase || (req.intent ? 'shape' : 'choose'));
  const [settings, setSettings] = useState<TransformSettings>(() => settingsFor(req.intent || 'restore', req.intent ? loadRemembered(req.intent) : null));
  const [status, setStatus] = useState<'idle' | 'submitting' | 'done' | 'error'>('idle');
  const [error, setError] = useState<string | undefined>();
  const [queuedCount, setQueuedCount] = useState(0);
  const [queueAhead, setQueueAhead] = useState(0);
  const thumbUrl = useCallback((p: string) => mediaUrl(p), []);

  const inputs = useMemo(() => classify(paths), [paths]);
  const first = inputs.images[0] || inputs.videos[0];
  const source = useSourceSize(first, thumbUrl);

  const chooseIntent = useCallback(
    (id: IntentId) => {
      setIntent(id);
      // Keep what the user typed if they switch between text intents; reset the rest.
      setSettings((prev) => ({ ...settingsFor(id, loadRemembered(id)), prompt: prev.prompt, describe: prev.describe }));
      setPhase('shape');
    },
    []
  );

  const patch = useCallback((p: Partial<TransformSettings>) => setSettings((prev) => ({ ...prev, ...p })), []);

  // GPU queue depth, fetched when the review opens.
  useEffect(() => {
    if (phase !== 'review' || !authToken) return undefined;
    const ctl = new AbortController();
    (async () => {
      try {
        const res = await fetch(`${mediaServerBase}/jobs/list`, { headers: { Authorization: `Bearer ${authToken}` }, signal: ctl.signal });
        if (!res.ok) return;
        const jobs = (await res.json()) as Array<{ command: string; state: string }>;
        setQueueAhead(jobs.filter((j) => ['retouch', 'reshoot', '4kify'].includes(j.command) && (j.state === 'pending' || j.state === 'in_progress')).length);
      } catch {
        // unknown: show nothing
      }
    })();
    return () => ctl.abort();
  }, [phase, authToken]);

  const run = useCallback(async () => {
    if (!intent) return;
    const problem = validateForRun(intent, settings, paths);
    if (problem) {
      setError(problem);
      setPhase('shape');
      return;
    }
    const built = buildJobs(intent, settings, paths, { videoTime: req.videoTime });
    if (built.error || built.jobs.length === 0) {
      setError(built.error || 'Nothing to run');
      return;
    }
    setStatus('submitting');
    setError(undefined);
    try {
      await submitJobs(built.jobs, { mediaServerBase, authToken });
      saveRemembered(intent, settings);
      setQueuedCount(built.jobs.length);
      setStatus('done');
    } catch (e) {
      if (e instanceof EngineSetupDeferred) {
        // Declined or left downloading: nothing was queued, stay on the review step.
        setStatus('idle');
        if (e.result === 'background') {
          libraryService.send({
            type: 'ADD_TOAST',
            data: { type: 'info', title: 'Downloading', message: 'The download continues in the background. Run this again when it has finished.' },
          });
        }
        return;
      }
      setStatus('error');
      setError(`Could not queue the job (${e instanceof Error ? e.message : String(e)}). Is the media server running?`);
    }
  }, [intent, settings, paths, authToken, req.videoTime, libraryService]);

  // Esc closes; Ctrl+Enter runs from the review phase.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        closeTransformFlow();
      } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && phase === 'review' && status !== 'submitting' && status !== 'done') {
        e.preventDefault();
        e.stopPropagation();
        run();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [phase, status, run]);

  const view = (
    <TransformFlowView
      paths={paths}
      intent={intent}
      phase={phase}
      settings={settings}
      source={source}
      videoTime={req.videoTime}
      queueAhead={queueAhead}
      status={status}
      error={error}
      queuedCount={queuedCount}
      thumbUrl={thumbUrl}
      onSettings={patch}
      onIntent={chooseIntent}
      onPhase={(p) => {
        setError(undefined);
        setPhase(p);
      }}
      onRun={run}
      onClose={closeTransformFlow}
      onRemovePath={(p) => setPaths((prev) => (prev.length > 1 ? prev.filter((x) => x !== p) : prev))}
      onAgain={() => {
        setStatus('idle');
        setPhase('shape');
      }}
    />
  );
  return createPortal(view, document.body);
}
