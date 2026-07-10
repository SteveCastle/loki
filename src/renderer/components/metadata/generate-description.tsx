import { useContext, useEffect, useId, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import './generate-description.css';
import {
  getCachedDefaultPrompt,
  setCachedDefaultPrompt,
  getLastCustomPrompt,
  setLastCustomPrompt,
  clearLastCustomPrompt,
} from './customPromptStore';
import { SparkleIcon, TuneIcon } from './section-action-icons';
import { useDepRequirement } from '../../onboarding/useDepRequirement';
import { mediaServerBase } from '../../platform';
import { createDescriptionJob } from './create-description-job';

type Props = {
  path: string;
  label?: string;
  variant?: 'centered' | 'corner';
};

const FALLBACK_PLACEHOLDER =
  'Describe this image, focusing on people, clothing, items, text, and actions.';

export default function GenerateDescription({
  path,
  label,
  variant = 'centered',
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const panelId = `gd-prompt-panel-${useId()}`;
  const jobServerAvailable = useJobServerAvailable(authToken);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [panelOpen, setPanelOpen] = useState<boolean>(false);
  const [promptDraft, setPromptDraft] = useState<string>(() =>
    getLastCustomPrompt()
  );
  const [defaultPrompt, setDefaultPrompt] = useState<string | null>(() =>
    getCachedDefaultPrompt()
  );
  // Non-blocking: descriptions can use other configured providers (LM Studio,
  // RunPod, llama.cpp), so a missing Ollama is only worth a hint, not a gate.
  const ollama = useDepRequirement('ollama');
  const ollamaHint =
    ollama.dep && ollama.dep.state === 'not_installed' ? (
      <div className="prompt-hint" style={{ marginTop: 4 }}>
        Uses your configured AI provider — Ollama not detected.{' '}
        <a href="https://ollama.com/download" target="_blank" rel="noreferrer">
          Get Ollama
        </a>{' '}
        if descriptions fail.
      </div>
    ) : null;

  // Lazily fetch the default prompt the first time the panel is opened, then
  // cache it module-wide so subsequent renders (and other component instances)
  // hit memory instead of the network.
  useEffect(() => {
    if (!panelOpen) return;
    if (defaultPrompt !== null) return;
    let cancelled = false;
    const load = async () => {
      try {
        const headers: HeadersInit = {};
        if (authToken) {
          headers['Authorization'] = `Bearer ${authToken}`;
        }
        const response = await fetch(
          `${mediaServerBase}/api/prompts/describe`,
          { headers, signal: AbortSignal.timeout(10000) }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = (await response.json()) as { prompt?: string };
        const prompt = body.prompt ?? FALLBACK_PLACEHOLDER;
        if (cancelled) return;
        setCachedDefaultPrompt(prompt);
        setDefaultPrompt(prompt);
      } catch (error) {
        if (cancelled) return;
        setCachedDefaultPrompt(FALLBACK_PLACEHOLDER);
        setDefaultPrompt(FALLBACK_PLACEHOLDER);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [panelOpen, defaultPrompt, authToken]);

  const handleGenerateDescription = async () => {
    try {
      setIsSubmitting(true);
      const trimmed = promptDraft.trim();
      if (trimmed !== '') {
        setLastCustomPrompt(trimmed);
      }
      await createDescriptionJob(path, authToken, trimmed);
      // Let the ToastSystem show job lifecycle
    } catch (error) {
      console.error('Failed to create description job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Forget the remembered override so this and future generates use the
  // configured default again. Clears both the live draft and the session store
  // so the reset survives the panel/component remounting.
  const handleResetPrompt = () => {
    setPromptDraft('');
    clearLastCustomPrompt();
  };

  const hasCustomPrompt = promptDraft.trim() !== '';

  const isCorner = variant === 'corner';

  // In the corner-pill context we stay quiet until the job service is
  // confirmed available — no placeholder, no warning box overlapping content.
  if (jobServerAvailable === null) {
    if (isCorner) return null;
    return (
      <div className={`GenerateDescription ${variant}`}>
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
    if (isCorner) return null;
    return (
      <div className={`GenerateDescription ${variant}`}>
        <div className="server-unavailable">
          <div className="icon">⚠️</div>
          <div className="message">
            <strong>Job Service Required</strong>
            <p>
              To generate descriptions and run other long-running tasks, you
              need to install and run the Lowkey Media Server job service.
            </p>
            <p>
              Start the service at <code>{mediaServerBase || 'this server'}</code>{' '}
              to enable this feature.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const placeholder =
    defaultPrompt ?? 'Loading default prompt…';

  const promptPanel = panelOpen ? (
    <div className="prompt-panel" id={panelId}>
      <textarea
        className="prompt-textarea"
        value={promptDraft}
        placeholder={placeholder}
        onChange={(e) => setPromptDraft(e.target.value)}
        onKeyDown={(e) => e.stopPropagation()}
        onKeyUp={(e) => e.stopPropagation()}
        rows={4}
      />
      <div className="prompt-footer">
        <div className="prompt-hint">
          Leave empty to use the configured default. This prompt is remembered
          for the rest of your session — your global config is unchanged.
        </div>
        {hasCustomPrompt && (
          <button
            type="button"
            className="prompt-reset"
            onClick={handleResetPrompt}
            title="Clear the custom prompt and use the configured default"
          >
            Reset to default
          </button>
        )}
      </div>
    </div>
  ) : null;

  // Corner-pill variant: the trigger row floats to the section's top-right
  // (via `.section-action`), while the optional prompt panel flows in below the
  // section content so it never overlaps text or gets clipped.
  if (isCorner) {
    return (
      <div className="GenerateDescription corner">
        <div className="section-action">
          <button
            className="section-action-pill"
            onClick={handleGenerateDescription}
            disabled={isSubmitting}
            title={label || 'Regenerate description'}
          >
            <SparkleIcon />
            <span>{label || 'Regenerate'}</span>
          </button>
          <button
            type="button"
            className="section-action-caret"
            onClick={() => setPanelOpen((v) => !v)}
            aria-expanded={panelOpen}
            aria-controls={panelId}
            aria-label="Customize prompt"
            title="Customize prompt"
          >
            <TuneIcon />
          </button>
        </div>
        {promptPanel}
      </div>
    );
  }

  return (
    <div className={`GenerateDescription ${variant}`}>
      <div className="generate-row">
        <button
          className="generate"
          onClick={handleGenerateDescription}
          disabled={isSubmitting}
        >
          {label || 'Generate Description'}
        </button>
        <button
          type="button"
          className="prompt-toggle"
          onClick={() => setPanelOpen((v) => !v)}
          aria-expanded={panelOpen}
          aria-controls={panelId}
        >
          {panelOpen ? 'Hide prompt' : 'Customize prompt'}
        </button>
      </div>
      {ollamaHint}
      {promptPanel}
    </div>
  );
}
