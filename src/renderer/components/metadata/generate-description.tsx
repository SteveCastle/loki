import { useContext, useEffect, useId, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import './generate-description.css';
import {
  getCachedDefaultPrompt,
  setCachedDefaultPrompt,
  getLastCustomPrompt,
  setLastCustomPrompt,
} from './customPromptStore';

type Props = {
  path: string;
  label?: string;
  variant?: 'centered' | 'inline';
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
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [panelOpen, setPanelOpen] = useState<boolean>(false);
  const [promptDraft, setPromptDraft] = useState<string>(() =>
    getLastCustomPrompt()
  );
  const [defaultPrompt, setDefaultPrompt] = useState<string | null>(() =>
    getCachedDefaultPrompt()
  );

  useEffect(() => {
    const checkJobServer = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const headers: HeadersInit = {};
        if (authToken) {
          headers['Authorization'] = `Bearer ${authToken}`;
        }

        const response = await fetch('http://localhost:8090/health', {
          method: 'GET',
          headers,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        setJobServerAvailable(response.ok);
      } catch (error) {
        setJobServerAvailable(false);
      }
    };

    checkJobServer();
  }, [authToken]);

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
          'http://localhost:8090/api/prompts/describe',
          { headers }
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
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const trimmed = promptDraft.trim();
      const body: { input: string; fields?: { prompt: string } } = {
        input: `metadata --type description --apply all --overwrite "${path}"`,
      };
      if (trimmed !== '') {
        body.fields = { prompt: trimmed };
        setLastCustomPrompt(trimmed);
      }

      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

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

  if (jobServerAvailable === null) {
    return (
      <div className={`GenerateDescription ${variant}`}>
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
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
              Start the service at <code>localhost:8090</code> to enable this
              feature.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const placeholder =
    defaultPrompt ?? 'Loading default prompt…';

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
      {panelOpen && (
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
          <div className="prompt-hint">
            Leave empty to use the configured default. This prompt is remembered
            for the rest of your session — your global config is unchanged.
          </div>
        </div>
      )}
    </div>
  );
}
