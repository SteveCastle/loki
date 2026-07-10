import { useContext, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useDepRequirement } from '../../onboarding/useDepRequirement';
import { fmtSize } from '../../onboarding/requirements';
import { mediaServerBase } from '../../platform';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import './generate-transcript.css';

type Props = {
  path: string;
  label?: string;
  variant?: 'centered' | 'inline';
};

export default function GenerateTranscript({
  path,
  label,
  variant = 'centered',
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const jobServerAvailable = useJobServerAvailable(authToken);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const whisper = useDepRequirement('faster-whisper');

  // No SSE subscription here; ToastSystem handles job progress globally

  const handleGenerateTranscript = async () => {
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

      const response = await fetch(`${mediaServerBase}/create`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          // The split-out transcribe task; --overwrite because this is an
          // explicit per-file request (regenerate even if one exists).
          input: `transcribe --overwrite "${path}"`,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Let the ToastSystem show job lifecycle
    } catch (error) {
      console.error('Failed to create transcript job:', error);
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
      <div className={`GenerateTranscript ${variant}`}>
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
    return (
      <div className={`GenerateTranscript ${variant}`}>
        <div className="server-unavailable">
          <div className="icon">⚠️</div>
          <div className="message">
            <strong>Job Service Required</strong>
            <p>
              To generate transcripts and run other long-running tasks, you need
              to install and run the Lowkey Media Server job service.
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

  // Transcription needs the Faster-Whisper tool; offer the one-time download
  // right here instead of letting the job fail in its log.
  if (whisper.needsDownload) {
    return (
      <div className={`GenerateTranscript ${variant}`}>
        <button
          className="generate"
          onClick={() => whisper.download().catch(() => {})}
          title="One-time download; transcription runs locally"
        >
          Download transcription tool ({fmtSize(whisper.dep?.size_bytes)})
        </button>
      </div>
    );
  }
  if (whisper.downloading) {
    return (
      <div className={`GenerateTranscript ${variant}`}>
        <button className="generate" disabled>
          Downloading transcription tool… {whisper.pct}%
        </button>
      </div>
    );
  }

  return (
    <div className={`GenerateTranscript ${variant}`}>
      <button
        className="generate"
        onClick={handleGenerateTranscript}
        disabled={isSubmitting}
      >
        {label || 'Generate Transcript'}
      </button>
    </div>
  );
}
