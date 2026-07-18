import { useContext, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useDepRequirement } from '../../onboarding/useDepRequirement';
import { fmtSize } from '../../onboarding/requirements';
import { mediaServerBase } from '../../platform';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import useJobsForPath from '../../hooks/useJobsForPath';
import { jobStatusLabel, pickActiveJob } from '../../job-status';
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
  const canWrite = useSelector(
    libraryService,
    (state) => state.context.canWrite
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const whisper = useDepRequirement('faster-whisper');

  // Path→job index lookup: while a transcribe job is queued/running for this
  // file, the button is replaced by a live status indicator. State/progress
  // updates arrive over the shared SSE bus; toasts stay ToastSystem's job.
  const { jobs, noteJob } = useJobsForPath(
    path,
    authToken,
    canWrite && jobServerAvailable === true
  );
  const activeJob = pickActiveJob(jobs, ['transcribe']);

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

      // Flip to the status indicator immediately; SSE updates take over from
      // here. The ToastSystem still shows the global job lifecycle.
      try {
        const created = (await response.json()) as { id?: string };
        if (created.id) {
          noteJob({ id: created.id, command: 'transcribe', state: 'pending' });
        }
      } catch {
        // No id in the response — the SSE 'create' refetch will pick it up.
      }
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

  // Cancel the active job via the server's existing cancel endpoint; the SSE
  // 'update' event (state → cancelled) is what clears the indicator.
  const handleCancelJob = async (jobId: string) => {
    try {
      setIsCancelling(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const headers: HeadersInit = {};
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }
      const response = await fetch(`${mediaServerBase}/job/${jobId}/cancel`, {
        method: 'POST',
        headers,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    } catch (error) {
      console.error('Failed to cancel transcript job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Cancel Job',
          message: 'Could not communicate with job service',
        },
      });
    } finally {
      setIsCancelling(false);
    }
  };

  if (!canWrite) return null;

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

  // A transcribe job is already queued/running for this file — show its
  // status (with a cancel affordance) instead of offering to submit another.
  if (activeJob) {
    return (
      <div className={`GenerateTranscript ${variant}`}>
        <div className="job-running">
          <div className="loading-spinner" />
          <span className="job-status">
            {jobStatusLabel(activeJob, 'Transcribing')}
          </span>
          <button
            type="button"
            className="job-cancel"
            onClick={() => handleCancelJob(activeJob.id)}
            disabled={isCancelling}
            title="Cancel this job — work finished so far is kept"
          >
            {isCancelling ? 'Cancelling…' : 'Cancel'}
          </button>
        </div>
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
