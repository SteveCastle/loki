import { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
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
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

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

      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          input: `metadata --type transcript --apply all "${path}"`,
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
              to install and run the Shrike job service.
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
