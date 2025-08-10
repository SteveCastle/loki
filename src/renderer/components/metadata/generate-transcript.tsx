import { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import './generate-transcript.css';

type Props = {
  path: string;
};

export default function GenerateTranscript({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  useEffect(() => {
    const checkJobServer = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        const response = await fetch('http://localhost:8090/health', {
          method: 'GET',
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        setJobServerAvailable(response.ok);
      } catch (error) {
        setJobServerAvailable(false);
      }
    };

    checkJobServer();
  }, []);

  // No SSE subscription here; ToastSystem handles job progress globally

  const handleGenerateTranscript = async () => {
    try {
      setIsSubmitting(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
      <div className="GenerateTranscript">
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
    return (
      <div className="GenerateTranscript">
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
    <div className="GenerateTranscript">
      <button
        className="generate"
        onClick={handleGenerateTranscript}
        disabled={isSubmitting}
      >
        Generate Transcript
      </button>
    </div>
  );
}
