import { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import './generate-tags.css';

type Props = {
  path: string;
};

export default function GenerateTags({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );

  useEffect(() => {
    const checkJobServer = async () => {
      try {
        const headers: HeadersInit = {};
        if (authToken) {
          headers['Authorization'] = `Bearer ${authToken}`;
        }

        const response = await fetch('http://localhost:8090/health', {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(3000), // 3 second timeout
        });
        setJobServerAvailable(response.ok);
      } catch (error) {
        setJobServerAvailable(false);
      }
    };

    checkJobServer();
  }, [authToken]);

  const handleGenerateTags = async () => {
    try {
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
          input: `autotag "${path}"`,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Tag generation job created:', result.id);
      // No toast needed - job will appear in job toast list automatically
    } catch (error) {
      console.error('Failed to create tag generation job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    }
  };

  // Don't show anything if server is not available
  if (jobServerAvailable !== true) {
    return null;
  }

  return (
    <div className="GenerateTags">
      <button className="generate" onClick={handleGenerateTags}>
        Generate Tags
      </button>
    </div>
  );
}
