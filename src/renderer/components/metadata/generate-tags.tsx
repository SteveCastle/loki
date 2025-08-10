import { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import './generate-tags.css';

type Props = {
  path: string;
};

export default function GenerateTags({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(
    null
  );

  useEffect(() => {
    const checkJobServer = async () => {
      try {
        const response = await fetch('http://localhost:8090/health', {
          method: 'GET',
          signal: AbortSignal.timeout(3000), // 3 second timeout
        });
        setJobServerAvailable(response.ok);
      } catch (error) {
        setJobServerAvailable(false);
      }
    };

    checkJobServer();
  }, []);

  const handleGenerateTags = async () => {
    try {
      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
