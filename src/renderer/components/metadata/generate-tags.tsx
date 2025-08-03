import { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import { useQueryClient } from '@tanstack/react-query';
import './generate-tags.css';

interface JobRunnerJob {
  id: string;
  command: string;
  arguments: string[];
  input: string;
  state: number; // 0=Pending, 1=InProgress, 2=Completed, 3=Cancelled, 4=Error
}

type Props = {
  path: string;
};

export default function GenerateTags({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean | null>(null);
  const [runningJobs, setRunningJobs] = useState<JobRunnerJob[]>([]);

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

  useEffect(() => {
    if (!jobServerAvailable) return;

    const eventSource = new EventSource('http://localhost:8090/stream');

    const updateRunningJobs = (job: JobRunnerJob) => {
      setRunningJobs(prev => {
        const filtered = prev.filter(j => j.id !== job.id);
        
        // Only include jobs that are pending or in progress and are metadata commands with autotag and our path
        if ((job.state === 0 || job.state === 1) && 
            job.command === 'metadata' && 
            job.arguments && 
            Array.isArray(job.arguments) &&
            job.arguments.some(arg => arg && arg.includes && arg.includes('autotag')) &&
            job.input && job.input.includes(path)) {
          return [...filtered, job];
        }
        
        return filtered;
      });
    };

    eventSource.addEventListener('create', (event) => {
      const data = JSON.parse(event.data);
      updateRunningJobs(data.job);
    });

    eventSource.addEventListener('update', (event) => {
      const data = JSON.parse(event.data);
      const job = data.job;
      updateRunningJobs(job);
      
      // If tag generation job completed, refresh the tags and metadata
      if (job.state === 2 && job.command === 'metadata' && 
          job.arguments && job.arguments.some(arg => arg && arg.includes('autotag')) &&
          job.input && job.input.includes(path)) {
        queryClient.invalidateQueries(['tags-by-path', path]);
        queryClient.invalidateQueries(['file-metadata', path]);
        queryClient.invalidateQueries(['metadata']);
      }
    });

    eventSource.addEventListener('delete', (event) => {
      const data = JSON.parse(event.data);
      setRunningJobs(prev => prev.filter(j => j.id !== data.job.id));
    });

    return () => {
      eventSource.close();
    };
  }, [jobServerAvailable, path, queryClient]);

  const handleGenerateTags = async () => {
    try {
      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: `metadata --type transcript,autotag --apply all --model qwen2.5vl:32b "${path}"`
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
          message: 'Could not communicate with job service' 
        } 
      });
    }
  };

  // Don't show anything if server is not available
  if (jobServerAvailable !== true) {
    return null;
  }

  const hasRunningJob = runningJobs.length > 0;

  if (hasRunningJob) {
    const job = runningJobs[0];
    const isInProgress = job.state === 1;
    
    return (
      <div className="GenerateTags">
        <div className="job-running">
          <div className="loading-spinner"></div>
          <div className="job-status">
            {isInProgress ? 'Generating tags...' : 'Tag generation queued...'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="GenerateTags">
      <button className="generate" onClick={handleGenerateTags}>
        Generate Tags
      </button>
    </div>
  );
}