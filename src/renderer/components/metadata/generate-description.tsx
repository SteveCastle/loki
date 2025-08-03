import { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import { useQueryClient } from '@tanstack/react-query';
import './generate-description.css';

interface JobRunnerJob {
  id: string;
  command: string;
  arguments: string[];
  state: number; // 0=Pending, 1=InProgress, 2=Completed, 3=Cancelled, 4=Error
}

type Props = {
  path: string;
};

export default function GenerateDescription({ path }: Props) {
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
        
        // Only include jobs that are pending or in progress and are metadata commands with our path
        if ((job.state === 0 || job.state === 1) && 
            job.command === 'metadata' && 
            job.arguments && 
            Array.isArray(job.arguments) &&
            job.arguments.some(arg => arg && arg.includes && arg.includes('description')) &&
            job.arguments.some(arg => arg && arg.includes && arg.includes(path))) {
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
      
      // If description job completed, refresh the metadata
      if (job.state === 2 && job.command === 'metadata' && 
          job.arguments && job.arguments.some(arg => arg && arg.includes('description')) &&
          job.arguments.some(arg => arg && arg.includes(path))) {
        queryClient.invalidateQueries(['file-metadata', path]);
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

  const handleGenerateDescription = async () => {
    try {
      const response = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input: `metadata --type description --apply all "${path}"`
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Job created:', result.id);
      // No toast needed - job will appear in job toast list automatically
    } catch (error) {
      console.error('Failed to create description job:', error);
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

  if (jobServerAvailable === null) {
    return (
      <div className="GenerateDescription">
        <div className="checking-server">Checking job service...</div>
      </div>
    );
  }

  if (jobServerAvailable === false) {
    return (
      <div className="GenerateDescription">
        <div className="server-unavailable">
          <div className="icon">⚠️</div>
          <div className="message">
            <strong>Job Service Required</strong>
            <p>To generate descriptions and run other long-running tasks, you need to install and run the Shrike job service.</p>
            <p>Start the service at <code>localhost:8090</code> to enable this feature.</p>
          </div>
        </div>
      </div>
    );
  }

  const hasRunningJob = runningJobs.length > 0;

  if (hasRunningJob) {
    const job = runningJobs[0];
    const isInProgress = job.state === 1;
    
    return (
      <div className="GenerateDescription">
        <div className="job-running">
          <div className="loading-spinner"></div>
          <div className="job-status">
            {isInProgress ? 'Generating description...' : 'Description job queued...'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="GenerateDescription">
      <button className="generate" onClick={handleGenerateDescription}>
        Generate Description
      </button>
    </div>
  );
}