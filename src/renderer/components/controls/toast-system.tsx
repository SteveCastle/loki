import React, { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import './toast-system.css';

interface JobRunnerJob {
  id: string;
  command: string;
  arguments: string[];
  input: string;
  state: number; // 0=Pending, 1=InProgress, 2=Completed, 3=Cancelled, 4=Error
  created_at: string;
  claimed_at?: string;
  completed_at?: string;
  errored_at?: string;
}

interface Toast {
  id: string;
  type: 'success' | 'error' | 'info';
  title: string;
  message?: string;
  timestamp: number;
}

interface JobToastProps {
  job: JobRunnerJob;
  onClear: () => void;
}

const getJobStatus = (state: number): string => {
  switch (state) {
    case 0:
      return 'pending';
    case 1:
      return 'started';
    case 2:
      return 'complete';
    case 3:
      return 'cancelled';
    case 4:
      return 'error';
    default:
      return 'pending';
  }
};

const getJobTitle = (command: string): string => {
  switch (command) {
    case 'wait':
      return 'Wait';
    case 'gallery-dl':
      return 'Gallery Download';
    case 'dce':
      return 'DCE';
    case 'yt-dlp':
      return 'Video Download';
    case 'ffmpeg':
      return 'Media Processing';
    case 'remove':
      return 'Remove Media';
    case 'cleanup':
      return 'Cleanup';
    case 'ingest':
      return 'Ingest Media';
    case 'metadata':
      return 'Generate Metadata';
    case 'move':
      return 'Move Media';
    case 'autotag':
      return 'Autotag Media';
    default:
      return command;
  }
};

const JobToast: React.FC<JobToastProps> = ({ job, onClear }) => {
  const { libraryService } = useContext(GlobalStateContext);
  const library = useSelector(libraryService, (state) => state.context.library);
  const status = getJobStatus(job.state);
  const title = getJobTitle(job.command);

  // Extract file path from input - look for quoted paths or file-like arguments
  const extractFilePath = (input: string): string | null => {
    // Match quoted paths first (most reliable)
    const quotedMatch = input.match(/"([^"]+)"/);
    if (quotedMatch) {
      return quotedMatch[1];
    }

    // Fallback: look for path-like strings (contain / or \ or have file extensions)
    const pathMatch = input.match(
      /([^\s]+(?:\/|\\)[^\s]*|[^\s]*\.[a-zA-Z0-9]{2,4})/
    );
    if (pathMatch) {
      return pathMatch[1];
    }

    return null;
  };

  const filePath = extractFilePath(job.input);

  return (
    <div className="toast job-toast">
      <div className="toast-content">
        <div
          className={[
            'loading-animation',
            status === 'started' ? 'started' : '',
            status === 'pending' ? 'pending' : '',
            status === 'error' ? 'error' : '',
            status === 'complete' ? 'complete' : '',
          ].join(' ')}
        ></div>
        <div className="toast-text">
          <span className="toast-title">{title}</span>
          {filePath && (
            <div className="toast-file-path-container">
              <span
                className="toast-file-path"
                onClick={() => {
                  const isInLibrary =
                    Array.isArray(library) &&
                    library.some((item) => item?.path === filePath);
                  if (isInLibrary) {
                    libraryService.send('RESET_CURSOR', {
                      currentItem: { path: filePath },
                    });
                  } else {
                    libraryService.send('SET_FILE', { path: filePath });
                  }
                }}
                title={filePath}
              >
                {filePath}
              </span>
            </div>
          )}
        </div>
      </div>
      <div className="toast-close" onClick={onClear}>
        ×
      </div>
    </div>
  );
};

interface ActionToastProps {
  toast: Toast;
  onClear: () => void;
}

const ActionToast: React.FC<ActionToastProps> = ({ toast, onClear }) => {
  useEffect(() => {
    // Auto-dismiss action toasts after 4 seconds
    const timer = setTimeout(() => {
      onClear();
    }, 4000);

    return () => clearTimeout(timer);
  }, [onClear]);

  return (
    <div className={`toast action-toast ${toast.type}`}>
      <div className="toast-content">
        <div className={`toast-indicator ${toast.type}`}></div>
        <div className="toast-text">
          <span className="toast-title">{toast.title}</span>
          {toast.message && (
            <span className="toast-message">{toast.message}</span>
          )}
        </div>
      </div>
      <div className="toast-close" onClick={onClear}>
        ×
      </div>
    </div>
  );
};

export function ToastSystem() {
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const [jobs, setJobs] = useState<Map<string, JobRunnerJob>>(new Map());
  const [jobServerAvailable, setJobServerAvailable] = useState<boolean>(false);

  const toasts = useSelector(
    libraryService,
    (state) => state.context.toasts || []
  );

  // Check if job server is available before attempting SSE connection
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

    // Recheck every 30 seconds if server is not available
    const interval = setInterval(() => {
      if (!jobServerAvailable) {
        checkJobServer();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [jobServerAvailable]);

  useEffect(() => {
    if (!jobServerAvailable) {
      return; // Don't attempt SSE connection if server is not available
    }

    const eventSource = new EventSource('http://localhost:8090/stream');

    eventSource.addEventListener('create', (event) => {
      const data = JSON.parse(event.data);
      const job = data.job as JobRunnerJob;
      console.log('Job created:', job);
      setJobs((prev) => new Map(prev).set(job.id, job));
      // No toast for job creation - the job toast itself shows the status
    });

    eventSource.addEventListener('update', (event) => {
      const data = JSON.parse(event.data);
      const job = data.job as JobRunnerJob;
      console.log('Job updated:', job);
      setJobs((prev) => new Map(prev).set(job.id, job));

      // Handle job completion
      if (job.state === 2) {
        // Completed
        // Invalidate queries for metadata jobs
        if (job.command === 'metadata' || job.command === 'ingest') {
          queryClient.invalidateQueries(['transcript']);
          queryClient.invalidateQueries(['media']);
          queryClient.invalidateQueries(['file-metadata']);
          queryClient.invalidateQueries(['tags-by-path']);
          // If this metadata job was an autotag job, also invalidate taxonomy queries
        }

        if (job.command === 'autotag') {
          queryClient.invalidateQueries(['tags-by-path']);
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        }

        // Auto-remove completed jobs after 3 seconds to show completion state briefly
        setTimeout(() => {
          setJobs((prev) => {
            const newJobs = new Map(prev);
            newJobs.delete(job.id);
            return newJobs;
          });
        }, 3000);
      } else if (job.state === 4) {
        // Error - also auto-remove after 5 seconds (longer to let user see the error)
        setTimeout(() => {
          setJobs((prev) => {
            const newJobs = new Map(prev);
            newJobs.delete(job.id);
            return newJobs;
          });
        }, 5000);
      }
    });

    eventSource.addEventListener('delete', (event) => {
      const data = JSON.parse(event.data);
      const jobId = data.job.id;
      setJobs((prev) => {
        const newJobs = new Map(prev);
        newJobs.delete(jobId);
        return newJobs;
      });
    });

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      // Mark server as unavailable and let the health check handle reconnection
      setJobServerAvailable(false);
    };

    return () => {
      eventSource.close();
    };
  }, [jobServerAvailable, queryClient, libraryService]);

  const handleClearJob = async (job: JobRunnerJob) => {
    try {
      if (job.state === 0 || job.state === 1) {
        // Pending or InProgress
        await fetch(`http://localhost:8090/job/${job.id}/cancel`, {
          method: 'POST',
        });
      } else {
        await fetch(`http://localhost:8090/job/${job.id}/remove`, {
          method: 'POST',
        });
      }
    } catch (error) {
      console.error('Failed to clear job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to clear job',
          message: 'Could not communicate with job service',
        },
      });
    }
  };

  const handleClearToast = (toastId: string) => {
    libraryService.send({ type: 'REMOVE_TOAST', data: { id: toastId } });
  };

  // Create array from jobs Map
  const jobsArray = Array.from(jobs, ([key, value]) => ({ key, value }));

  return (
    <div className="ToastSystem">
      {/* Job Toasts */}
      {jobsArray.map(({ key, value: job }) => (
        <JobToast key={key} job={job} onClear={() => handleClearJob(job)} />
      ))}

      {/* Action Toasts */}
      {toasts.map((toast) => (
        <ActionToast
          key={toast.id}
          toast={toast}
          onClear={() => handleClearToast(toast.id)}
        />
      ))}
    </div>
  );
}
