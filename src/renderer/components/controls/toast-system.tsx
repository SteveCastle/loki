import React, { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import './toast-system.css';

type JobState = 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'error';

interface JobRunnerJob {
  id: string;
  command: string;
  arguments: string[];
  input: string;
  state: JobState; // 0=Pending, 1=InProgress, 2=Completed, 3=Cancelled, 4=Error
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
  durationMs?: number;
}

interface JobToastProps {
  job: JobRunnerJob;
  onClear: () => void;
}

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
  const status = job.state;
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
            status === 'in_progress' ? 'in_progress' : '',
            status === 'pending' ? 'pending' : '',
            status === 'error' ? 'error' : '',
            status === 'completed' ? 'completed' : '',
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
  const [isClosing, setIsClosing] = useState(false);
  const clearedRef = useRef(false);

  const handleClear = () => {
    if (clearedRef.current) return;
    clearedRef.current = true;
    setIsClosing(true);
    setTimeout(() => {
      onClear();
    }, 250);
  };

  useEffect(() => {
    // Auto-dismiss action toasts after specified duration or default 4 seconds
    const timer = setTimeout(
      () => {
        handleClear();
      },
      typeof toast.durationMs === 'number' ? toast.durationMs : 4000
    );

    return () => clearTimeout(timer);
  }, [toast.durationMs]);

  return (
    <div
      className={`toast action-toast ${toast.type} ${
        isClosing ? 'closing' : ''
      }`}
    >
      <div className="toast-content">
        <div className={`toast-indicator ${toast.type}`}></div>
        <div className="toast-text">
          <span className="toast-title">{toast.title}</span>
          {toast.message && (
            <span className="toast-message">{toast.message}</span>
          )}
        </div>
      </div>
      <div className="toast-close" onClick={handleClear}>
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
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);
  const [sseGeneration, setSseGeneration] = useState<number>(0);

  // Track last activity time to aid in debugging/health visibility
  const lastActivityAtRef = useRef<number>(Date.now());
  const eventSourceRef = useRef<EventSource | null>(null);

  const toasts = useSelector(
    libraryService,
    (state) => state.context.toasts || []
  );

  // Check if job server is available before attempting SSE connection
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

    if (isOnline) {
      checkJobServer();
    } else {
      setJobServerAvailable(false);
    }

    // Recheck every 30 seconds if server is not available
    const interval = setInterval(() => {
      if (!jobServerAvailable && isOnline) {
        checkJobServer();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [jobServerAvailable, isOnline]);

  // Track online/offline to prevent futile reconnect loops when offline
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => {
      setIsOnline(false);
      setJobServerAvailable(false);
      // Close any existing SSE connection if we go offline
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    if (!jobServerAvailable || !isOnline) {
      return; // Don't attempt SSE connection if server is not available
    }

    const eventSource = new EventSource('http://localhost:8090/stream');
    eventSourceRef.current = eventSource;

    type JobEventPayload = { job: JobRunnerJob };
    const parseJobEvent = (data: string): JobEventPayload | null => {
      try {
        const parsed: unknown = JSON.parse(data);
        if (
          parsed &&
          typeof parsed === 'object' &&
          'job' in (parsed as Record<string, unknown>)
        ) {
          return parsed as JobEventPayload;
        }
        return null;
      } catch (e) {
        console.error('Malformed SSE data:', e, data);
        return null;
      }
    };

    eventSource.onopen = () => {
      // Connection established or re-established
      setJobServerAvailable(true);
      lastActivityAtRef.current = Date.now();
    };

    eventSource.addEventListener('create', (event) => {
      lastActivityAtRef.current = Date.now();
      const data = parseJobEvent((event as MessageEvent).data);
      if (!data || !data.job) return;
      const job = data.job as JobRunnerJob;
      console.log('Job created:', job);
      setJobs((prev) => new Map(prev).set(job.id, job));
      // No toast for job creation - the job toast itself shows the status
    });

    eventSource.addEventListener('update', (event) => {
      lastActivityAtRef.current = Date.now();
      const data = parseJobEvent((event as MessageEvent).data);
      if (!data || !data.job) return;
      const job = data.job as JobRunnerJob;
      console.log('Job updated:', job);
      setJobs((prev) => new Map(prev).set(job.id, job));

      // Handle job completion
      if (job.state === 'completed') {
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
      } else if (job.state === 'error') {
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
      lastActivityAtRef.current = Date.now();
      const data = parseJobEvent((event as MessageEvent).data);
      if (!data || !data.job) return;
      const jobId = data.job.id;
      setJobs((prev) => {
        const newJobs = new Map(prev);
        newJobs.delete(jobId);
        return newJobs;
      });
    });

    // Some servers emit default 'message' events (e.g., ping). Track activity.
    eventSource.onmessage = () => {
      lastActivityAtRef.current = Date.now();
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      // Do not immediately mark unavailable; EventSource will auto-reconnect.
      // If we are offline, ensure we reflect unavailable state.
      if (!navigator.onLine) {
        setJobServerAvailable(false);
      }
    };

    return () => {
      eventSource.close();
      if (eventSourceRef.current === eventSource) {
        eventSourceRef.current = null;
      }
    };
  }, [jobServerAvailable, isOnline, sseGeneration, queryClient]);

  // Watchdog: if connection is stale or closed and no activity for a while, force a fresh subscribe
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const secondsSinceActivity = (now - lastActivityAtRef.current) / 1000;
      const current = eventSourceRef.current;
      if (!isOnline) return;

      // If we have an EventSource but it's closed or has been idle too long, force regeneration
      if (current && (current.readyState === 2 || secondsSinceActivity > 90)) {
        try {
          current.close();
        } catch (err) {
          // noop; closing a dead EventSource can throw in some environments
        }
        eventSourceRef.current = null;
        // Trigger effect to recreate the EventSource immediately
        setSseGeneration((g) => g + 1);
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [isOnline]);

  const handleClearJob = async (job: JobRunnerJob) => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const url =
        job.state === 'pending' || job.state === 'in_progress'
          ? `http://localhost:8090/job/${job.id}/cancel`
          : `http://localhost:8090/job/${job.id}/remove`;
      await fetch(url, { method: 'POST', signal: controller.signal });
      clearTimeout(timeoutId);
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
