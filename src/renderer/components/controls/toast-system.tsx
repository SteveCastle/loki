import React, { useContext, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
// Removed cancel import - using Unicode character instead
import { GlobalStateContext } from '../../state';
import { Job } from 'main/jobs';
import './toast-system.css';

interface Toast {
  id: string;
  type: 'success' | 'error' | 'info';
  title: string;
  message?: string;
  timestamp: number;
}

interface JobToastProps {
  job: Job;
  onClear: () => void;
}

const JobToast: React.FC<JobToastProps> = ({ job, onClear }) => {
  const { libraryService } = useContext(GlobalStateContext);

  return (
    <div className="toast job-toast">
      <div className="toast-content">
        <div
          className={[
            'loading-animation',
            job.status === 'started' ? 'started' : '',
            job.status === 'pending' ? 'pending' : '',
            job.status === 'error' ? 'error' : '',
            job.status === 'complete' ? 'complete' : '',
          ].join(' ')}
        ></div>
        <div className="toast-text">
          <span className="toast-title">{job.title}</span>
          {job.mediaPaths.map((path) => (
            <span
              key={path}
              className="toast-file"
              onClick={() => {
                libraryService.send('RESET_CURSOR', {
                  currentItem: { path },
                });
              }}
            >
              {path.split(/[\\/]/).pop()}
            </span>
          ))}
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
  
  const jobs = useSelector(libraryService, (state) => state.context.jobs);
  const toasts = useSelector(libraryService, (state) => state.context.toasts);
  
  // Create array from jobs Map
  const jobsArray = Array.from(jobs, ([key, value]) => ({ key, value }));

  const handleJobComplete = (args: any) => {
    const job = args as Job;
    libraryService.send({ type: 'COMPLETE_JOB', job });
    (job.invalidations || []).forEach((invalidation) => {
      console.log('invalidating', invalidation);
      queryClient.invalidateQueries(invalidation);
    });
  };

  const handleJobUpdated = (args: any) => {
    const job = args as Job;
    libraryService.send({ type: 'UPDATE_JOB', job });
  };

  useEffect(() => {
    // Here we are listening for the 'message-from-server' event
    window.electron.ipcRenderer.on('job-complete', handleJobComplete);
    window.electron.ipcRenderer.on('job-updated', handleJobUpdated);

    // Clean up by removing the listener when the component unmounts
    return () => {
      window.electron.ipcRenderer.removeListener(
        'job-complete',
        handleJobComplete
      );
      window.electron.ipcRenderer.removeListener(
        'job-updated',
        handleJobUpdated
      );
    };
  }, []);

  const handleClearJob = (job: Job) => {
    libraryService.send({ type: 'CLEAR_JOB', job });
  };

  const handleClearToast = (toastId: string) => {
    libraryService.send({ type: 'REMOVE_TOAST', data: { id: toastId } });
  };

  return (
    <div className="ToastSystem">
      {/* Job Toasts */}
      {jobsArray.map(({ key, value: job }) => (
        <JobToast
          key={key}
          job={job}
          onClear={() => handleClearJob(job)}
        />
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