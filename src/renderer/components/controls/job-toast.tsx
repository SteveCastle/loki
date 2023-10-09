import React, { useContext, useEffect } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';
import { GlobalStateContext } from '../../state';
import './job-toast.css';
import { Job } from 'main/jobs';

export function JobToast() {
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const jobs = useSelector(libraryService, (state) => state.context.jobs);
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

  return (
    <div className="JobToast">
      <ul>
        {jobsArray.map(({ key, value: job }) => (
          <li key={key}>
            <div className="job">
              <div className="content">
                <div
                  className={[
                    'loading-animation ',
                    job.status === 'started' ? 'started' : '',
                    job.status === 'pending' ? 'pending' : '',
                    job.status === 'error' ? 'error' : '',
                    job.status === 'complete' ? 'complete' : '',
                  ].join(' ')}
                ></div>
                <span className="job-label">{job.title}</span>
                {job.mediaPaths.map((path) => (
                  <span
                    key={path}
                    className="job-file"
                    onClick={() => {
                      libraryService.send('RESET_CURSOR', {
                        currentItem: { path },
                      });
                    }}
                  >
                    {path}
                  </span>
                ))}
              </div>
              <div
                className="clear-job"
                onClick={() => {
                  libraryService.send({ type: 'CLEAR_JOB', job });
                }}
              >
                <img src={cancel} />
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
