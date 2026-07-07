import React, { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import { send, mediaServerBase } from '../../platform';
import { subscribeStream } from '../../stream-bus';
import './toast-system.css';

type JobState =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'cancelled'
  | 'error'
  | 'paused';

interface JobRunnerJob {
  id: string;
  command: string;
  arguments: string[];
  input: string;
  state: JobState; // 0=Pending, 1=InProgress, 2=Completed, 3=Cancelled, 4=Error, 5=Paused
  created_at: string;
  claimed_at?: string;
  completed_at?: string;
  errored_at?: string;
  progress_done?: number;
  progress_total?: number;
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
  // Pause/resume the job (graceful: the current item finishes and all
  // completed work is kept). Shown for active and paused jobs.
  onPauseResume: () => void;
}

// parseFlag pulls a CLI flag value out of a job's raw input string, e.g.
// parseFlag('metadata --type description --apply all', 'type') === 'description'.
const parseFlag = (input: string, flag: string): string | null => {
  if (!input) return null;
  const m = input.match(new RegExp(`--${flag}[=\\s]+([\\w-]+)`));
  return m ? m[1] : null;
};

// getJobTitle returns a short, human-readable title for a background job.
const getJobTitle = (job: JobRunnerJob): string => {
  switch (job.command) {
    case 'wait':
      return 'Wait';
    case 'gallery-dl':
      return 'Gallery Download';
    case 'dce':
      return 'DCE';
    case 'yt-dlp':
      return 'Video Download';
    case 'ffmpeg':
      return 'Processing Media';
    case 'remove':
      return 'Removing Media';
    case 'cleanup':
      return 'Cleaning Up';
    case 'ingest':
      return 'Importing Media';
    case 'move':
      return 'Moving Media';
    case 'embed':
      return 'Visual Embedding';
    case 'autotag':
      return 'Auto-Tagging';
    case 'faces':
      return 'Scanning Faces';
    case 'faces-cluster':
      return job.input.includes('--reset')
        ? 'Rebuilding Face Groups'
        : 'Grouping Faces into People';
    case 'describe':
      return 'Generating Descriptions';
    case 'transcribe':
      return 'Transcribing Audio';
    case 'hash':
      return 'Hashing Files';
    case 'dimensions':
      return 'Reading Dimensions';
    case 'llm-autotag':
      return 'Auto-Tagging (LLM)';
    case 'process':
      return 'Processing Media';
    case 'metadata':
      switch (parseFlag(job.input, 'type')) {
        case 'description':
          return 'Generating Descriptions';
        case 'transcript':
          return 'Transcribing Audio';
        case 'hash':
          return 'Hashing Files';
        case 'dimensions':
          return 'Reading Dimensions';
        default:
          return 'Generating Metadata';
      }
    default:
      return job.command;
  }
};

// getJobSubtitle returns a one-line description of what the job is doing and why
// — context that reassures the user the right thing was kicked off.
const getJobSubtitle = (job: JobRunnerJob): string | null => {
  switch (job.command) {
    case 'embed':
      return 'Indexing images so you can search by visual similarity.';
    case 'autotag':
      return 'Detecting tags from each image’s content.';
    case 'faces':
      return 'Finding faces and characters so they can be grouped into people.';
    case 'faces-cluster':
      return job.input.includes('--reset')
        ? 'Regrouping the unnamed clusters from scratch — named people are kept.'
        : 'Matching new faces to your people; nothing already grouped is moved.';
    case 'describe':
      return 'Writing AI descriptions of your media.';
    case 'transcribe':
      return 'Transcribing speech into searchable text.';
    case 'hash':
      return 'Computing content hashes to find duplicates.';
    case 'dimensions':
      return 'Reading the width & height of your media.';
    case 'llm-autotag':
      return 'Selecting tags from your taxonomy with a vision model.';
    case 'process':
      return 'Applying multiple operations to each file in one pass.';
    case 'metadata':
      switch (parseFlag(job.input, 'type')) {
        case 'description':
          return 'Writing AI descriptions of your media.';
        case 'transcript':
          return 'Transcribing speech into searchable text.';
        case 'hash':
          return 'Computing content hashes to find duplicates.';
        case 'dimensions':
          return 'Reading the width & height of your media.';
        default:
          return 'Generating metadata for your media.';
      }
    case 'ingest':
      return 'Scanning and importing files into your library.';
    case 'move':
      return 'Relocating files on disk.';
    case 'cleanup':
      return 'Removing orphaned entries from the library.';
    case 'remove':
      return 'Removing media from the library.';
    case 'ffmpeg':
      return 'Transcoding media.';
    case 'yt-dlp':
      return 'Downloading video from the web.';
    case 'gallery-dl':
      return 'Downloading a gallery from the web.';
    default:
      return null;
  }
};

const JobToast: React.FC<JobToastProps> = ({ job, onClear, onPauseResume }) => {
  const { libraryService } = useContext(GlobalStateContext);
  const library = useSelector(libraryService, (state) => state.context.library);
  const status = job.state;
  const title = getJobTitle(job);
  const subtitle = getJobSubtitle(job);

  // Extract file path from input - look for quoted paths or file-like arguments
  const extractFilePath = (input: string): string | null => {
    if (!input) return null;

    // Match quoted paths first (most reliable)
    // Supports both double and single quotes
    const quotedMatch = input.match(/["']([^"']+)["']/);
    if (quotedMatch) {
      return quotedMatch[1];
    }

    const trimmedInput = input.trim();

    // If the whole string looks like a path and contains spaces,
    // it's likely a single path passed as input (common for ingest/metadata jobs)
    const hasPathSeparator =
      trimmedInput.includes('/') || trimmedInput.includes('\\');
    const hasDriveLetter = /^[a-zA-Z]:/.test(trimmedInput);
    const hasMediaExtension =
      /\.(mp4|mkv|avi|mov|m4v|webm|mp3|wav|flac|aac|ogg|m4a|opus|jpg|jpeg|jfif|webp|avif|png|gif|vtt|srt|ass)$/i.test(
        trimmedInput
      );

    if (hasPathSeparator || hasDriveLetter || hasMediaExtension) {
      // If it doesn't look like a complex command with flags, treat the whole thing as a path
      const hasFlags = /\s--?[a-zA-Z0-9]/.test(trimmedInput);
      if (!hasFlags) {
        return trimmedInput;
      }
    }

    // Fallback: look for path-like strings (original logic, but slightly improved)
    // We try to match as much as possible that looks like a path
    const pathMatch = input.match(
      /([^\s]+(?:\/|\\)[^\s]*|[^\s]*\.[a-zA-Z0-9]{2,4})/
    );
    if (pathMatch) {
      return pathMatch[1];
    }

    return null;
  };

  const filePath = extractFilePath(job.input);

  const handleOpenJobDetail = () => {
    send('open-external', [`${mediaServerBase}/job/${job.id}`]);
  };

  const progressDone = job.progress_done ?? 0;
  const progressTotal = job.progress_total ?? 0;
  const progressPct =
    progressTotal > 0
      ? Math.max(0, Math.min(100, Math.round((progressDone / progressTotal) * 100)))
      : 0;

  return (
    <div className="toast job-toast">
      <div className="toast-content toast-clickable" onClick={handleOpenJobDetail}>
        <div
          className={[
            'loading-animation',
            status === 'in_progress' ? 'in_progress' : '',
            status === 'pending' ? 'pending' : '',
            status === 'paused' ? 'paused' : '',
            status === 'error' ? 'error' : '',
            status === 'completed' ? 'completed' : '',
          ].join(' ')}
        ></div>
        <div className="toast-text">
          <span className="toast-title">
            {title}
            {status === 'paused' ? ' (paused)' : ''}
          </span>
          {subtitle && <span className="toast-message">{subtitle}</span>}
          {progressTotal > 0 && (
            <div className="toast-progress">
              <div className="toast-progress-track">
                <div
                  className="toast-progress-fill"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <span className="toast-progress-label">
                {progressDone}/{progressTotal}
              </span>
            </div>
          )}
          {filePath && (
            <div className="toast-file-path-container">
              <span
                className="toast-file-path"
                onClick={(e) => {
                  e.stopPropagation();
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
      {(status === 'pending' ||
        status === 'in_progress' ||
        status === 'paused') && (
        <button
          type="button"
          className="toast-pause"
          onClick={(e) => {
            e.stopPropagation();
            onPauseResume();
          }}
          title={
            status === 'paused'
              ? 'Resume from where it stopped'
              : 'Pause after the current item — finished work is kept'
          }
        >
          {status === 'paused' ? '▶' : '⏸'}
        </button>
      )}
      <div
        className="toast-close"
        onClick={onClear}
        title="Dismiss this notification — the job keeps running"
      >
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
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const [jobs, setJobs] = useState<Map<string, JobRunnerJob>>(new Map());

  // Jobs whose toast the user dismissed with the × button. Clearing a toast is
  // a view-only action — it hides the notification and never touches the job —
  // so we must also suppress the toast when later SSE updates for that job
  // arrive (otherwise a progress/state update would resurrect it).
  const dismissedJobsRef = useRef<Set<string>>(new Set());

  // When a media-created event fires, we store the target path and the
  // current libraryLoadId. Once libraryLoadId changes (the refresh completed),
  // we send RESET_CURSOR which searches the filtered/sorted view.
  const pendingNavigateRef = useRef<{
    path: string;
    sinceLoadId: string;
  } | null>(null);


  const libraryLoadId = useSelector(
    libraryService,
    (state) => state.context.libraryLoadId
  );

  // Once the library has been refreshed (libraryLoadId changed), navigate
  // to the pending target. RESET_CURSOR handles filter/sort internally.
  useEffect(() => {
    const pending = pendingNavigateRef.current;
    if (!pending) return;
    // Wait until the library has actually been reloaded.
    if (libraryLoadId === pending.sinceLoadId) return;
    pendingNavigateRef.current = null;
    libraryService.send('RESET_CURSOR', {
      currentItem: { path: pending.path },
    });
  }, [libraryLoadId, libraryService]);

  const toasts = useSelector(
    libraryService,
    (state) => state.context.toasts || []
  );

  // All /stream consumption goes through the shared bus (one EventSource for
  // the whole renderer — see stream-bus.ts for why): reconnects, zombie
  // detection, and availability all live there, so this component just
  // subscribes to events.
  useEffect(() => {
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

    const onCreate = (event: MessageEvent) => {
      const data = parseJobEvent(event.data);
      if (!data || !data.job) return;
      const job = data.job as JobRunnerJob;
      console.log('Job created:', job);
      if (dismissedJobsRef.current.has(job.id)) return;
      setJobs((prev) => new Map(prev).set(job.id, job));
      // No toast for job creation - the job toast itself shows the status
    };

    const onUpdate = (event: MessageEvent) => {
      const data = parseJobEvent(event.data);
      if (!data || !data.job) return;
      const job = data.job as JobRunnerJob;
      console.log('Job updated:', job);
      // A dismissed toast stays hidden; the query invalidations below still run
      // so data refreshes regardless of whether the toast is showing.
      if (!dismissedJobsRef.current.has(job.id)) {
        setJobs((prev) => new Map(prev).set(job.id, job));
      }

      // Handle job completion
      if (job.state === 'completed') {
        // Completed
        // Invalidate queries for metadata-producing jobs (the split-out ops
        // and the combined "process" task included)
        const metadataCommands = [
          'metadata',
          'ingest',
          'describe',
          'transcribe',
          'hash',
          'dimensions',
          'process',
        ];
        if (metadataCommands.includes(job.command)) {
          queryClient.invalidateQueries(['transcript']);
          queryClient.invalidateQueries(['media']);
          queryClient.invalidateQueries(['file-metadata']);
          queryClient.invalidateQueries(['tags-by-path']);
        }

        if (
          job.command === 'autotag' ||
          job.command === 'llm-autotag' ||
          job.command === 'process'
        ) {
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
    };

    const onDelete = (event: MessageEvent) => {
      const data = parseJobEvent(event.data);
      if (!data || !data.job) return;
      const jobId = data.job.id;
      dismissedJobsRef.current.delete(jobId);
      setJobs((prev) => {
        const newJobs = new Map(prev);
        newJobs.delete(jobId);
        return newJobs;
      });
    };

    // Item-level progress events ({id, done, total}) — merge into the job's
    // toast so long-running tasks show a live bar.
    const onProgress = (event: MessageEvent) => {
      try {
        const p = JSON.parse(event.data) as {
          id?: string;
          done?: number;
          total?: number;
        };
        const jobId = p.id;
        if (!jobId || typeof p.total !== 'number') return;
        setJobs((prev) => {
          const existing = prev.get(jobId);
          if (!existing) return prev; // job not shown (started before app opened)
          const newJobs = new Map(prev);
          newJobs.set(jobId, {
            ...existing,
            progress_done: p.done ?? 0,
            progress_total: p.total ?? 0,
          });
          return newJobs;
        });
      } catch {
        // Best-effort parse; ignore malformed events
      }
    };

    // When media files are overwritten (e.g. save task in "replace" mode),
    // invalidate cached previews so the UI shows the updated file.
    const onMediaUpdated = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data);
        const inner = typeof payload.msg === 'string' ? JSON.parse(payload.msg) : payload;
        const paths: string[] = inner.paths || [];
        for (const p of paths) {
          queryClient.invalidateQueries(['media', 'preview', p]);
        }
        // Also invalidate the general media list to refresh thumbnails
        if (paths.length > 0) {
          queryClient.invalidateQueries(['media']);
        }
        // Notify detail view (and any other listener) so it can bust the
        // browser HTTP cache for direct media URLs.
        if (paths.length > 0) {
          window.dispatchEvent(
            new CustomEvent('loki-media-updated', { detail: { paths } })
          );
        }
      } catch {
        // Best-effort parse; ignore malformed events
      }
    };

    // When new files are created (e.g. save task in "alongside" or "folder" mode),
    // refresh the library so they appear immediately and navigate to the first new file.
    const onMediaCreated = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data);
        const inner = typeof payload.msg === 'string' ? JSON.parse(payload.msg) : payload;
        const paths: string[] = inner.paths || [];
        if (paths.length > 0) {
          // Capture the current libraryLoadId so the useEffect knows to
          // wait for it to change before navigating.
          const snapshot = libraryService.getSnapshot();
          pendingNavigateRef.current = {
            path: paths[0],
            sinceLoadId: snapshot.context.libraryLoadId,
          };

          // Refresh the library by re-querying the current view.
          if (snapshot.matches({ library: 'loadedFromDB' })) {
            libraryService.send({ type: 'SORTED_WEIGHTS' });
          } else if (snapshot.matches({ library: 'loadedFromFS' })) {
            libraryService.send('REFRESH_LIBRARY');
          }
        }
      } catch {
        // Best-effort parse; ignore malformed events
      }
    };

    // One bus subscription covers all event types; reconnects and zombie
    // detection live in the bus itself.
    return subscribeStream((type, event) => {
      switch (type) {
        case 'create':
          onCreate(event);
          break;
        case 'update':
          onUpdate(event);
          break;
        case 'delete':
          onDelete(event);
          break;
        case 'progress':
          onProgress(event);
          break;
        case 'media-updated':
          onMediaUpdated(event);
          break;
        case 'media-created':
          onMediaCreated(event);
          break;
        default:
          break;
      }
    });
  }, [queryClient]);

  // Clearing a toast is view-only: hide the notification, leave the job alone
  // (pause/resume has its own button; the job detail page can cancel). We also
  // record the id so a later SSE update for the same job doesn't resurrect it.
  const handleClearJob = (job: JobRunnerJob) => {
    dismissedJobsRef.current.add(job.id);
    setJobs((prev) => {
      const newJobs = new Map(prev);
      newJobs.delete(job.id);
      return newJobs;
    });
  };

  // Graceful pause / resume: the server parks the job after its current item
  // (all completed work is kept) and resumes from where it stopped. The
  // toast's state flips via the SSE update event.
  const handlePauseResumeJob = async (job: JobRunnerJob) => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const action = job.state === 'paused' ? 'resume' : 'pause';
      const headers: HeadersInit = {};
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }
      const res = await fetch(`${mediaServerBase}/job/${job.id}/${action}`, {
        method: 'POST',
        headers,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (error) {
      console.error('Failed to pause/resume job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to pause/resume job',
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
        <JobToast
          key={key}
          job={job}
          onClear={() => handleClearJob(job)}
          onPauseResume={() => handlePauseResumeJob(job)}
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
