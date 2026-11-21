import { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { debounce } from 'lodash';
import { Metadata } from 'main/metadata';
import { invoke, mediaServerBase, store } from '../../platform';
import { GlobalStateContext } from '../../state';
import { useCanWrite } from '../../hooks/useCanWrite';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import useJobsForPath from '../../hooks/useJobsForPath';
import { jobStatusLabel, pickActiveJob } from '../../job-status';
import { TEXT_CAPTURE_LANGUAGES, createTextCaptureJob } from './text-capture';
import './generate-transcript.css';
import './image-transcript.css';

const LANGUAGE_KEY = 'textCapture.language';

const loadFileMetadata = (path: string) => async (): Promise<Metadata> => {
  const metadata = await invoke('load-file-metadata', [path]);
  return (metadata || { path, tags: '' }) as Metadata;
};

// An image's transcript: the text captured from the picture (the Text capture
// task), stored in the media transcript column and read through the same
// ['file-metadata', path] query as the description, so the caption overlay
// and this panel refresh together. Video/audio transcripts are timed .vtt
// cues instead (Transcript).
export default function ImageTranscript({ path }: { path: string }) {
  const canWrite = useCanWrite();
  const client = useQueryClient();
  const { data } = useQuery<Metadata, Error>(
    ['file-metadata', path],
    loadFileMetadata(path),
    { retry: true }
  );
  const saved = data?.transcript || '';
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(saved);
  useEffect(() => setText(saved), [saved, path]);

  const save = useRef(
    debounce(async (p: string, value: string) => {
      await invoke('update-transcript', [p, value]);
      client.invalidateQueries(['file-metadata', p]);
    }, 500)
  );
  useEffect(() => {
    if (canWrite && editing && text !== saved) save.current(path, text);
  }, [text]);
  // A pending edit lands on the file it was typed for, not the next one.
  useEffect(
    () => () => {
      save.current.flush();
    },
    [path]
  );

  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = 'auto';
      ref.current.style.height = ref.current.scrollHeight + 'px';
    }
  }, [text, editing]);

  return (
    <div className="ImageTranscript">
      <CaptureText path={path} hasText={!!saved} />
      {editing ? (
        <textarea
          className="image-transcript-text image-transcript-editor"
          ref={ref}
          autoFocus
          value={text}
          onKeyDown={(e) => e.stopPropagation()}
          onKeyUp={(e) => e.stopPropagation()}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => setEditing(false)}
        />
      ) : (
        <div
          className={`image-transcript-text ${text ? '' : 'empty'}`}
          title={canWrite ? 'Double-click to edit' : undefined}
          onDoubleClick={() => canWrite && setEditing(true)}
        >
          {text ||
            (canWrite
              ? 'No text transcribed yet. Transcribe it from the image, or double-click to type it.'
              : 'No text captured for this image.')}
        </div>
      )}
    </div>
  );
}

// The Capture text action: a language picker (translate into it, or keep the
// text as written) and the button, swapped for the job's status while one runs.
function CaptureText({ path, hasText }: { path: string; hasText: boolean }) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const canWrite = useCanWrite();
  const jobServerAvailable = useJobServerAvailable(authToken);
  const [language, setLanguage] = useState<string>(
    () => (store.get(LANGUAGE_KEY, '') as string) || ''
  );
  const [busy, setBusy] = useState(false);
  const { jobs, noteJob } = useJobsForPath(
    path,
    authToken,
    canWrite && jobServerAvailable === true
  );
  const activeJob = pickActiveJob(jobs, ['capture-text']);

  if (!canWrite || jobServerAvailable !== true) return null;

  const pick = (v: string) => {
    setLanguage(v);
    store.set(LANGUAGE_KEY, v);
  };

  const capture = async () => {
    setBusy(true);
    try {
      const id = await createTextCaptureJob(path, authToken, language);
      if (id) noteJob({ id, command: 'capture-text', state: 'pending' });
    } catch (error) {
      console.error('Failed to create text capture job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    } finally {
      setBusy(false);
    }
  };

  // The SSE 'update' event (state → cancelled) clears the indicator.
  const cancel = async (jobId: string) => {
    const headers: HeadersInit = {};
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    await fetch(`${mediaServerBase}/job/${jobId}/cancel`, {
      method: 'POST',
      headers,
      signal: AbortSignal.timeout(5000),
    }).catch((err) => console.error('Failed to cancel text capture job:', err));
  };

  if (activeJob) {
    return (
      <div className="GenerateTranscript inline image-transcript-actions">
        <div className="job-running">
          <div className="loading-spinner" />
          <span className="job-status">
            {jobStatusLabel(activeJob, 'Reading text')}
          </span>
          <button
            type="button"
            className="job-cancel"
            onClick={() => cancel(activeJob.id)}
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="GenerateTranscript inline image-transcript-actions">
      <select
        className="image-transcript-language"
        value={language}
        onChange={(e) => pick(e.target.value)}
        onKeyDown={(e) => e.stopPropagation()}
        title="Translate the text into this language, whatever it is written in"
      >
        <option value="">Language: server default</option>
        <option value="Original">As written</option>
        {TEXT_CAPTURE_LANGUAGES.map((l) => (
          <option key={l} value={l}>
            Translate to {l}
          </option>
        ))}
      </select>
      <button className="generate" onClick={capture} disabled={busy}>
        {hasText ? 'Transcribe again' : 'Transcribe'}
      </button>
    </div>
  );
}
