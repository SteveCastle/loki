import React, { useContext, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { invoke } from '../../platform';
import './media-error.css';

// What the cleanup removed, for the confirmation toast. Both backends (the
// Electron IPC handler and POST /api/media/forget) return the same shape;
// older/partial responses just yield an empty summary.
interface ForgetResult {
  tags?: number;
  embeddings?: number;
  faces?: number;
  battles?: number;
}

function summarize(result: ForgetResult | undefined): string {
  if (!result) return '';
  const parts: string[] = [];
  const add = (n: number | undefined, one: string, many: string) => {
    if (!n) return;
    parts.push(`${n} ${n === 1 ? one : many}`);
  };
  add(result.tags, 'tag', 'tags');
  add(result.embeddings, 'embedding', 'embeddings');
  add(result.faces, 'face', 'faces');
  add(result.battles, 'battle', 'battles');
  return parts.length ? `removed ${parts.join(', ')}` : '';
}

export default function MediaError({ path }: { path: string }) {
  const { libraryService } = useContext(GlobalStateContext);
  const canWrite = useSelector(libraryService, (s) => s.context.canWrite);
  // Two-click confirm: the rows (tags, ratings, faces, embeddings) can't be
  // restored, and this button sits on media the user is only glancing at.
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);

  // Erase every database reference to this path — tags, the media row,
  // embeddings, faces, and battle history — then advance to the next item.
  // The file itself is untouched (it's missing, which is why we're here);
  // "Delete" can't do this job because trashing a nonexistent file throws
  // before any of the rows are cleaned up.
  const handleForget = async () => {
    if (!armed) {
      setArmed(true);
      window.setTimeout(() => setArmed(false), 4000);
      return;
    }
    setArmed(false);
    setBusy(true);
    try {
      const result = (await invoke('forget-media', [path])) as
        | ForgetResult
        | undefined;
      // Sent only after the cleanup resolves, so landing on the next item is
      // the signal that the database is actually clean.
      libraryService.send({
        type: 'FORGET_FILE',
        data: { path, summary: summarize(result) },
      });
    } catch (err) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to remove from library',
          message: String(err),
        },
      });
    } finally {
      // Usually a no-op: this component unmounts as the item leaves the
      // library. It matters when the item wasn't in the current library —
      // the button must not stay stuck on "Cleaning up…".
      setBusy(false);
    }
  };

  return (
    <div className="MediaError">
      <p>Media unavailable.</p>
      <button onClick={() => libraryService.send('UPDATE_FILE_PATH', { path })}>
        Find Media
      </button>
      {canWrite && (
        <button
          className={armed ? 'danger' : undefined}
          onClick={handleForget}
          disabled={busy}
          title={
            armed
              ? 'Click again to erase this item’s tags, ratings, embeddings, and faces from the database. The file itself is not touched.'
              : 'Erase every database reference to this missing file — tags, embeddings, faces, battle history — and move to the next item.'
          }
        >
          {busy
            ? 'Cleaning up…'
            : armed
            ? 'Confirm removal'
            : 'Remove from Library'}
        </button>
      )}
      {/* <button
        onClick={() =>
          libraryService.send('UPDATE_FILE_PATH', { path, updateAll: true })
        }
      >
        Find All With Same Base Path
      </button> */}
    </div>
  );
}
