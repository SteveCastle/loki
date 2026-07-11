import { useContext, useCallback } from 'react';
import { useDrop } from 'react-dnd';
import { NativeTypes } from 'react-dnd-html5-backend';
import { GlobalStateContext } from '../state';
import { useSelector } from '@xstate/react';
import { invoke, isElectron, store } from '../platform';
import { getFileType, FileTypes } from '../../file-types';

// Track shift key globally for move-vs-copy on drop
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') (window as any).__shiftHeld = true;
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') (window as any).__shiftHeld = false;
  });
  window.addEventListener('blur', () => {
    (window as any).__shiftHeld = false;
  });

  // In Electron, dropping a file on the window navigates to it by default.
  // Prevent this so react-dnd can handle the drop instead.
  document.addEventListener('dragover', (e) => {
    e.preventDefault();
  });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
  });
}

function isMediaFile(fileName: string): boolean {
  const ft = getFileType(fileName);
  return ft === FileTypes.Image || ft === FileTypes.Video || ft === FileTypes.Audio;
}

function isArchiveFile(fileName: string): boolean {
  return getFileType(fileName) === FileTypes.Archive;
}

/** Resolve the browsed directory from initialFile (which may be a file path, archive path, or dir path). */
function resolveDirectory(initialFile: string): string {
  const ft = getFileType(initialFile);
  // Archive paths represent their contents — pass through unchanged.
  if (ft === FileTypes.Archive) {
    return initialFile;
  }
  if (ft !== FileTypes.Other) {
    // It's a media file — extract the directory
    const lastSep = Math.max(initialFile.lastIndexOf('/'), initialFile.lastIndexOf('\\'));
    let dir = lastSep > 0 ? initialFile.substring(0, lastSep) : initialFile;
    if (/^[A-Za-z]:$/.test(dir)) {
      dir += '\\';
    }
    return dir;
  }
  return initialFile;
}

export default function useFileDrop() {
  const { libraryService } = useContext(GlobalStateContext);

  const currentStateType = useSelector(
    libraryService,
    (state) => state.context.currentStateType
  );
  const initialFile = useSelector(
    libraryService,
    (state) => state.context.initialFile
  );
  const dbQueryTags = useSelector(
    libraryService,
    (state) => state.context.dbQuery.tags
  );
  const activeCategory = useSelector(
    libraryService,
    (state) => state.context.activeCategory
  );

  /** Re-trigger the current browse to pick up newly added files. */
  function refreshCurrentView() {
    if (isElectron) {
      // In Electron, REFRESH_LIBRARY does an incremental diff (adds/removes)
      // without resetting cursor position. Works for FS, no-op for DB/search
      // but DB/search use SORTED_WEIGHTS below.
      if (currentStateType === 'fs') {
        libraryService.send('REFRESH_LIBRARY');
      } else if (currentStateType === 'db') {
        libraryService.send({ type: 'SORTED_WEIGHTS' });
      }
    } else {
      // In web mode, REFRESH_LIBRARY is stubbed. Use mode-specific events.
      if (currentStateType === 'fs') {
        libraryService.send('SET_FILE', { path: initialFile });
      } else if (currentStateType === 'db') {
        libraryService.send({ type: 'SORTED_WEIGHTS' });
      }
    }
  }

  const handleDrop = useCallback(
    async (item: { files: File[] }) => {
      // Guard: only allow drops when library is loaded
      const snapshot = libraryService.getSnapshot();
      const isLoaded =
        snapshot.matches({ library: 'loadedFromFS' }) ||
        snapshot.matches({ library: 'loadedFromDB' });
      if (!isLoaded) return;
      // View-only public visitors can't upload or import.
      if (!snapshot.context.canWrite) return;

      const nativeFiles = item.files;
      if (!nativeFiles || nativeFiles.length === 0) return;

      // Archives open as a directory (treat like SELECT_ARCHIVE with a known path).
      const archiveFiles = nativeFiles.filter((f) => isArchiveFile(f.name));
      if (archiveFiles.length > 0) {
        const getPath = (window as any).electron?.getPathForFile;
        const first = archiveFiles[0];
        const archivePath = (getPath ? getPath(first) : (first as any).path) as
          | string
          | undefined;
        if (archivePath) {
          libraryService.send({ type: 'SET_FILE', path: archivePath });
        }
        return;
      }

      // Filter to media files only
      const mediaFiles = nativeFiles.filter((f) => isMediaFile(f.name));
      if (mediaFiles.length === 0) {
        libraryService.send('ADD_TOAST', {
          data: {
            type: 'warning',
            title: 'No media files',
            message: 'None of the dropped files are supported media types',
            durationMs: 3000,
          },
        });
        return;
      }

      const move = (window as any).__shiftHeld === true;

      if (isElectron) {
        await handleElectronDrop(mediaFiles, move);
      } else {
        await handleWebDrop(mediaFiles);
      }
    },
    [currentStateType, initialFile, dbQueryTags, activeCategory, libraryService]
  );

  async function handleElectronDrop(files: File[], move: boolean) {
    // Use Electron's webUtils.getPathForFile to get absolute paths from dropped files.
    // File.path was removed in recent Electron versions.
    const getPath = (window as any).electron?.getPathForFile;
    const filePaths = files.map((f) => {
      if (getPath) return getPath(f) as string;
      return (f as any).path as string | undefined;
    }).filter(Boolean) as string[];
    if (filePaths.length === 0) return;

    let destination: string;
    let tags: { label: string; category: string }[] = [];

    if (currentStateType === 'fs') {
      destination = resolveDirectory(initialFile);
    } else {
      // DB or search mode — use configured default import path
      let defaultImportPath = store.get('defaultImportPath', '') as string;
      if (!defaultImportPath) {
        const selected = await invoke('select-directory', []);
        if (!selected) return;
        store.set('defaultImportPath', selected);
        defaultImportPath = selected;
      }
      destination = defaultImportPath + '/imports';

      // In DB mode, apply the active tags.
      // NOTE: We pass empty category so the IPC handler resolves each tag's
      // actual category from the database. Using activeCategory here would
      // assign the wrong category in multi-tag views across categories.
      if (currentStateType === 'db' && dbQueryTags.length > 0) {
        tags = dbQueryTags.map((tagLabel: string) => ({
          label: tagLabel,
          category: '', // Category resolved by IPC handler
        }));
      }
    }

    try {
      const result = await invoke('import-files', [
        { files: filePaths, destination, move, tags },
      ]);

      const count = result?.imported?.length || 0;
      if (count === 0) {
        libraryService.send('ADD_TOAST', {
          data: {
            type: 'error',
            title: 'Import failed',
            message: `Failed to import ${result?.failed?.length || 0} file(s)`,
            durationMs: 5000,
          },
        });
        return;
      }
      const action = move ? 'Moved' : 'Added';
      let message = `${action} ${count} file${count !== 1 ? 's' : ''}`;
      if (currentStateType === 'fs') {
        message += ` to ${destination}`;
      } else if (tags.length > 0) {
        message += `, tagged: ${tags.map((t) => t.label).join(', ')}`;
      }

      libraryService.send('ADD_TOAST', {
        data: { type: 'success', title: 'Import complete', message, durationMs: 4000 },
      });

      refreshCurrentView();
    } catch (err) {
      console.error('Import failed:', err);
      libraryService.send('ADD_TOAST', {
        data: { type: 'error', title: 'Import failed', message: String(err), durationMs: 5000 },
      });
    }
  }

  async function handleWebDrop(files: File[]) {
    if (files.length === 0) return;

    // Upload files through a bounded queue instead of one giant request:
    // at most UPLOAD_CONCURRENCY in flight, the rest queued, with a live
    // progress toast. Keeps memory/bandwidth sane for large drops and lets
    // one bad file fail in isolation instead of sinking the whole batch.
    const UPLOAD_CONCURRENCY = 3;
    const total = files.length;
    const toastId = `upload-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const destination =
      currentStateType === 'fs' && initialFile
        ? resolveDirectory(initialFile)
        : '';

    const uploadedPaths: string[] = [];
    let done = 0;
    let failed = 0;
    let unauthorized = false;

    const progressMessage = () =>
      `${done} / ${total}${failed > 0 ? ` — ${failed} failed` : ''}`;

    libraryService.send('ADD_TOAST', {
      data: {
        id: toastId,
        type: 'info',
        title: 'Uploading…',
        message: progressMessage(),
        // Effectively sticky while the queue drains; set to a short duration
        // on completion below.
        durationMs: 10 * 60 * 1000,
      },
    });

    const uploadOne = async (file: File) => {
      const formData = new FormData();
      formData.append('files', file);
      if (destination) formData.append('destination', destination);

      const response = await fetch('/api/upload', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });
      if (response.status === 401) {
        unauthorized = true;
        throw new Error('unauthorized');
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const result = await response.json();
      for (const p of (result.files as string[]) || []) uploadedPaths.push(p);
    };

    const queue = [...files];
    const worker = async () => {
      while (queue.length > 0 && !unauthorized) {
        const file = queue.shift()!;
        try {
          await uploadOne(file);
        } catch (err) {
          failed += 1;
          console.error('Upload failed:', file.name, err);
        } finally {
          done += 1;
          libraryService.send('UPDATE_TOAST', {
            data: { id: toastId, message: progressMessage() },
          });
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(UPLOAD_CONCURRENCY, total) }, worker)
    );

    if (unauthorized) {
      window.location.href = '/login';
      return;
    }

    // Apply tags if in DB mode
    if (
      currentStateType === 'db' &&
      dbQueryTags.length > 0 &&
      uploadedPaths.length > 0
    ) {
      for (const tagLabel of dbQueryTags) {
        await invoke('create-assignment', [
          uploadedPaths,
          tagLabel,
          activeCategory || '',
          null,
          false,
        ]);
      }
    }

    const count = uploadedPaths.length;
    if (count === 0) {
      libraryService.send('UPDATE_TOAST', {
        data: {
          id: toastId,
          type: 'error',
          title: 'Upload failed',
          message: `All ${total} file${total !== 1 ? 's' : ''} failed to upload`,
          durationMs: 6000,
        },
      });
      return;
    }

    let message = `Added ${count} file${count !== 1 ? 's' : ''}`;
    if (currentStateType === 'fs') {
      message += ` to ${destination}`;
    } else if (dbQueryTags.length > 0) {
      message += `, tagged: ${dbQueryTags.join(', ')}`;
    }
    if (failed > 0) message += ` (${failed} failed)`;

    libraryService.send('UPDATE_TOAST', {
      data: {
        id: toastId,
        type: failed > 0 ? 'info' : 'success',
        title: 'Import complete',
        message,
        durationMs: 4000,
      },
    });

    refreshCurrentView();
  }

  const [{ isOver, canDrop }, dropRef] = useDrop(
    () => ({
      accept: [NativeTypes.FILE],
      drop: (item: any) => {
        handleDrop(item);
      },
      canDrop: () => true,
      collect: (monitor) => ({
        isOver: monitor.isOver({ shallow: true }),
        canDrop: monitor.canDrop(),
      }),
    }),
    [handleDrop]
  );

  return { dropRef, isOver, canDrop };
}
