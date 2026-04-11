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
}

function isMediaFile(fileName: string): boolean {
  const ft = getFileType(fileName);
  return ft === FileTypes.Image || ft === FileTypes.Video || ft === FileTypes.Audio;
}

/** Resolve the browsed directory from initialFile (which may be a file path or dir path). */
function resolveDirectory(initialFile: string): string {
  // Check if it has a media file extension — if so, it's a file and we want its directory
  const ft = getFileType(initialFile);
  if (ft !== FileTypes.Other) {
    // It's a media file — extract the directory
    const lastSep = Math.max(initialFile.lastIndexOf('/'), initialFile.lastIndexOf('\\'));
    return lastSep > 0 ? initialFile.substring(0, lastSep) : initialFile;
  }
  // No media extension — treat as directory
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

  const handleDrop = useCallback(
    async (item: { files: File[] }) => {
      // Guard: only allow drops when library is loaded
      const snapshot = libraryService.getSnapshot();
      const isLoaded =
        snapshot.matches({ library: 'loadedFromFS' }) ||
        snapshot.matches({ library: 'loadedFromDB' }) ||
        snapshot.matches({ library: 'loadedFromSearch' });
      if (!isLoaded) return;

      const nativeFiles = item.files;
      if (!nativeFiles || nativeFiles.length === 0) return;

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
    // Electron File objects have a .path property with the absolute path
    const filePaths = files.map((f) => (f as any).path).filter(Boolean) as string[];
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

      libraryService.send('REFRESH_LIBRARY');
    } catch (err) {
      console.error('Import failed:', err);
      libraryService.send('ADD_TOAST', {
        data: { type: 'error', title: 'Import failed', message: String(err), durationMs: 5000 },
      });
    }
  }

  async function handleWebDrop(files: File[]) {
    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file);
      }

      // In FS mode, tell the server to place files in the browsed directory
      if (currentStateType === 'fs' && initialFile) {
        const destination = resolveDirectory(initialFile);
        formData.append('destination', destination);
      }

      const response = await fetch('/api/upload', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (response.status === 401) {
        window.location.href = '/login';
        return;
      }
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();
      const uploadedPaths: string[] = result.files || [];

      // Apply tags if in DB mode
      if (currentStateType === 'db' && dbQueryTags.length > 0 && uploadedPaths.length > 0) {
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
      let message = `Added ${count} file${count !== 1 ? 's' : ''}`;
      if (currentStateType === 'fs') {
        message += ` to ${resolveDirectory(initialFile)}`;
      } else if (dbQueryTags.length > 0) {
        message += `, tagged: ${dbQueryTags.join(', ')}`;
      }

      libraryService.send('ADD_TOAST', {
        data: { type: 'success', title: 'Import complete', message, durationMs: 4000 },
      });

      libraryService.send('REFRESH_LIBRARY');
    } catch (err) {
      console.error('Upload failed:', err);
      libraryService.send('ADD_TOAST', {
        data: { type: 'error', title: 'Upload failed', message: String(err), durationMs: 5000 },
      });
    }
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
