# Drag & Drop File Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to drag native files onto the app to import them, with placement determined by the current browsing context (directory, tag, or search).

**Architecture:** A `useFileDrop` hook accepts `NativeTypes.FILE` drops and reads the XState machine context to determine the destination. In Electron mode, a new `import-files` IPC handler copies/moves files and inserts them into SQLite. In web mode, the hook POSTs files to the existing `/api/upload` endpoint. Both paths support tag application when browsing by tag.

**Tech Stack:** React, react-dnd (NativeTypes.FILE), XState, Electron IPC, Node.js fs module

---

### Task 1: Electron IPC handler — `import-files`

**Files:**
- Modify: `src/main/media.ts` (add `importFiles` function after the existing `deleteMedia` function, ~line 439)
- Modify: `src/main/main.ts:293` (register the new IPC handler)
- Modify: `src/main/preload.ts:15-61` (add channel to Channels type)

- [ ] **Step 1: Add `import-files` channel to preload.ts**

In `src/main/preload.ts`, add `'import-files'` to the `Channels` type union:

```typescript
// In the Channels type union, after 'delete-file':
  | 'delete-file'
  | 'import-files'
  | 'minimize'
```

- [ ] **Step 2: Implement `importFiles` in media.ts**

Add this function at the end of `src/main/media.ts`, before the final `export { listThumbnails, regenerateThumbnail };` line:

```typescript
type ImportFilesInput = [
  {
    files: string[];
    destination: string;
    move: boolean;
    tags: { label: string; category: string }[];
  },
];

const importFiles =
  (db: Database) => async (_: IpcMainInvokeEvent, args: ImportFilesInput) => {
    const { files, destination, move, tags } = args[0];
    const imported: string[] = [];
    const failed: string[] = [];

    // Ensure destination directory exists
    await fs.promises.mkdir(destination, { recursive: true });

    for (const sourcePath of files) {
      try {
        const baseName = path.basename(sourcePath);
        const ext = path.extname(baseName);
        const nameWithoutExt = path.basename(baseName, ext);

        // Find a non-colliding filename
        let destPath = path.join(destination, baseName);
        let counter = 1;
        while (
          await fs.promises
            .access(destPath)
            .then(() => true)
            .catch(() => false)
        ) {
          destPath = path.join(
            destination,
            `${nameWithoutExt}_${counter}${ext}`
          );
          counter++;
        }

        // Copy or move the file
        await fs.promises.copyFile(sourcePath, destPath);
        if (move) {
          await fs.promises.unlink(sourcePath);
        }

        imported.push(destPath);
      } catch (err) {
        console.error(`Failed to import ${sourcePath}:`, err);
        failed.push(sourcePath);
      }
    }

    // Insert all imported files into the database
    if (imported.length > 0) {
      await insertBulkMedia(db, imported);
    }

    // Apply tags if provided
    if (tags.length > 0 && imported.length > 0) {
      for (const tag of tags) {
        const insertStmt = await db.prepare(
          `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp, created_at)
           VALUES (?, ?, ?, ?, 0, ?)
           ON CONFLICT(media_path, tag_label, category_label, time_stamp) DO NOTHING`
        );
        for (const mediaPath of imported) {
          const countRow = await db.get(
            `SELECT COUNT(*) AS count FROM media_tag_by_category WHERE tag_label = ?`,
            [tag.label]
          );
          const weight = (countRow?.count || 0) + 1;
          await insertStmt.run(
            mediaPath,
            tag.label,
            tag.category,
            weight,
            Date.now()
          );
        }
      }
    }

    return { imported, failed };
  };
```

- [ ] **Step 3: Add `importFiles` to the exports in media.ts**

Update the main export block in `src/main/media.ts` (~line 634):

```typescript
export {
  loadMediaByTags,
  loadMediaByDescriptionSearch,
  fetchMediaPreview,
  copyFileIntoClipboard,
  deleteMedia,
  updateElo,
  updateDescription,
  loadDuplicatesByPath,
  mergeDuplicatesByPath,
  importFiles,
};
```

- [ ] **Step 4: Register the IPC handler in main.ts**

In `src/main/main.ts`, after the `delete-file` handler (~line 293), add:

```typescript
  ipcMain.handle('import-files', mediaModule.importFiles(db));
```

- [ ] **Step 5: Test manually**

Build the Electron app and test from the dev console:

```
Run: npm run start (or however the dev server starts)
```

In the Electron DevTools console, verify the channel is registered:

```javascript
window.electron.ipcRenderer.invoke('import-files', [{ files: [], destination: '/tmp/test', move: false, tags: [] }])
```

Expected: resolves with `{ imported: [], failed: [] }`

- [ ] **Step 6: Commit**

```bash
git add src/main/media.ts src/main/main.ts src/main/preload.ts
git commit -m "feat: add import-files IPC handler for drag-and-drop file import"
```

---

### Task 2: `useFileDrop` hook — Electron mode

**Files:**
- Create: `src/renderer/hooks/useFileDrop.ts`

- [ ] **Step 1: Create the useFileDrop hook**

Create `src/renderer/hooks/useFileDrop.ts`:

```typescript
import { useContext, useCallback } from 'react';
import { useDrop } from 'react-dnd';
import { NativeTypes } from 'react-dnd-html5-backend';
import { GlobalStateContext } from '../state';
import { useSelector } from '@xstate/react';
import { invoke, isElectron, store } from '../platform';
import { getFileType } from '../../file-types';
import { FileTypes } from '../../file-types';

interface FileDropResult {
  dropRef: ReturnType<typeof useDrop>[1];
  isOver: boolean;
  canDrop: boolean;
}

function isMediaFile(fileName: string): boolean {
  const ft = getFileType(fileName);
  return (
    ft === FileTypes.Image || ft === FileTypes.Video || ft === FileTypes.Audio
  );
}

export default function useFileDrop(): FileDropResult {
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
  const taxonomy = useSelector(
    libraryService,
    (state) => state.context.taxonomy
  );

  const handleDrop = useCallback(
    async (item: { files: File[]; dataTransfer: DataTransfer }) => {
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

      // Detect shift key for move-vs-copy
      const move = item.dataTransfer?.dropEffect === 'move' ||
        (window as any).__shiftHeld === true;

      if (isElectron) {
        await handleElectronDrop(mediaFiles, move);
      } else {
        await handleWebDrop(mediaFiles);
      }
    },
    [currentStateType, initialFile, dbQueryTags, activeCategory, taxonomy, libraryService]
  );

  const handleElectronDrop = async (files: File[], move: boolean) => {
    // Get absolute paths from the native files
    const filePaths = files.map((f) => (f as any).path).filter(Boolean);
    if (filePaths.length === 0) return;

    // Determine destination based on browse mode
    let destination: string;
    let tags: { label: string; category: string }[] = [];

    if (currentStateType === 'fs') {
      // Use the currently browsed directory
      // initialFile is a file path — get its directory
      const pathParts = initialFile.replace(/\\/g, '/').split('/');
      pathParts.pop(); // remove filename
      destination = pathParts.join('/') || initialFile;

      // If initialFile looks like a directory (no extension or ends with /), use it directly
      if (
        initialFile.endsWith('/') ||
        initialFile.endsWith('\\') ||
        !initialFile.includes('.')
      ) {
        destination = initialFile;
      }
    } else {
      // DB or search mode — use configured default import path
      const defaultImportPath = store.get('defaultImportPath', '') as string;
      if (!defaultImportPath) {
        // Prompt user to set a default import path
        const selected = await invoke('select-directory', []);
        if (!selected) return;
        store.set('defaultImportPath', selected);
        destination = selected + '/imports';
      } else {
        destination = defaultImportPath + '/imports';
      }

      // In DB mode, apply the active tags
      if (currentStateType === 'db' && dbQueryTags.length > 0) {
        // Look up category for each tag from taxonomy
        tags = dbQueryTags.map((tagLabel: string) => {
          let category = activeCategory || '';
          if (taxonomy) {
            for (const cat of taxonomy) {
              const found = cat.tags?.find(
                (t: any) => t.label === tagLabel
              );
              if (found) {
                category = cat.label;
                break;
              }
            }
          }
          return { label: tagLabel, category };
        });
      }
    }

    try {
      const result = await invoke('import-files', [
        { files: filePaths, destination, move, tags },
      ]);

      const count = result?.imported?.length || 0;
      const action = move ? 'Moved' : 'Added';
      let message = `${action} ${count} file${count !== 1 ? 's' : ''}`;
      if (currentStateType === 'fs') {
        message += ` to ${destination}`;
      } else if (tags.length > 0) {
        message += `, tagged: ${tags.map((t) => t.label).join(', ')}`;
      }

      libraryService.send('ADD_TOAST', {
        data: {
          type: 'success',
          title: 'Import complete',
          message,
          durationMs: 4000,
        },
      });

      // Refresh the library
      libraryService.send('REFRESH_LIBRARY');
    } catch (err) {
      console.error('Import failed:', err);
      libraryService.send('ADD_TOAST', {
        data: {
          type: 'error',
          title: 'Import failed',
          message: String(err),
          durationMs: 5000,
        },
      });
    }
  };

  const handleWebDrop = async (files: File[]) => {
    try {
      // Build FormData for upload
      const formData = new FormData();
      for (const file of files) {
        formData.append('files', file);
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
      if (
        currentStateType === 'db' &&
        dbQueryTags.length > 0 &&
        uploadedPaths.length > 0
      ) {
        for (const tagLabel of dbQueryTags) {
          let category = activeCategory || '';
          if (taxonomy) {
            for (const cat of taxonomy) {
              const found = cat.tags?.find(
                (t: any) => t.label === tagLabel
              );
              if (found) {
                category = cat.label;
                break;
              }
            }
          }
          await invoke('create-assignment', [
            uploadedPaths,
            tagLabel,
            category,
            null,
            false,
          ]);
        }
      }

      const count = uploadedPaths.length;
      let message = `Added ${count} file${count !== 1 ? 's' : ''}`;
      if (dbQueryTags.length > 0) {
        message += `, tagged: ${dbQueryTags.join(', ')}`;
      }

      libraryService.send('ADD_TOAST', {
        data: {
          type: 'success',
          title: 'Import complete',
          message,
          durationMs: 4000,
        },
      });

      // Refresh the library
      libraryService.send('REFRESH_LIBRARY');
    } catch (err) {
      console.error('Upload failed:', err);
      libraryService.send('ADD_TOAST', {
        data: {
          type: 'error',
          title: 'Upload failed',
          message: String(err),
          durationMs: 5000,
        },
      });
    }
  };

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
```

- [ ] **Step 2: Add shift-key tracking**

The `NativeTypes.FILE` drop doesn't expose modifier keys directly. Add a global shift tracker at the top of the hook file, below the imports:

```typescript
// Track shift key globally for move-vs-copy on drop
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') (window as any).__shiftHeld = true;
  });
  window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') (window as any).__shiftHeld = false;
  });
}
```

This is already included in the Step 1 code — the `handleDrop` function checks `(window as any).__shiftHeld`.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/hooks/useFileDrop.ts
git commit -m "feat: add useFileDrop hook for native file drag-and-drop import"
```

---

### Task 3: Attach drop zone to panels

**Files:**
- Modify: `src/renderer/components/layout/panels.tsx`
- Modify: `src/renderer/components/layout/panels.css`

- [ ] **Step 1: Wire up the drop zone in panels.tsx**

Replace the contents of `src/renderer/components/layout/panels.tsx`:

```typescript
import React, { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import Layout from './layout';
import useFileDrop from '../../hooks/useFileDrop';
import './panels.css';

export function Panels() {
  const { libraryService } = useContext(GlobalStateContext);
  const { dropRef, isOver, canDrop } = useFileDrop();

  return (
    <>
      <div className="drag-handle" />
      <div
        ref={dropRef}
        className={`Panels${isOver && canDrop ? ' file-drop-active' : ''}`}
        onContextMenu={(e) => {
          e.preventDefault();
          libraryService.send('SHOW_COMMAND_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
      >
        <Layout />
      </div>
    </>
  );
}
```

- [ ] **Step 2: Add the drop indicator style in panels.css**

Add at the end of `src/renderer/components/layout/panels.css`:

```css
.Panels.file-drop-active {
  outline: 2px solid rgba(74, 158, 255, 0.7);
  outline-offset: -2px;
}
```

- [ ] **Step 3: Test visually**

Run the dev server. Drag a file over the app window.

Expected: A blue outline appears around the main content area when hovering with a file. Releasing the file triggers the import flow (toast appears showing result).

- [ ] **Step 4: Commit**

```bash
git add src/renderer/components/layout/panels.tsx src/renderer/components/layout/panels.css
git commit -m "feat: attach native file drop zone to main panels with visual indicator"
```

---

### Task 4: Handle `initialFile` directory resolution

**Context:** The `initialFile` in the state machine is the path that was originally loaded — it could be a file path or a directory path depending on how the user opened it. The hook needs to resolve the browsed directory correctly.

**Files:**
- Modify: `src/renderer/hooks/useFileDrop.ts` (update the FS mode destination logic)

- [ ] **Step 1: Check how `initialFile` is set**

Read `src/renderer/state.tsx` around the `initialFile` assignment in the `loadingFromFS` state to understand what value it holds. Key locations:

- `state.tsx:419` — default: `appArgs?.filePath || ''`
- `state.tsx:1103-1121` — set when user selects a directory or file
- `state.tsx:1549` — set on `LOAD_FROM_PATH` event

The `initialFile` is the path the user selected — when they select a directory, it's the directory. When they select a file, it's that file's path. The `load-files` IPC call uses it as the root for scanning.

- [ ] **Step 2: Read load-files to understand what initialFile represents**

Look at `src/main/load-files.ts` to confirm `initialFile` is used as a directory for scanning.

```
Read: src/main/load-files.ts (the loadFiles function signature and first ~20 lines)
```

If `initialFile` is always a directory when in FS mode, the hook can use it directly. If it can be a file, it needs `path.dirname()`.

- [ ] **Step 3: Update the FS destination logic if needed**

In `src/renderer/hooks/useFileDrop.ts`, the `handleElectronDrop` function's FS branch should be updated based on findings. If `initialFile` is always a directory in FS mode, simplify to:

```typescript
if (currentStateType === 'fs') {
  destination = initialFile;
}
```

If it can be a file, use the directory-resolution logic already written in Task 2.

- [ ] **Step 4: Test FS mode drop**

1. Open a directory in the Electron app
2. Drag an image file from another folder onto the app
3. Expected: File appears in the browsed directory, toast shows "Added 1 file to /path/to/dir"
4. Check the filesystem to confirm the file was copied

- [ ] **Step 5: Test DB mode drop**

1. Switch to tag browsing (click a tag)
2. Drag an image file onto the app
3. If `defaultImportPath` not set: Expected: folder picker dialog appears
4. After setting: Expected: file copied to `defaultImportPath/imports/`, toast shows "Added 1 file, tagged: tagname"
5. Refresh the tag view to confirm the file appears

- [ ] **Step 6: Test shift-to-move**

1. Open a directory in FS mode
2. Hold Shift and drag a file onto the app
3. Expected: File appears in destination, source file is removed
4. Toast shows "Moved 1 file to /path/to/dir"

- [ ] **Step 7: Commit any adjustments**

```bash
git add src/renderer/hooks/useFileDrop.ts
git commit -m "fix: resolve initialFile directory for FS mode file drop"
```

---

### Task 5: Web mode — verify upload integration

**Files:**
- No new files — the web upload path is already in `useFileDrop.ts` from Task 2.

- [ ] **Step 1: Verify `/api/upload` endpoint compatibility**

The existing upload handler in `media-server/main.go:1231` expects multipart form with field name `files`. Confirm by reading the handler:

```go
// In main.go uploadHandler:
files := r.MultipartForm.File["files"]
```

The hook's `FormData` uses `formData.append('files', file)` — this matches.

- [ ] **Step 2: Verify the upload response shape**

The upload handler returns:

```json
{ "success": true, "files": ["uploads/image.jpg"], "message": "Uploaded 1 file(s)" }
```

The hook reads `result.files` — this matches.

- [ ] **Step 3: Test web mode drop**

1. Start the media server with `go run .` (or docker-compose)
2. Open the web UI in a browser
3. Browse by tag, drag a file onto the app
4. Expected: File uploads, toast shows success, tag is applied

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -A
git commit -m "fix: web mode upload integration for drag-and-drop"
```

---

### Task 6: Edge cases and polish

**Files:**
- Modify: `src/renderer/hooks/useFileDrop.ts`

- [ ] **Step 1: Handle multiple file drops**

Already handled — the hook iterates all files. Verify by dropping 3+ files at once and checking the toast shows the correct count.

- [ ] **Step 2: Handle non-media files gracefully**

Already handled — the `isMediaFile` filter skips non-media files. Verify by dropping a `.txt` file — expect the "No media files" warning toast.

- [ ] **Step 3: Handle drop when no database is loaded**

Check the state machine — if the user hasn't loaded a DB yet, imports should be a no-op. Add a guard in `handleDrop`:

At the top of the `handleDrop` callback, before filtering media files, add:

```typescript
const snapshot = libraryService.getSnapshot();
if (!snapshot.matches('ready')) {
  return;
}
```

This checks the machine is in the `ready` state (DB loaded, library available) before allowing imports.

- [ ] **Step 4: Commit**

```bash
git add src/renderer/hooks/useFileDrop.ts
git commit -m "feat: add guard for file drop when app not ready"
```
