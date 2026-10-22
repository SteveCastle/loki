import * as path from 'path';
import naturalCompare from 'natural-compare';
import { Database } from './database';
import { IpcMainInvokeEvent } from 'electron';
import { insertBulkMedia } from './media';
import fsPromises from 'fs/promises';
import os from 'os';
import { spawn, ChildProcess } from 'child_process';
import { isArchivePath, extractArchive } from './archives';
import { logEvent } from './errorLog';

// A directory scan delegates to an external process (PowerShell on Windows,
// `find` on macOS). If that child stalls — a disconnected network drive, a
// wedged shell — it can emit nothing and never close, leaving the scan promise
// pending forever and stranding the renderer on a permanent loading spinner.
// This watchdog kills a child that has produced no output for too long so the
// promise always settles (with whatever partial results we have).
const SCAN_INACTIVITY_MS = 30_000;

function watchChildInactivity(
  child: ChildProcess,
  label: string,
  rootDir: string,
  onStall: () => void
): { bump: () => void; clear: () => void } {
  let last = Date.now();
  const interval = setInterval(() => {
    if (Date.now() - last < SCAN_INACTIVITY_MS) return;
    clearInterval(interval);
    logEvent({
      scope: 'ipc:load-files',
      message: `${label} produced no output for ${SCAN_INACTIVITY_MS}ms; killing stalled child`,
      data: { pid: child.pid, rootDir },
    });
    try {
      child.kill();
    } catch {
      // best-effort
    }
    onStall();
  }, 5000);
  return {
    bump: () => {
      last = Date.now();
    },
    clear: () => clearInterval(interval),
  };
}

// Cancellation: the user can leave a slow scan (open another folder, pick a
// new file, or stop it outright). Killing the child is what actually frees the
// disk/network; settling immediately lets loadFiles return its partial
// results instead of waiting for the child's close.
function watchAbort(
  child: ChildProcess,
  signal: AbortSignal | undefined,
  settle: () => void
) {
  if (!signal) return;
  const onAbort = () => {
    try {
      child.kill();
    } catch {
      // best-effort
    }
    settle();
  };
  if (signal.aborted) {
    onAbort();
    return;
  }
  signal.addEventListener('abort', onAbort, { once: true });
  child.once('close', () => signal.removeEventListener('abort', onAbort));
}

// One live scan per window: a new load-files from the same renderer
// supersedes (aborts) the previous one, and cancel-load-files aborts it
// without starting another.
const activeScans = new Map<
  number,
  { controller: AbortController; scanId?: string }
>();

// args[0] is the scanId being abandoned; a cancel that arrives after a newer
// scan has started must not take that one down with it.
export const cancelLoadFiles = (
  event: { sender: { id: number } },
  args?: [string?]
) => {
  const scan = activeScans.get(event.sender.id);
  if (!scan) return;
  const scanId = args?.[0];
  if (scanId && scan.scanId && scanId !== scan.scanId) return;
  activeScans.delete(event.sender.id);
  scan.controller.abort();
};

type File = {
  path: string;
  mtimeMs: number;
};

type Sorts = {
  [key: string]: (a: File, b: File) => number;
};

type Filters = {
  [key: string]: {
    label: string;
    value: RegExp;
  };
};

const sorts: Sorts = {
  date: (a: File, b: File) => b.mtimeMs - a.mtimeMs,
  name: (a: File, b: File) =>
    naturalCompare(a.path.toLowerCase(), b.path.toLowerCase()),
};

const filters: Filters = {
  all: {
    label: 'All',
    value:
      /jpg$|jpeg$|jfif$|webp$|avif$|heic$|heif$|png$|webm$|mp4$|mov$|mpeg$|gif$|mkv|m4v$|mp3$|wav$|flac$|aac$|ogg$|m4a$|opus$|wma$|aiff$|ape$/i,
  },
  static: {
    label: 'Static',
    value: /jpg$|jpeg$|webp$|avif$|heic$|heif$|jfif$|png$/i,
  },
  video: {
    label: 'Videos',
    value: /mp4$|mpeg$|gif$|webm$|mkv$|m4v$/i,
  },
  audio: {
    label: 'Audio',
    value: /mp3$|wav$|flac$|aac$|ogg$|m4a$|opus$|wma$|aiff$|ape$/i,
  },
};

async function walkDirectory(
  rootDir: string,
  recursive: boolean,
  filterRegex: RegExp,
  onFile: (file: File) => void,
  needMtime: boolean,
  signal?: AbortSignal
) {
  const queue: string[] = [rootDir];
  while (queue.length > 0 && !signal?.aborted) {
    const currentDir = queue.shift() as string;
    let dir;
    try {
      dir = await fsPromises.opendir(currentDir);
    } catch (e) {
      continue;
    }
    for await (const dirent of dir) {
      // Breaking out of for-await closes the Dir handle.
      if (signal?.aborted) break;
      const fullPath = path.join(currentDir, dirent.name);
      if (dirent.isDirectory()) {
        if (recursive) queue.push(fullPath);
        continue;
      }
      if (!filterRegex.test(dirent.name)) continue;
      if (needMtime) {
        try {
          const stat = await fsPromises.stat(fullPath);
          onFile({ path: fullPath, mtimeMs: stat.mtimeMs });
        } catch (e) {
          // ignore files we cannot stat
        }
      } else {
        onFile({ path: fullPath, mtimeMs: 0 });
      }
    }
  }
}

async function listFilesFastDarwin(
  rootDir: string,
  recursive: boolean,
  filterRegex: RegExp,
  onFile: (file: File) => void,
  needMtime: boolean,
  signal?: AbortSignal
) {
  return new Promise<void>((resolve) => {
    // If not recursive, a simple directory read is faster and avoids BSD find limitations
    if (!recursive) {
      (async () => {
        try {
          const dir = await fsPromises.opendir(rootDir);
          for await (const dirent of dir) {
            if (signal?.aborted) break;
            if (dirent.isDirectory()) continue;
            const fullPath = path.join(rootDir, dirent.name);
            const base = dirent.name;
            if (!filterRegex.test(base)) continue;
            if (needMtime) {
              try {
                const st = await fsPromises.stat(fullPath);
                onFile({ path: fullPath, mtimeMs: st.mtimeMs });
              } catch {
                onFile({ path: fullPath, mtimeMs: 0 });
              }
            } else {
              onFile({ path: fullPath, mtimeMs: 0 });
            }
          }
        } catch (e) {
          // ignore
        } finally {
          resolve();
        }
      })();
      return;
    }

    const args = [rootDir, '-type', 'f', '-print0'];
    console.log('[fastest-mac] Spawning find:', { cmd: 'find', args });
    const child = spawn('find', args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    console.log('[fastest-mac] child pid:', child.pid);

    let buffer = '';
    const filesQueued: string[] = [];
    let activeStats = 0;
    let settled = false;
    // Control holder breaks the settle<->watchdog cycle so watchdog can be const.
    const ctl: { settle: () => void } = { settle: () => undefined };
    const watchdog = watchChildInactivity(child, 'find scan', rootDir, () =>
      ctl.settle()
    );
    ctl.settle = () => {
      if (settled) return;
      settled = true;
      watchdog.clear();
      resolve();
    };
    const settle = ctl.settle;
    watchAbort(child, signal, settle);
    const MAX_CONC_STATS = needMtime ? 32 : 0;

    const trySchedule = () => {
      if (!needMtime) return;
      while (activeStats < MAX_CONC_STATS && filesQueued.length > 0) {
        const p = filesQueued.shift() as string;
        activeStats += 1;
        fsPromises
          .stat(p)
          .then((st) => {
            onFile({ path: p, mtimeMs: st.mtimeMs });
          })
          .catch(() => {
            onFile({ path: p, mtimeMs: 0 });
          })
          .finally(() => {
            activeStats -= 1;
            trySchedule();
          });
      }
    };

    const handlePath = (filePath: string) => {
      const base = path.basename(filePath);
      if (!filterRegex.test(base)) return;
      if (needMtime) {
        filesQueued.push(filePath);
        trySchedule();
      } else {
        onFile({ path: filePath, mtimeMs: 0 });
      }
    };

    child.stdout.on('data', (chunk: Buffer) => {
      watchdog.bump();
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\0')) !== -1) {
        const filePath = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        if (filePath) handlePath(filePath);
      }
    });
    child.stderr.on('data', (chunk: Buffer) => {
      console.log('[fastest-mac] stderr:', chunk.toString('utf8'));
    });
    const finalize = () => {
      const onIdle = () => {
        if (activeStats === 0) settle();
        else setTimeout(onIdle, 10);
      };
      onIdle();
    };
    child.on('error', (err) => {
      console.log('[fastest-mac] child error:', err);
      logEvent({
        scope: 'ipc:load-files',
        message: 'find scan child error',
        data: { rootDir },
        error: err,
      });
      finalize();
    });
    child.on('close', () => {
      if (!settled && buffer.length > 0) {
        const filePath = buffer;
        if (filePath) handlePath(filePath);
      }
      finalize();
    });
  });
}

async function listFilesFastWindows(
  rootDir: string,
  recursive: boolean,
  filterRegex: RegExp,
  onFile: (file: File) => void,
  signal?: AbortSignal
) {
  return new Promise<void>((resolve) => {
    // Use PowerShell for robust quoting, UNC support, and UTF-8 output
    const psPath = rootDir.replace(/'/g, "''");
    const recurseFlag = recursive ? '-Recurse' : '';
    const psCommand = `[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-ChildItem -LiteralPath '${psPath}' -File ${recurseFlag} -Force | ForEach-Object { '{0}\t{1}' -f $_.FullName, ([DateTimeOffset]$_.LastWriteTimeUtc).ToUnixTimeMilliseconds() }`;
    console.log('[fastest] Spawning powershell.exe for dir:', {
      rootDir,
      psCommand,
    });
    const child = spawn(
      'powershell.exe',
      ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', psCommand],
      {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      }
    );
    console.log('[fastest] child pid:', child.pid);
    let buffer = '';
    let totalBytes = 0;
    let lineCount = 0;
    let settled = false;
    // Control holder breaks the settle<->watchdog cycle so watchdog can be const.
    const ctl: { settle: () => void } = { settle: () => undefined };
    const watchdog = watchChildInactivity(child, 'powershell scan', rootDir, () =>
      ctl.settle()
    );
    ctl.settle = () => {
      if (settled) return;
      settled = true;
      watchdog.clear();
      resolve();
    };
    const settle = ctl.settle;
    watchAbort(child, signal, settle);
    child.stdout.on('data', (chunk: Buffer) => {
      if (settled) return;
      watchdog.bump();
      totalBytes += chunk.length;
      buffer += chunk.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).replace(/\r$/, '');
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let filePath = line;
        let mtimeMs = 0;
        const sep = line.lastIndexOf('\t');
        if (sep > -1) {
          filePath = line.slice(0, sep);
          const ts = line.slice(sep + 1);
          const n = Number(ts);
          mtimeMs = Number.isFinite(n) ? n : 0;
        }
        const base = path.basename(filePath);
        if (filterRegex.test(base)) {
          onFile({ path: filePath, mtimeMs });
        }
        lineCount += 1;
      }
    });
    child.stderr.on('data', (chunk: Buffer) => {
      console.log('[fastest] stderr:', chunk.toString('utf8'));
    });
    const flushRemainder = () => {
      if (settled) return;
      const trimmed = buffer.trim();
      if (trimmed.length > 0) {
        let filePath = trimmed;
        let mtimeMs = 0;
        const sep = trimmed.lastIndexOf('\t');
        if (sep > -1) {
          filePath = trimmed.slice(0, sep);
          const ts = trimmed.slice(sep + 1);
          const n = Number(ts);
          mtimeMs = Number.isFinite(n) ? n : 0;
        }
        const base = path.basename(filePath);
        if (filterRegex.test(base)) {
          onFile({ path: filePath, mtimeMs });
        }
        lineCount += 1;
      }
    };
    child.stdout.on('end', () => {
      flushRemainder();
      console.log(
        '[fastest] stdout end, bytes:',
        totalBytes,
        'lines:',
        lineCount
      );
      settle();
    });
    child.on('error', (err) => {
      console.log('[fastest] child error:', err);
      logEvent({
        scope: 'ipc:load-files',
        message: 'powershell scan child error',
        data: { rootDir },
        error: err,
      });
      settle();
    });
    child.on('close', (code, signal) => {
      flushRemainder();
      console.log('[fastest] child close:', {
        code,
        signal,
        bytes: totalBytes,
        lines: lineCount,
      });
      settle();
    });
  });
}

type LoadFilesOptions = {
  fastest?: boolean;
  skipStat?: boolean;
  // Echoed on load-files-batch / load-files-done so the renderer can drop
  // events from a scan it has already walked away from.
  scanId?: string;
};
type LoadFilesInput =
  | [string, string, boolean]
  | [string, string, boolean, LoadFilesOptions];

type RefreshLibraryInput = {
  initialFile: string;
  currentPaths: string[];
  recursive: boolean;
};

type RefreshLibraryResult = {
  added: File[];
  removed: string[];
};

function releaseScan(controller: AbortController, senderId: number) {
  if (activeScans.get(senderId)?.controller === controller) {
    activeScans.delete(senderId);
  }
}

function cancelledResult(controller: AbortController, senderId: number) {
  releaseScan(controller, senderId);
  return { library: [] as File[], cursor: 0, cancelled: true };
}

export const loadFiles =
  (db: Database) => async (event: IpcMainInvokeEvent, args: LoadFilesInput) => {
    const filePath = args[0] as string;
    const sortOrder = args[1] as string;
    const recursive = args[2] as boolean;
    const options: LoadFilesOptions = (args as any)[3] || {};
    const senderId = event.sender.id;
    activeScans.get(senderId)?.controller.abort();
    const controller = new AbortController();
    activeScans.set(senderId, { controller, scanId: options.scanId });
    const { signal } = controller;
    let folderPath: string;
    let fileName: string;
    let archiveExtracted = false;

    try {
      if (isArchivePath(filePath)) {
        const extracted = await extractArchive(filePath);
        folderPath = extracted;
        fileName = '';
        archiveExtracted = true;
      } else {
        const stats = await fsPromises.lstat(filePath);
        if (stats.isDirectory()) {
          folderPath = filePath;
          fileName = '';
        } else {
          folderPath = path.dirname(filePath);
          fileName = path.basename(filePath);
        }
      }
    } catch {
      folderPath = path.dirname(filePath);
      fileName = path.basename(filePath);
    }

    const effectiveRecursive = archiveExtracted ? true : recursive;
    logEvent({
      level: 'info',
      scope: 'ipc:load-files',
      message: 'load-files start',
      data: {
        filePath,
        folderPath,
        recursive: effectiveRecursive,
        fastest: !!options.fastest,
      },
    });
    if (signal.aborted) return cancelledResult(controller, senderId);

    // Stream the directory and emit batches to the renderer for incremental UI updates
    const files: File[] = [];
    const batch: File[] = [];
    const BATCH_SIZE = 500;

    const flushBatch = () => {
      if (batch.length === 0) return;
      try {
        // Send and clear the batch
        event.sender.send(
          'load-files-batch',
          batch.splice(0, batch.length),
          options.scanId
        );
        console.log('[loader] sent batch, cumulative files:', files.length);
      } catch (err) {
        // Ignore send errors (window might be gone)
        // but continue processing to resolve invoke
        // console.error('load-files-batch send error', err);
      }
    };

    const needMtime = sortOrder === 'date' && !options.skipStat;
    console.log('[loader] start', {
      filePath,
      sortOrder,
      recursive,
      options,
      needMtime,
      BATCH_SIZE,
    });
    const onFile = (file: File) => {
      if (signal.aborted) return;
      files.push(file);
      batch.push(file);
      if (batch.length >= BATCH_SIZE) {
        flushBatch();
      }
    };
    if (options.fastest && os.platform() === 'win32') {
      console.log(
        '[loader] using fastest Windows mode for folder:',
        folderPath
      );
      await listFilesFastWindows(
        folderPath,
        effectiveRecursive,
        filters.all.value,
        onFile,
        signal
      );
    } else if (options.fastest && os.platform() === 'darwin') {
      console.log('[loader] using fastest macOS mode for folder:', folderPath);
      await listFilesFastDarwin(
        folderPath,
        effectiveRecursive,
        filters.all.value,
        onFile,
        sortOrder === 'date' && !options.skipStat,
        signal
      );
    } else {
      console.log('[loader] using Node walker for folder:', folderPath);
      await walkDirectory(
        folderPath,
        effectiveRecursive,
        filters.all.value,
        onFile,
        needMtime,
        signal
      );
    }
    if (signal.aborted) {
      // Superseded or stopped: the renderer has already moved on and keeps
      // whatever streamed in, so skip the final batch and done event. Still
      // import what was found — those files were on screen and may be acted
      // on like any other library item.
      insertBulkMedia(
        db,
        files.map((file) => file.path)
      );
      logEvent({
        level: 'info',
        scope: 'ipc:load-files',
        message: 'load-files cancelled',
        data: { filePath, scanned: files.length },
      });
      return cancelledResult(controller, senderId);
    }
    releaseScan(controller, senderId);
    flushBatch();

    const effectiveSort =
      sortOrder === 'date' && (options.skipStat || options.fastest)
        ? 'name'
        : sortOrder;
    console.log('[loader] sorting', { effectiveSort, total: files.length });
    const sortedFiles = files.sort(sorts[effectiveSort]);
    // Use case-insensitive matching for Windows compatibility
    const cursorIndex = fileName
      ? sortedFiles.findIndex(
          (item) =>
            path.basename(item.path).toLowerCase() === fileName.toLowerCase()
        )
      : 0;
    const cursor = cursorIndex === -1 ? 0 : cursorIndex;

    // Insert into DB after scan to avoid frequent writes
    console.log('[loader] inserting into DB', { count: sortedFiles.length });
    insertBulkMedia(
      db,
      sortedFiles.map((file) => file.path)
    );

    logEvent({
      level: 'info',
      scope: 'ipc:load-files',
      message: 'load-files done',
      data: { total: sortedFiles.length, cursor },
    });

    try {
      console.log('[loader] sending done', {
        total: sortedFiles.length,
        cursor,
      });
      event.sender.send('load-files-done', {
        total: sortedFiles.length,
        cursor,
        scanId: options.scanId,
      });
    } catch (err) {
      console.log('[loader] error sending done', err);
    }

    return {
      library: sortedFiles,
      cursor,
    };
  };

export const refreshLibrary =
  (db: Database) =>
  async (
    event: IpcMainInvokeEvent,
    args: [RefreshLibraryInput]
  ): Promise<RefreshLibraryResult> => {
    const { initialFile, currentPaths, recursive } = args[0];
    // Archives are read-only; refresh is a no-op. The extracted temp dir is
    // a cached snapshot — rescanning it would never find anything new.
    if (isArchivePath(initialFile)) {
      return { added: [], removed: [] };
    }
    let folderPath: string;
    try {
      const stats = await fsPromises.lstat(initialFile);
      folderPath = stats.isDirectory() ? initialFile : path.dirname(initialFile);
    } catch (e) {
      // The file this view was opened from may itself be gone by now — a
      // dedupe or merge deletes files the library still lists, and the opener
      // is as deletable as any other item. The FOLDER is what a refresh is
      // about, so fall back to the file's directory before giving up.
      folderPath = path.dirname(initialFile);
      try {
        const stats = await fsPromises.lstat(folderPath);
        if (!stats.isDirectory()) {
          throw new Error(`not a directory: ${folderPath}`);
        }
      } catch (dirErr) {
        // The whole folder is unreachable. That stats exactly like an
        // unplugged drive, and reporting every file as removed would let one
        // refresh purge a library that still exists — so report no changes.
        console.log('[refresh] folder unreachable, skipping:', dirErr);
        return { added: [], removed: [] };
      }
    }

    // Build a set of current paths for fast lookup (case-insensitive on Windows)
    const isWindows = os.platform() === 'win32';
    const normalizePath = (p: string) =>
      isWindows ? p.toLowerCase() : p;
    const currentPathSet = new Set(currentPaths.map(normalizePath));

    // Scan the filesystem for current files
    const filesOnDisk: File[] = [];

    console.log('[refresh] scanning folder:', folderPath, { recursive });

    if (os.platform() === 'win32') {
      await listFilesFastWindows(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          filesOnDisk.push(file);
        }
      );
    } else if (os.platform() === 'darwin') {
      await listFilesFastDarwin(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          filesOnDisk.push(file);
        },
        false // don't need mtime for refresh
      );
    } else {
      await walkDirectory(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          filesOnDisk.push(file);
        },
        false // don't need mtime for refresh
      );
    }

    // Build a set of paths found on disk
    const diskPathSet = new Set(filesOnDisk.map((f) => normalizePath(f.path)));

    // Find new files (on disk but not in current library)
    const added = filesOnDisk.filter(
      (f) => !currentPathSet.has(normalizePath(f.path))
    );

    // Find removed files (in current library but not on disk)
    const removed = currentPaths.filter(
      (p) => !diskPathSet.has(normalizePath(p))
    );

    console.log('[refresh] scan complete:', {
      onDisk: filesOnDisk.length,
      current: currentPaths.length,
      added: added.length,
      removed: removed.length,
    });

    // Insert new files into DB
    if (added.length > 0) {
      insertBulkMedia(
        db,
        added.map((file) => file.path)
      );
    }

    return { added, removed };
  };
