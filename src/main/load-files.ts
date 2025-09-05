import * as path from 'path';
import naturalCompare from 'natural-compare';
import { Database } from './database';
import { IpcMainInvokeEvent } from 'electron';
import { insertBulkMedia } from './media';
import fsPromises from 'fs/promises';
import os from 'os';
import { spawn } from 'child_process';

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
      /jpg$|jpeg$|jfif$|webp$|png$|webm$|mp4$|mov$|mpeg$|gif$|mkv|m4v$|mp3$|wav$|flac$|aac$|ogg$|m4a$|opus$|wma$|aiff$|ape$/i,
  },
  static: {
    label: 'Static',
    value: /jpg$|jpeg$|webp$|jfif$|png$/i,
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
  needMtime: boolean
) {
  const queue: string[] = [rootDir];
  while (queue.length > 0) {
    const currentDir = queue.shift() as string;
    let dir;
    try {
      dir = await fsPromises.opendir(currentDir);
    } catch (e) {
      continue;
    }
    for await (const dirent of dir) {
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
  needMtime: boolean
) {
  return new Promise<void>((resolve) => {
    const args = [rootDir, '-type', 'f', '-print0'];
    console.log('[fastest-mac] Spawning find:', { cmd: 'find', args });
    const child = spawn('find', args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    console.log('[fastest-mac] child pid:', child.pid);

    let buffer = Buffer.alloc(0);
    let filesQueued: string[] = [];
    let activeStats = 0;
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
      buffer = Buffer.concat([buffer, chunk]);
      let idx;
      while ((idx = buffer.indexOf(0)) !== -1) {
        const filePath = buffer.slice(0, idx).toString('utf8');
        buffer = buffer.slice(idx + 1);
        if (filePath) handlePath(filePath);
      }
    });
    child.stderr.on('data', (chunk: Buffer) => {
      console.log('[fastest-mac] stderr:', chunk.toString('utf8'));
    });
    const finalize = () => {
      const onIdle = () => {
        if (activeStats === 0) resolve();
        else setTimeout(onIdle, 10);
      };
      onIdle();
    };
    child.on('error', (err) => {
      console.log('[fastest-mac] child error:', err);
      finalize();
    });
    child.on('close', () => {
      if (buffer.length > 0) {
        const filePath = buffer.toString('utf8');
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
  onFile: (file: File) => void
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
    child.stdout.on('data', (chunk: Buffer) => {
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
      resolve();
    });
    child.on('error', (err) => {
      console.log('[fastest] child error:', err);
      resolve();
    });
    child.on('close', (code, signal) => {
      flushRemainder();
      console.log('[fastest] child close:', {
        code,
        signal,
        bytes: totalBytes,
        lines: lineCount,
      });
      resolve();
    });
  });
}

type LoadFilesOptions = { fastest?: boolean; skipStat?: boolean };
type LoadFilesInput =
  | [string, string, boolean]
  | [string, string, boolean, LoadFilesOptions];

export const loadFiles =
  (db: Database) => async (event: IpcMainInvokeEvent, args: LoadFilesInput) => {
    const filePath = args[0] as string;
    const sortOrder = args[1] as string;
    const recursive = args[2] as boolean;
    const options: LoadFilesOptions = (args as any)[3] || {};
    const fs = require('fs');

    // Check if the path is a directory or file
    const stats = fs.lstatSync(filePath);
    let folderPath: string;
    let fileName: string;

    if (stats.isDirectory()) {
      // If it's a directory, use it as the folder and set cursor to 0
      folderPath = filePath;
      fileName = '';
    } else {
      // If it's a file, extract directory and filename
      folderPath = path.dirname(filePath);
      fileName = path.basename(filePath);
    }

    // Stream the directory and emit batches to the renderer for incremental UI updates
    const files: File[] = [];
    const batch: File[] = [];
    const BATCH_SIZE = 500;

    const flushBatch = () => {
      if (batch.length === 0) return;
      try {
        // Send and clear the batch
        event.sender.send('load-files-batch', batch.splice(0, batch.length));
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
    if (options.fastest && os.platform() === 'win32') {
      console.log(
        '[loader] using fastest Windows mode for folder:',
        folderPath
      );
      await listFilesFastWindows(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          files.push(file);
          batch.push(file);
          if (batch.length >= BATCH_SIZE) {
            flushBatch();
          }
        }
      );
    } else if (options.fastest && os.platform() === 'darwin') {
      console.log('[loader] using fastest macOS mode for folder:', folderPath);
      await listFilesFastDarwin(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          files.push(file);
          batch.push(file);
          if (batch.length >= BATCH_SIZE) {
            flushBatch();
          }
        },
        sortOrder === 'date' && !options.skipStat
      );
    } else {
      console.log('[loader] using Node walker for folder:', folderPath);
      await walkDirectory(
        folderPath,
        recursive,
        filters.all.value,
        (file) => {
          files.push(file);
          batch.push(file);
          if (batch.length >= BATCH_SIZE) {
            flushBatch();
          }
        },
        needMtime
      );
    }
    flushBatch();

    const effectiveSort =
      sortOrder === 'date' && (options.skipStat || options.fastest)
        ? 'name'
        : sortOrder;
    console.log('[loader] sorting', { effectiveSort, total: files.length });
    const sortedFiles = files.sort(sorts[effectiveSort]);
    const cursorIndex = fileName
      ? sortedFiles.findIndex((item) => path.basename(item.path) === fileName)
      : 0;
    const cursor = cursorIndex === -1 ? 0 : cursorIndex;

    // Insert into DB after scan to avoid frequent writes
    console.log('[loader] inserting into DB', { count: sortedFiles.length });
    insertBulkMedia(
      db,
      sortedFiles.map((file) => file.path)
    );

    try {
      console.log('[loader] sending done', {
        total: sortedFiles.length,
        cursor,
      });
      event.sender.send('load-files-done', {
        total: sortedFiles.length,
        cursor,
      });
    } catch (err) {
      console.log('[loader] error sending done', err);
    }

    return {
      library: sortedFiles,
      cursor,
    };
  };
