import readdir from 'readdir-enhanced';

import * as path from 'path';
import naturalCompare from 'natural-compare';
import { Database } from './database';
import { IpcMainInvokeEvent } from 'electron';
import { insertBulkMedia } from './media';

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

const readdirStreamAsync = (
  folderPath: string,
  recursive: boolean
): Promise<File[]> => {
  return new Promise((resolve) => {
    const files: File[] = [];
    readdir
      .stream(folderPath, {
        basePath: folderPath,
        deep: recursive,
        filter: filters.all.value,
        stats: true,
      })
      .on('error', (err) => {
        console.log(err);
      })
      .on('data', (data: File) => {
        files.push({ path: data.path, mtimeMs: data.mtimeMs });
      })
      .on('end', () => {
        // end timer for performance
        resolve(files);
      });
  });
};

type LoadFilesInput = [string, string, boolean];

export const loadFiles =
  (db: Database) => async (event: IpcMainInvokeEvent, args: LoadFilesInput) => {
    const [filePath, sortOrder, recursive] = args;
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
    const BATCH_SIZE = 250;

    const flushBatch = () => {
      if (batch.length === 0) return;
      try {
        // Send and clear the batch
        event.sender.send('load-files-batch', batch.splice(0, batch.length));
      } catch (err) {
        // Ignore send errors (window might be gone)
        // but continue processing to resolve invoke
        // console.error('load-files-batch send error', err);
      }
    };

    await new Promise<void>((resolve) => {
      readdir
        .stream(folderPath, {
          basePath: folderPath,
          deep: recursive,
          filter: filters.all.value,
          stats: true,
        })
        .on('error', (err) => {
          console.log(err);
          resolve();
        })
        .on('data', (data: File) => {
          const file: File = { path: data.path, mtimeMs: data.mtimeMs };
          files.push(file);
          batch.push(file);
          if (batch.length >= BATCH_SIZE) {
            flushBatch();
          }
        })
        .on('end', () => {
          flushBatch();
          resolve();
        });
    });

    const sortedFiles = files.sort(sorts[sortOrder]);
    const cursorIndex = fileName
      ? sortedFiles.findIndex((item) => path.basename(item.path) === fileName)
      : 0;
    const cursor = cursorIndex === -1 ? 0 : cursorIndex;

    // Insert into DB after scan to avoid frequent writes
    insertBulkMedia(
      db,
      sortedFiles.map((file) => file.path)
    );

    try {
      event.sender.send('load-files-done', {
        total: sortedFiles.length,
        cursor,
      });
    } catch (err) {
      // ignore
    }

    return {
      library: sortedFiles,
      cursor,
    };
  };
