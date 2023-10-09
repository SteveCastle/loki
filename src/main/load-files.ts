import readdir from 'readdir-enhanced';

import * as path from 'path';
import naturalCompare from 'natural-compare';

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
    value: /jpg$|jpeg$|jfif$|webp$|png$|webm$|mp4$|mpeg$|gif$|mkv$/i,
  },
  static: {
    label: 'Static',
    value: /jpg$|jpeg$|webp$|jfif$|png$/i,
  },
  video: {
    label: 'Videos',
    value: /mp4$|mpeg$|gif$|webm$|mkv$|mkv$/i,
  },
};

const readdirStreamAsync = (
  folderPath: string,
  filter: string,
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

export default async function loadFiles(
  filePath: string,
  filter = 'all',
  sortOrder = 'name',
  recursive = false
) {
  // folderPath is the directory the filePath is in.
  const folderPath = path.dirname(filePath);

  // fileName is the fileName portion of the filePath
  const fileName = path.basename(filePath);

  // Get the list of files in the folderPath
  const files = await readdirStreamAsync(folderPath, filter, recursive);

  const sortedFiles = files.sort(sorts[sortOrder]);
  console.log('sorting with', sortOrder);
  const cursorIndex = sortedFiles.findIndex(
    (item) => path.basename(item.path) === fileName
  );
  const cursor = cursorIndex === -1 ? 0 : cursorIndex;

  return {
    library: sortedFiles,
    cursor,
  };
}
