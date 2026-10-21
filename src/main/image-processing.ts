import { heicDisplayFile, isHeicPath } from './heic';

const workerpool = require('workerpool');

const pool = workerpool.pool(__dirname + '/image-processing-worker.js');

async function asyncCreateThumbnail(
  filePath: string,
  basePath: string,
  cache: string,
  timeStamp = 0
): Promise<string> {
  // HEIC thumbnails are made from the display JPEG (see heic.ts): plain ffmpeg
  // can't put a scale filter on top of a tile grid, and the JPEG carries the
  // ICC profile the grid assembly drops. The thumbnail is still named after
  // the original path.
  const source = isHeicPath(filePath) ? await heicDisplayFile(filePath) : filePath;
  return pool.exec('createThumbnail', [filePath, basePath, cache, timeStamp, source]);
}

export { asyncCreateThumbnail };
