const workerpool = require('workerpool');

const pool = workerpool.pool(__dirname + '/image-processing-worker.js');

// Deduplicate concurrent thumbnail generation requests for the same target
const inFlightThumbnails: Map<string, Promise<string>> = new Map();

function asyncCreateThumbnail(
  filePath: string,
  basePath: string,
  cache: string,
  timeStamp = 0
): Promise<string> {
  const key = [filePath, basePath, cache, timeStamp].join('|');
  const existing = inFlightThumbnails.get(key);
  if (existing) return existing;

  const promise = pool
    .exec('createThumbnail', [filePath, basePath, cache, timeStamp])
    .finally(() => {
      inFlightThumbnails.delete(key);
    });
  inFlightThumbnails.set(key, promise);
  return promise;
}

export { asyncCreateThumbnail };
