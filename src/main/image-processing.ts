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

  const promise = Promise.resolve(
    pool.exec('createThumbnail', [filePath, basePath, cache, timeStamp])
  ).then(
    (result: string) => {
      inFlightThumbnails.delete(key);
      return result;
    },
    (error: unknown) => {
      inFlightThumbnails.delete(key);
      throw error;
    }
  );
  inFlightThumbnails.set(key, promise);
  return promise;
}

export { asyncCreateThumbnail };
