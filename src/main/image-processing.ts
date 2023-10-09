const workerpool = require('workerpool');

const pool = workerpool.pool(__dirname + '/image-processing-worker.js');

function asyncCreateThumbnail(
  filePath: string,
  basePath: string,
  cache: string,
  timeStamp = 0
): Promise<string> {
  return pool.exec('createThumbnail', [filePath, basePath, cache, timeStamp]);
}

export { asyncCreateThumbnail };
