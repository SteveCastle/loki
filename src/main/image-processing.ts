const workerpool = require('workerpool');
const log = require('electron-log');
const path = require('path');
const { app } = require('electron');

// Use separate processes to isolate libvips (sharp) from Electron/worker-threads
const thumbLogPath = path.join(app.getPath('userData'), 'thumbnail-worker.log');
const pool = workerpool.pool(__dirname + '/image-processing-worker.js', {
  workerType: 'process',
  minWorkers: 1,
  maxWorkers: 1,
  forkOpts: {
    env: {
      ...process.env,
      THUMB_LOG: thumbLogPath,
      NODE_NO_WARNINGS: '1',
    },
  },
});

// Ensure workers terminate on exit/reload
const terminatePool = () => {
  try {
    pool.terminate(true);
  } catch (_) {}
};
process.on('exit', terminatePool);
process.on('SIGINT', () => {
  terminatePool();
  process.exit(0);
});
process.on('SIGTERM', () => {
  terminatePool();
  process.exit(0);
});

// Deduplicate concurrent thumbnail generation requests for the same target
const inFlightThumbnails: Map<string, Promise<string>> = new Map();

function asyncCreateThumbnail(
  filePath: string,
  basePath: string,
  cache: string,
  timeStamp = 0
): Promise<string> {
  try {
    log.debug?.(
      '[thumb] schedule',
      JSON.stringify({ filePath, basePath, cache, timeStamp })
    );
  } catch (_) {}
  const key = [filePath, basePath, cache, timeStamp].join('|');
  const existing = inFlightThumbnails.get(key);
  if (existing) return existing;

  const promise = Promise.resolve(
    pool.exec('createThumbnail', [filePath, basePath, cache, timeStamp])
  ).then(
    (result: string) => {
      inFlightThumbnails.delete(key);
      try {
        log.debug?.('[thumb] done', JSON.stringify({ key, result }));
      } catch (_) {}
      return result;
    },
    (error: unknown) => {
      inFlightThumbnails.delete(key);
      try {
        log.error?.(
          '[thumb] failed',
          JSON.stringify({ filePath, basePath, cache, timeStamp }),
          error instanceof Error ? error.stack || error.message : String(error)
        );
        log.error?.('[thumb] worker log path:', thumbLogPath);
      } catch (_) {}
      throw error;
    }
  );
  inFlightThumbnails.set(key, promise);
  return promise;
}

export { asyncCreateThumbnail };
