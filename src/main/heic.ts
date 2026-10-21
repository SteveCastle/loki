// HEIC/HEIF display conversion. Chromium can't decode HEIC, so anything that
// would hand one to an <img> instead gets a full-resolution JPEG rendered by
// the bundled ffmpeg (8.0+ assembles iPhone tile grids and applies irot/imir)
// with the photo's ICC profile restored (see heic-format.ts).
//
// The JPEGs are cached under userData keyed by source path + mtime + size, so
// an edited file re-renders and an unchanged one is converted once. The gsm://
// handler and thumbnail generation both go through heicDisplayFile, so viewing
// a photo after its thumbnail was made (or vice versa) costs nothing extra.
import { app } from 'electron';
import { execFile } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { logEvent } from './errorLog';
import { isHeicPath, jpegSize, parseHeicMeta, withJpegIcc, type HeicInfo } from './heic-format';

export { isHeicPath };

// The cache is pruned back under this, least recently viewed first. A 12 MP
// iPhone photo converts to ~1.5–3.5 MB.
const CACHE_MAX_BYTES = 2 * 1024 ** 3;
const PRUNE_INTERVAL_MS = 5 * 60 * 1000;
// Each ffmpeg already decodes tiles across cores; more than two at once only
// slows the photo the user is actually waiting on.
const MAX_CONCURRENT = 2;
const CONVERT_TIMEOUT_MS = 60_000;

// Mirrors metadata.ts's getBinaryPath.
function ffmpegPath(): string {
  let platformDir = 'linux';
  let binaryFile = 'ffmpeg';
  if (process.platform === 'darwin') {
    platformDir = 'darwin';
  } else if (process.platform === 'win32') {
    platformDir = 'win32';
    binaryFile = 'ffmpeg.exe';
  }
  return app.isPackaged
    ? path.join(__dirname, '../../../bin', binaryFile)
    : path.join(__dirname, 'resources/bin', platformDir, binaryFile);
}

let cacheDir: string | null = null;
function displayCacheDir(): string {
  if (!cacheDir) cacheDir = path.join(app.getPath('userData'), 'heic-display');
  return cacheDir;
}

const inflight = new Map<string, Promise<string>>();
let running = 0;
const waiting: Array<() => void> = [];

async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (running >= MAX_CONCURRENT) {
    await new Promise<void>((resolve) => waiting.push(resolve));
  } else {
    running++;
  }
  try {
    return await fn();
  } finally {
    const next = waiting.shift();
    if (next) next();
    else running--;
  }
}

/**
 * Path of a browser-decodable JPEG rendering of the HEIC at `srcPath`,
 * converting it on first use. Pass `stats` when the caller already has them.
 */
export async function heicDisplayFile(srcPath: string, stats?: fs.Stats): Promise<string> {
  const st = stats ?? (await fs.promises.stat(srcPath));
  const key = crypto
    .createHash('sha256')
    .update(`${srcPath}|${st.mtimeMs}|${st.size}`)
    .digest('hex');
  const out = path.join(displayCacheDir(), `${key}.jpg`);

  const pending = inflight.get(out);
  if (pending) return pending;
  try {
    await fs.promises.access(out);
    markUsed(out);
    return out;
  } catch {
    // not cached yet
  }

  const job = withSlot(() => convert(srcPath, out)).finally(() => inflight.delete(out));
  inflight.set(out, job);
  return job;
}

async function convert(srcPath: string, out: string): Promise<string> {
  const started = Date.now();
  const [jpeg, info] = await Promise.all([runFfmpeg(srcPath), readHeicInfo(srcPath)]);

  // ffmpeg before 8.0 can't assemble tile grids: it silently returns a single
  // 512×512 tile, or another image in the file entirely (often the depth map).
  // Refuse rather than show the wrong picture.
  const got = jpegSize(jpeg);
  if (info && info.width > 0 && !info.cropped && got &&
      (got.width !== info.width || got.height !== info.height)) {
    throw new Error(
      `ffmpeg decoded ${got.width}x${got.height} but the HEIC's primary image is ` +
        `${info.width}x${info.height}; ffmpeg 8.0+ is needed to assemble HEIC tile grids`
    );
  }
  const body = info?.icc ? withJpegIcc(jpeg, info.icc) : jpeg;

  await fs.promises.mkdir(path.dirname(out), { recursive: true });
  const tmp = `${out}.${process.pid}.${crypto.randomBytes(4).toString('hex')}.tmp`;
  await fs.promises.writeFile(tmp, body);
  try {
    await fs.promises.rename(tmp, out);
  } catch (err) {
    // Windows refuses to replace a file another reader has open; a concurrent
    // conversion of the same source already produced it.
    await fs.promises.unlink(tmp).catch(() => undefined);
    await fs.promises.access(out).catch(() => {
      throw err;
    });
  }
  logEvent({
    level: 'info',
    scope: 'heic',
    message: 'converted',
    data: { path: srcPath, ms: Date.now() - started, bytes: body.length, ...got },
  });
  schedulePrune();
  return out;
}

function runFfmpeg(srcPath: string): Promise<Buffer> {
  // No -vf: ffmpeg builds its own filtergraph to assemble a tile grid and
  // refuses a simple filter on top of it. -pix_fmt still applies at the output.
  const args = [
    '-hide_banner',
    '-loglevel', 'error',
    '-i', srcPath,
    '-frames:v', '1',
    '-pix_fmt', 'yuvj420p',
    '-f', 'image2pipe',
    '-c:v', 'mjpeg',
    '-q:v', '2',
    '-',
  ];
  return new Promise((resolve, reject) => {
    execFile(
      ffmpegPath(),
      args,
      {
        encoding: 'buffer',
        maxBuffer: 512 * 1024 * 1024,
        timeout: CONVERT_TIMEOUT_MS,
        windowsHide: true,
      },
      (err, stdout, stderr) => {
        if (err) {
          const detail = stderr?.toString().trim();
          reject(new Error(`ffmpeg HEIC conversion failed: ${detail || err.message}`));
        } else if (!stdout?.length) {
          reject(new Error('ffmpeg HEIC conversion produced no image'));
        } else {
          resolve(stdout);
        }
      }
    );
  });
}

// The item properties live in the `meta` box, which HEIC writers put before
// the image data, so only the top of the file is read.
async function readHeicInfo(srcPath: string): Promise<HeicInfo | null> {
  let fh: fs.promises.FileHandle | undefined;
  try {
    fh = await fs.promises.open(srcPath, 'r');
    const header = Buffer.alloc(16);
    let offset = 0;
    for (let i = 0; i < 16; i++) {
      const { bytesRead } = await fh.read(header, 0, 16, offset);
      if (bytesRead < 8) return null;
      let size = header.readUInt32BE(0);
      if (size === 1 && bytesRead === 16) size = Number(header.readBigUInt64BE(8));
      const type = header.toString('latin1', 4, 8);
      if (type === 'meta') {
        // A meta box this big isn't a real photo's; don't read it into memory.
        if (size < 16 || size > 64 * 1024 * 1024) return null;
        const meta = Buffer.alloc(size);
        await fh.read(meta, 0, size, offset);
        return parseHeicMeta(meta);
      }
      if (size < 8) return null;
      offset += size;
    }
    return null;
  } catch {
    return null;
  } finally {
    await fh?.close();
  }
}

// Recency is tracked in atime, set explicitly since most filesystems no longer
// update it on read. mtime is left alone: it backs the gsm:// ETag.
function markUsed(file: string) {
  fs.promises
    .stat(file)
    .then((st) => fs.promises.utimes(file, new Date(), st.mtime))
    .catch(() => undefined);
}

let lastPrune = 0;
function schedulePrune() {
  const now = Date.now();
  if (now - lastPrune < PRUNE_INTERVAL_MS) return;
  lastPrune = now;
  pruneCache().catch((err) =>
    logEvent({ level: 'warn', scope: 'heic', message: 'cache prune failed', error: err })
  );
}

async function pruneCache() {
  const dir = displayCacheDir();
  const files: Array<{ file: string; size: number; used: number }> = [];
  let total = 0;
  for (const name of await fs.promises.readdir(dir)) {
    const file = path.join(dir, name);
    try {
      const st = await fs.promises.stat(file);
      // Stray .tmp files from a crash mid-write age out like everything else.
      files.push({ file, size: st.size, used: Math.max(st.atimeMs, st.mtimeMs) });
      total += st.size;
    } catch {
      // removed concurrently
    }
  }
  if (total <= CACHE_MAX_BYTES) return;
  files.sort((a, b) => a.used - b.used);
  // Prune to 75% so the next few conversions don't each trigger a scan.
  for (const f of files) {
    if (total <= CACHE_MAX_BYTES * 0.75) break;
    try {
      await fs.promises.unlink(f.file);
      total -= f.size;
    } catch {
      // in use (Windows) — try again next prune
    }
  }
}
