import path from 'path';
import fs from 'fs';
import os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { app, BrowserWindow, protocol, shell } from 'electron';

const execFileP = promisify(execFile);

// Lowkey Studio (the WebGPU compositor under /studio) is a plain static
// ES-module web app. Module scripts, fetch, and WebGPU all require a real
// secure origin — file:// doesn't qualify — so the studio is served through
// a privileged custom scheme instead of pointing a window at the filesystem.
//
// One origin (`studio://app`), three routes:
//   studio://app/<file>              → static file from the studio directory
//   studio://app/launch-media?path=… → streams a local media file; same origin,
//                                      so the studio's boot importer can fetch()
//                                      it without CORS.
//   PUT studio://app/save-media?path=… → "edit and save": accepts the studio's
//                                      offline WebM render and writes it back
//                                      over the original file (transcoding via
//                                      the bundled ffmpeg when the original
//                                      isn't .webm). Restricted to paths that
//                                      were launched into the studio.

const studioRoot = () =>
  app.isPackaged
    ? path.join(process.resourcesPath, 'studio')
    : path.join(__dirname, '../../studio');

const staticMimeTypes: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain',
};

// Media served to the studio importer. The studio turns the response into a
// File and creates blob: URLs for playback, so no Range support is needed —
// unlike gsm://, this route only ever streams whole files.
const mediaMimeTypes: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.avi': 'video/x-msvideo',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.jfif': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.bmp': 'image/bmp',
};

// Video extensions the studio can meaningfully edit-and-save in place.
const SAVABLE_VIDEO_RE = /\.(mp4|webm|mov|mkv|avi|m4v|flv)$/i;

// Images can't be saved in place — the studio's render is video — so they
// save alongside the original: same basename, .mp4 extension. Repeat saves
// keep overwriting that derived file, so iteration doesn't pile up copies.
const SAVABLE_IMAGE_RE = /\.(jpe?g|jfif|png|webp|avif|bmp|gif)$/i;

// Save-back allowlist: only files that were actually launched into the studio
// window may be overwritten through /save-media.
const savablePaths = new Set<string>();

// Mirrors metadata.ts's getBinaryPath: dev runs main.ts from src/main so
// resources/bin sits next to it; packaged builds ship the binaries to
// resources/bin via extraResources.
function getBinaryPath(binaryName: string): string {
  let platformDir = 'linux';
  let binaryFile = binaryName;
  if (process.platform === 'darwin') {
    platformDir = 'darwin';
  } else if (process.platform === 'win32') {
    platformDir = 'win32';
    binaryFile = `${binaryName}.exe`;
  }
  return app.isPackaged
    ? path.join(__dirname, '../../../bin', binaryFile)
    : path.join(__dirname, 'resources/bin', platformDir, binaryFile);
}

function streamFile(filePath: string, size: number, contentType: string) {
  const stream = fs.createReadStream(filePath);
  return new Response(
    new ReadableStream({
      start(controller) {
        stream.on('data', (chunk: Buffer) =>
          controller.enqueue(new Uint8Array(chunk))
        );
        stream.on('end', () => controller.close());
        stream.on('error', (e) => controller.error(e));
      },
      cancel() {
        stream.destroy();
      },
    }),
    {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': size.toString(),
      },
    }
  );
}

async function serveLaunchMedia(mediaPath: string | null): Promise<Response> {
  if (!mediaPath) return new Response('Bad Request', { status: 400 });
  const filePath = path.normalize(mediaPath);
  let stats: fs.Stats;
  try {
    stats = await fs.promises.stat(filePath);
  } catch {
    return new Response('Not Found', { status: 404 });
  }
  const ext = path.extname(filePath).toLowerCase();
  const contentType = mediaMimeTypes[ext] || 'application/octet-stream';
  return streamFile(filePath, stats.size, contentType);
}

const jsonResponse = (status: number, body: object) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

// Write the request body (the studio's offline WebM render) over the original
// file. Non-webm targets are transcoded with the bundled ffmpeg so the
// container still matches the file's extension. The original is renamed to a
// backup for the duration of the swap, so a failed write can't destroy it.
async function handleSaveMedia(
  request: Request,
  targetPath: string | null
): Promise<Response> {
  if (!targetPath) return jsonResponse(400, { ok: false, error: 'missing path' });
  const target = path.normalize(targetPath);
  if (!savablePaths.has(target)) {
    return jsonResponse(403, {
      ok: false,
      error: 'path was not opened by the studio',
    });
  }
  if (!request.body) {
    return jsonResponse(400, { ok: false, error: 'empty body' });
  }

  const tmpDir = await fs.promises.mkdtemp(
    path.join(os.tmpdir(), 'studio-save-')
  );
  const uploadPath = path.join(tmpDir, 'render.webm');
  const targetExt = path.extname(target).toLowerCase();
  const backupPath = `${target}.studio-bak`;
  try {
    await new Promise<void>((resolve, reject) => {
      const out = fs.createWriteStream(uploadPath);
      out.on('finish', resolve);
      out.on('error', reject);
      const reader = (request.body as ReadableStream<Uint8Array>).getReader();
      const pump = () => {
        reader
          .read()
          .then(({ done, value }) => {
            if (done) {
              out.end();
              return;
            }
            out.write(Buffer.from(value), (err) => (err ? reject(err) : pump()));
          })
          .catch(reject);
      };
      pump();
    });

    let finalPath = uploadPath;
    if (targetExt !== '.webm') {
      finalPath = path.join(tmpDir, `out${targetExt}`);
      const args = [
        '-y',
        '-i',
        uploadPath,
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        '18',
        '-preset',
        'veryfast',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
      ];
      if (['.mp4', '.m4v', '.mov'].includes(targetExt)) {
        args.push('-movflags', '+faststart');
      }
      args.push(finalPath);
      await execFileP(getBinaryPath('ffmpeg'), args, {
        maxBuffer: 32 * 1024 * 1024,
      });
    }

    // Image launches save to a derived .mp4 that may not exist yet — only an
    // existing target gets the backup-swap treatment.
    const existed = await fs.promises
      .stat(target)
      .then(() => true)
      .catch(() => false);
    if (existed) await fs.promises.rename(target, backupPath);
    try {
      await fs.promises.copyFile(finalPath, target);
    } catch (err) {
      if (existed) {
        await fs.promises.rename(backupPath, target).catch(() => undefined);
      } else {
        await fs.promises.unlink(target).catch(() => undefined);
      }
      throw err;
    }
    if (existed) await fs.promises.unlink(backupPath).catch(() => undefined);

    // Tell the viewer window(s) what happened on disk (handled in
    // toast-system.tsx): an overwrite busts caches and reloads the media
    // element; a brand-new file refreshes the library and navigates to it.
    for (const w of BrowserWindow.getAllWindows()) {
      if (w !== studioWindow) {
        w.webContents.send('studio-media-saved', [target], !existed);
      }
    }
    return jsonResponse(200, { ok: true });
  } catch (error) {
    console.error('studio save-media error:', error);
    return jsonResponse(500, {
      ok: false,
      error: String((error as Error)?.message ?? error),
    });
  } finally {
    fs.promises.rm(tmpDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

// Called from main.ts inside app.on('ready') — protocol.handle is only
// available once the app is ready (the scheme itself is registered as
// privileged before ready, alongside gsm).
export function registerStudioProtocol() {
  protocol.handle('studio', async (request) => {
    try {
      const url = new URL(request.url);
      if (url.host !== 'app') return new Response('Not Found', { status: 404 });

      if (url.pathname === '/launch-media') {
        return await serveLaunchMedia(url.searchParams.get('path'));
      }

      if (url.pathname === '/save-media') {
        if (request.method !== 'PUT') {
          return new Response('Method Not Allowed', { status: 405 });
        }
        return await handleSaveMedia(request, url.searchParams.get('path'));
      }

      const root = studioRoot();
      const rel =
        decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
      const filePath = path.normalize(path.join(root, rel));
      if (!filePath.startsWith(path.normalize(root + path.sep))) {
        return new Response('Forbidden', { status: 403 });
      }

      let stats: fs.Stats;
      try {
        stats = await fs.promises.stat(filePath);
      } catch {
        return new Response('Not Found', { status: 404 });
      }
      if (!stats.isFile()) return new Response('Not Found', { status: 404 });

      const ext = path.extname(filePath).toLowerCase();
      const contentType = staticMimeTypes[ext] || 'application/octet-stream';
      return streamFile(filePath, stats.size, contentType);
    } catch (error) {
      console.error('studio protocol error:', error);
      return new Response('Internal Error', { status: 500 });
    }
  });
}

let studioWindow: BrowserWindow | null = null;

// Open (or reuse) the studio window. Media paths become ?import= entries the
// studio consumes at boot: each entry's url points back at this origin's
// /launch-media route, so the studio fetches the bytes and imports them
// through its normal File pipeline (see collectLaunchImports in studio/app.js).
export function openStudioWindow(mediaPaths: string[]) {
  const entries = mediaPaths.filter(Boolean).map((p) => {
    const entry: { url: string; name: string; saveUrl?: string; saveName?: string } = {
      url: `/launch-media?path=${encodeURIComponent(p)}`,
      name: path.basename(p),
    };
    // Edit-and-save target: videos save back over the original; images save a
    // video next to the original (same basename, .mp4).
    if (SAVABLE_VIDEO_RE.test(p)) {
      entry.saveUrl = `/save-media?path=${encodeURIComponent(p)}`;
      savablePaths.add(path.normalize(p));
    } else if (SAVABLE_IMAGE_RE.test(p)) {
      const derived = p.replace(/\.[^.\\/]+$/, '.mp4');
      entry.saveUrl = `/save-media?path=${encodeURIComponent(derived)}`;
      entry.saveName = path.basename(derived);
      savablePaths.add(path.normalize(derived));
    }
    return entry;
  });
  // app=1 hides the marketing chrome (site nav) — it only makes sense on the
  // statically published docs site. The studio preserves the flag across its
  // own URL cleanup, so it survives reloads.
  const params = new URLSearchParams({ app: '1' });
  if (entries.length) params.set('import', JSON.stringify(entries));
  const target = `studio://app/index.html?${params.toString()}`;

  if (studioWindow && !studioWindow.isDestroyed()) {
    studioWindow.loadURL(target);
    if (studioWindow.isMinimized()) studioWindow.restore();
    studioWindow.focus();
    return;
  }

  studioWindow = new BrowserWindow({
    show: false,
    width: 1440,
    height: 900,
    backgroundColor: '#111111',
    title: 'Lowkey Studio',
    icon: path.join(studioRoot(), 'studio-icon.png'),
    autoHideMenuBar: true,
    webPreferences: {
      // Plain web app: no preload, no node access. Media reaches it over the
      // studio:// origin only.
      webSecurity: true,
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  studioWindow.loadURL(target);
  studioWindow.once('ready-to-show', () => studioWindow?.show());
  studioWindow.on('closed', () => {
    studioWindow = null;
  });
  studioWindow.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });
}
