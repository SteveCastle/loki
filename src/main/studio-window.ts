import path from 'path';
import fs from 'fs';
import { app, BrowserWindow, protocol, shell } from 'electron';

// Lowkey Studio (the WebGPU compositor under /studio) is a plain static
// ES-module web app. Module scripts, fetch, and WebGPU all require a real
// secure origin — file:// doesn't qualify — so the studio is served through
// a privileged custom scheme instead of pointing a window at the filesystem.
//
// One origin (`studio://app`), two routes:
//   studio://app/<file>              → static file from the studio directory
//   studio://app/launch-media?path=… → streams a local media file; same origin,
//                                      so the studio's boot importer can fetch()
//                                      it without CORS.

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
  const entries = mediaPaths.filter(Boolean).map((p) => ({
    url: `/launch-media?path=${encodeURIComponent(p)}`,
    name: path.basename(p),
  }));
  const query = entries.length
    ? `?import=${encodeURIComponent(JSON.stringify(entries))}`
    : '';
  const target = `studio://app/index.html${query}`;

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
