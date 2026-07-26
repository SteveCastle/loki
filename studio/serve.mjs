// Dev server for the published docs site (run `node studio/publish.mjs`
// after editing studio/ sources, then browse /studio/).
// Usage: node studio/serve.mjs [port]   →  http://localhost:8790/studio/
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'docs');
const port = parseInt(process.argv[2] ?? '8790', 10);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp4': 'video/mp4',
  '.slang': 'text/plain; charset=utf-8',
  '.slangp': 'text/plain; charset=utf-8',
  '.inc': 'text/plain; charset=utf-8',
};

createServer(async (req, res) => {
  try {
    let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (urlPath.endsWith('/')) urlPath += 'index.html';
    const file = path.join(root, urlPath.replaceAll('/', path.sep));
    if (!file.startsWith(root)) throw new Error('forbidden');
    const data = await readFile(file);
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] ?? 'application/octet-stream',
      // Without this the responses carry no freshness information at all, so
      // Chrome heuristically caches them and a reload after publish.mjs can
      // quietly run the previous build. Never what you want from a dev server.
      'Cache-Control': 'no-store, must-revalidate',
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
}).listen(port, () => {
  console.log(`lowkey docs (incl. studio): http://localhost:${port}/studio/`);
});
