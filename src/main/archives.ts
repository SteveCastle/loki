import * as path from 'path';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as crypto from 'crypto';
import * as os from 'os';
import { spawn } from 'child_process';
import yauzl from 'yauzl';
import { Extensions, FileTypes, getFileType } from '../file-types';

let cacheRoot = path.join(os.tmpdir(), 'lowkey-archives');
const inFlight = new Map<string, Promise<string>>();

/** Test-only: override the cache root before calling extractArchive. */
export function _setCacheRoot(dir: string): void {
  cacheRoot = dir;
}

/**
 * Test-only: override the unrar binary path. Used by the CBR test which
 * stubs unrar with a small shell script so we don't depend on RARLAB's
 * binary being downloaded in CI.
 */
let unrarOverride: string | null = null;
export function _setUnrarPath(p: string | null): void {
  unrarOverride = p;
}

export function isArchivePath(p: string): boolean {
  const ext = path.extname(p).toLowerCase();
  return ext === '.cbz' || ext === '.zip' || ext === '.cbr';
}

/**
 * Resolve the bundled unrar binary. Mirrors metadata.ts's getBinaryPath
 * for ffprobe: in dev we read from src/main/resources/bin, in a packaged
 * app the binary is staged under <resources>/bin via electron-builder's
 * extraResources rule.
 */
function getUnrarPath(): string {
  if (unrarOverride) return unrarOverride;
  const platform = os.platform();
  let platformDir: string;
  let binaryFile = 'unrar';
  if (platform === 'darwin') {
    platformDir = 'darwin';
  } else if (platform === 'win32') {
    platformDir = 'win32';
    binaryFile = 'unrar.exe';
  } else {
    platformDir = 'linux';
  }
  // Lazy-require electron-is-dev: importing it at module load fails under
  // jest (no Electron context), and archives.ts is exercised by unit
  // tests. Resolving only when we actually need the binary keeps the
  // import side-effect-free.
  // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
  const isDev = require('electron-is-dev');
  return isDev
    ? path.join(__dirname, 'resources/bin', platformDir, binaryFile)
    : path.join(__dirname, '../../../bin', binaryFile);
}

function isMediaEntry(name: string): boolean {
  const ft = getFileType(path.basename(name));
  return (
    ft === FileTypes.Image || ft === FileTypes.Video || ft === FileTypes.Audio
  );
}

/** Zip-slip guard: ensure resolved child stays within parent. */
function isSafeEntryPath(entryName: string, extractRoot: string): boolean {
  if (path.isAbsolute(entryName)) return false;
  const resolved = path.resolve(extractRoot, entryName);
  const rel = path.relative(extractRoot, resolved);
  return !rel.startsWith('..') && !path.isAbsolute(rel);
}

async function hashForArchive(archivePath: string): Promise<string> {
  const abs = path.resolve(archivePath);
  const stat = await fsp.stat(abs);
  const h = crypto.createHash('sha1');
  h.update(abs);
  h.update('\0');
  h.update(String(stat.mtimeMs));
  return h.digest('hex').slice(0, 16);
}

async function extractZipTo(
  archivePath: string,
  extractRoot: string
): Promise<void> {
  await fsp.mkdir(extractRoot, { recursive: true });
  await new Promise<void>((resolve, reject) => {
    yauzl.open(
      archivePath,
      { lazyEntries: true, autoClose: true, decodeStrings: false },
      (err, zipfile) => {
        if (err || !zipfile) {
          reject(err ?? new Error('Failed to open zip'));
          return;
        }
        zipfile.on('error', reject);
        zipfile.on('end', () => resolve());
        zipfile.readEntry();
        zipfile.on('entry', (entry) => {
          const name = Buffer.isBuffer(entry.fileName)
            ? entry.fileName.toString('utf8')
            : (entry.fileName as string);
          if (/\/$/.test(name)) {
            zipfile.readEntry();
            return;
          }
          if (!isSafeEntryPath(name, extractRoot)) {
            console.warn('[archives] skipping unsafe entry:', name);
            zipfile.readEntry();
            return;
          }
          if (!isMediaEntry(name)) {
            zipfile.readEntry();
            return;
          }
          zipfile.openReadStream(entry, (rErr, rs) => {
            if (rErr || !rs) {
              reject(rErr ?? new Error('Failed to open entry stream'));
              return;
            }
            const target = path.resolve(extractRoot, name);
            fsp
              .mkdir(path.dirname(target), { recursive: true })
              .then(() => {
                const ws = fs.createWriteStream(target);
                rs.on('error', reject);
                ws.on('error', reject);
                ws.on('close', () => zipfile.readEntry());
                rs.pipe(ws);
              })
              .catch(reject);
          });
        });
      }
    );
  });
}

/**
 * Extract a CBR/RAR archive by shelling out to the bundled unrar binary.
 *
 * unrar's `x` command preserves directory structure inside the archive and
 * — in modern versions — refuses to write paths that escape the
 * destination, but we still walk the result and prune anything outside the
 * destination as a belt-and-suspenders zip-slip guard.
 *
 * Flags:
 *   x       — extract with full paths
 *   -inul   — suppress all stdout/stderr noise (we rely on exit code + a
 *             post-extraction directory check)
 *   -y      — answer yes to any prompts (overwrite, etc.)
 *   -p-     — never query for a password; if the archive is encrypted we
 *             want a clean failure rather than a hung process
 *   -o+     — explicit overwrite of existing files (cache hits short-
 *             circuit before we get here, so this only matters for re-runs)
 *   --      — terminate switches; everything that follows is positional
 *
 * The trailing `archiveMediaMasks` list filters at extraction time so we
 * don't drop ComicInfo.xml etc. on disk in the first place.
 */
const archiveMediaMasks = (() => {
  const exts = [Extensions.Image, Extensions.Video, Extensions.Audio]
    .join('|')
    .split('|')
    .filter(Boolean);
  return Array.from(new Set(exts)).map((e) => `*.${e}`);
})();

async function extractRarTo(
  archivePath: string,
  extractRoot: string
): Promise<void> {
  await fsp.mkdir(extractRoot, { recursive: true });

  const bin = getUnrarPath();
  if (!fs.existsSync(bin)) {
    throw new Error(
      `unrar binary not found at ${bin}. Run src/main/resources/bin/${os.platform()}/download-binaries.sh.`
    );
  }

  // Trailing slash matters: unrar treats the last positional arg as a
  // destination directory only if it ends with the platform path separator.
  const dest = extractRoot.endsWith(path.sep)
    ? extractRoot
    : extractRoot + path.sep;

  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      bin,
      ['x', '-inul', '-y', '-p-', '-o+', '--', archivePath, ...archiveMediaMasks, dest],
      { stdio: ['ignore', 'ignore', 'pipe'] }
    );
    let stderr = '';
    child.stderr?.on('data', (d) => {
      stderr += d.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
      // unrar exit codes: 0 = success, 1 = warning (e.g., a few entries
      // skipped). Anything else is a hard failure for our purposes.
      if (code === 0 || code === 1) {
        resolve();
      } else {
        reject(
          new Error(
            `unrar exited with code ${code}${stderr ? `: ${stderr.trim()}` : ''}`
          )
        );
      }
    });
  });

  // Belt-and-suspenders: walk the result and remove any file that
  // resolved outside extractRoot or isn't a media entry. unrar should
  // have already filtered both, but we never trust an external tool with
  // path-traversal safety.
  await pruneNonMediaAndUnsafe(extractRoot);
}

/** Test-only: drive the post-extraction pruning step in isolation. */
export { pruneNonMediaAndUnsafe as _pruneNonMediaAndUnsafe };

async function pruneNonMediaAndUnsafe(extractRoot: string): Promise<void> {
  async function walk(dir: string): Promise<void> {
    let entries: fs.Dirent[];
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      const rel = path.relative(extractRoot, full);
      const escapes = rel.startsWith('..') || path.isAbsolute(rel);
      if (entry.isDirectory()) {
        if (escapes) {
          await fsp.rm(full, { recursive: true, force: true });
          continue;
        }
        await walk(full);
        // Drop empty directories left behind after pruning.
        try {
          const remaining = await fsp.readdir(full);
          if (remaining.length === 0) await fsp.rmdir(full);
        } catch {
          /* ignore */
        }
      } else {
        if (escapes || !isMediaEntry(entry.name)) {
          await fsp.unlink(full).catch(() => undefined);
        }
      }
    }
  }
  await walk(extractRoot);
}

function pickExtractor(
  archivePath: string
): (archive: string, dest: string) => Promise<void> {
  const ext = path.extname(archivePath).toLowerCase();
  if (ext === '.cbr') return extractRarTo;
  return extractZipTo;
}

export async function extractArchive(archivePath: string): Promise<string> {
  const hash = await hashForArchive(archivePath);
  const extractRoot = path.join(cacheRoot, hash);

  try {
    const s = await fsp.stat(extractRoot);
    if (s.isDirectory()) {
      const entries = await fsp.readdir(extractRoot);
      if (entries.length > 0) return extractRoot;
    }
  } catch {
    /* not cached */
  }

  const pending = inFlight.get(hash);
  if (pending) return pending;

  const extractor = pickExtractor(archivePath);

  const p = (async () => {
    try {
      await extractor(archivePath, extractRoot);
      return extractRoot;
    } catch (err) {
      await fsp.rm(extractRoot, { recursive: true, force: true });
      throw err;
    } finally {
      inFlight.delete(hash);
    }
  })();
  inFlight.set(hash, p);
  return p;
}

export async function cleanupArchives(): Promise<void> {
  await fsp.rm(cacheRoot, { recursive: true, force: true });
}
