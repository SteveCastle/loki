import * as path from 'path';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as crypto from 'crypto';
import * as os from 'os';
import yauzl from 'yauzl';
import { Extensions, FileTypes, getFileType } from '../file-types';

let cacheRoot = path.join(os.tmpdir(), 'lowkey-archives');

/** Test-only: override the cache root before calling extractArchive. */
export function _setCacheRoot(dir: string): void {
  cacheRoot = dir;
}

export function isArchivePath(p: string): boolean {
  const ext = path.extname(p).toLowerCase();
  return ext === '.cbz' || ext === '.zip';
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

  try {
    await extractZipTo(archivePath, extractRoot);
  } catch (err) {
    await fsp.rm(extractRoot, { recursive: true, force: true });
    throw err;
  }

  return extractRoot;
}
