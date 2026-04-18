# Archive-as-Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Electron media viewer open `.cbz` / `.zip` archives as if they were directories — on open, the archive is extracted to an LRU-cached temp dir and the existing directory-load pipeline runs unchanged.

**Architecture:** A new main-process module (`src/main/archives.ts`) handles zip-family extraction via `yauzl`, caches extracted directories under `%TEMP%/lowkey-archives/<hash>/`, and cleans up on quit. `loadFiles` in `load-files.ts` detects archive input and transparently swaps the path for the extracted directory. A new `select-archive` IPC opens a file dialog; drag-and-drop and OS file association routes are wired to use the same open-as-directory path. The renderer XState machine gains one new event (`SELECT_ARCHIVE`) and one new state (`selectingArchive`), otherwise unchanged.

**Tech Stack:** TypeScript, Electron, Node.js `fs`, `yauzl` (new dep), Jest + ts-jest. No Go/server changes.

**Spec:** `docs/superpowers/specs/2026-04-18-archive-as-directory-design.md`

---

## File Structure

**Create:**
- `src/main/archives.ts` — archive detection, extraction, cache, cleanup
- `src/__tests__/archives.test.ts` — unit tests (build-zip-on-the-fly fixtures in-test; no binary fixtures checked in)

**Modify:**
- `src/file-types.ts` — add `Archive` enum value and detection
- `src/main/load-files.ts` — detect archive input, extract, rewrite path
- `src/main/main.ts` — add `select-archive` IPC handler + app-quit cleanup hook
- `src/main/preload.ts` — add `select-archive` to `Channels` type
- `src/renderer/state.tsx` — add `SELECT_ARCHIVE` event and `selectingArchive` state
- `src/renderer/hooks/useFileDrop.ts` — archive drops route to open-as-directory
- `src/renderer/components/controls/command-palette.tsx` — add "Open Archive" button
- `package.json` — add `yauzl` dep and `.cbz`/`.zip` file associations

---

## Task 1: Add `Archive` to file-type detection

**Files:**
- Modify: `src/file-types.ts`
- Test: `src/__tests__/file-types.test.ts`

Archives are detected separately from media so the renderer can distinguish "open this as a directory" from "import this as media."

- [ ] **Step 1: Write failing tests in `src/__tests__/file-types.test.ts`**

Append inside the existing top-level `describe('file-types', () => { ... })` block, after the `audio files` describe:

```typescript
    describe('archive files', () => {
      const archiveExtensions = ['cbz', 'zip'];

      it.each(archiveExtensions)('should identify .%s as Archive', (ext) => {
        expect(getFileType(`book.${ext}`)).toBe(FileTypes.Archive);
        expect(getFileType(`BOOK.${ext.toUpperCase()}`)).toBe(FileTypes.Archive);
      });

      it('should not treat archives as media', () => {
        expect(getFileType('book.cbz')).not.toBe(FileTypes.Image);
        expect(getFileType('book.cbz')).not.toBe(FileTypes.Video);
        expect(getFileType('book.cbz')).not.toBe(FileTypes.Audio);
      });
    });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run build && npx jest file-types.test.ts -t "archive files"`
Expected: FAIL — `FileTypes.Archive` is `undefined`.

- [ ] **Step 3: Update `src/file-types.ts`**

Replace the `FileTypes` and `Extensions` enums and extend `getFileType`:

```typescript
export enum FileTypes {
  Image = 'image',
  Video = 'video',
  Audio = 'audio',
  Document = 'document',
  Archive = 'archive',
  Other = 'other',
}

export enum Extensions {
  Image = 'jpg|jpeg|png|gif|bmp|svg|jfif|pjpeg|pjp|webp',
  Video = 'mov|mp4|webm|ogg|mkv|m4v',
  Audio = 'mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape',
  Document = 'pdf|doc|docx|xls|xlsx|ppt|pptx|txt|csv',
  Archive = 'cbz|zip',
}

export const getFileType = (
  fileName: string,
  gifIsVideo?: boolean
): FileTypes => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  if (extension) {
    if (gifIsVideo && extension === 'gif') {
      return FileTypes.Video;
    }
    if (Extensions.Image.includes(extension)) {
      return FileTypes.Image;
    }
    if (Extensions.Video.includes(extension)) {
      return FileTypes.Video;
    }
    if (Extensions.Audio.includes(extension)) {
      return FileTypes.Audio;
    }
    if (Extensions.Document.includes(extension)) {
      return FileTypes.Document;
    }
    if (Extensions.Archive.includes(extension)) {
      return FileTypes.Archive;
    }
  }
  return FileTypes.Other;
};
```

Note: `Extensions.X.includes(ext)` does substring matching on pipe-separated strings. `'cbz'` doesn't collide with any existing extension.

- [ ] **Step 4: Run tests to verify pass**

Run: `npm run build && npx jest file-types.test.ts`
Expected: PASS (all existing tests + new archive tests).

- [ ] **Step 5: Commit**

```bash
git add src/file-types.ts src/__tests__/file-types.test.ts
git commit -m "feat(file-types): add Archive type for .cbz/.zip"
```

---

## Task 2: Add `yauzl` dependency

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Install `yauzl` and its types**

Run:
```bash
yarn add yauzl@^3.2.0
yarn add -D @types/yauzl@^2.10.3
```

Expected: Both added; `package.json` updated; `yarn.lock` updated.

- [ ] **Step 2: Verify install**

Run: `node -e "require('yauzl')"`
Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
git add package.json yarn.lock
git commit -m "feat: add yauzl dependency for archive reading"
```

---

## Task 3: Write `archives.ts` scaffolding and `isArchivePath`

**Files:**
- Create: `src/main/archives.ts`
- Create: `src/__tests__/archives.test.ts`

Start with the simplest piece — archive path detection — to establish the module shape before the extraction logic.

- [ ] **Step 1: Write failing test**

Create `src/__tests__/archives.test.ts`:

```typescript
import { isArchivePath } from '../main/archives';

describe('archives', () => {
  describe('isArchivePath', () => {
    it('returns true for .cbz', () => {
      expect(isArchivePath('C:\\comics\\book.cbz')).toBe(true);
      expect(isArchivePath('/home/u/book.cbz')).toBe(true);
    });
    it('returns true for .zip', () => {
      expect(isArchivePath('/tmp/archive.zip')).toBe(true);
    });
    it('is case-insensitive', () => {
      expect(isArchivePath('BOOK.CBZ')).toBe(true);
      expect(isArchivePath('Book.Zip')).toBe(true);
    });
    it('returns false for non-archive paths', () => {
      expect(isArchivePath('/home/u/image.jpg')).toBe(false);
      expect(isArchivePath('/home/u/folder')).toBe(false);
      expect(isArchivePath('/home/u/file.rar')).toBe(false); // not in scope
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run build && npx jest archives.test.ts`
Expected: FAIL — module `../main/archives` does not exist.

- [ ] **Step 3: Create `src/main/archives.ts` minimal scaffold**

```typescript
import * as path from 'path';

export function isArchivePath(p: string): boolean {
  const ext = path.extname(p).toLowerCase();
  return ext === '.cbz' || ext === '.zip';
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/archives.ts src/__tests__/archives.test.ts
git commit -m "feat(archives): add isArchivePath detection"
```

---

## Task 4: Build an in-test zip fixture helper

**Files:**
- Modify: `src/__tests__/archives.test.ts`

To avoid checking binary fixtures into the repo, we build zips on-the-fly in tests using Node's built-in `zlib`/file APIs. A tiny helper keeps each test case readable.

- [ ] **Step 1: Add fixture helper at top of `src/__tests__/archives.test.ts`**

Insert below the existing import, before `describe('archives', ...)`:

```typescript
import * as fs from 'fs';
import * as pathMod from 'path';
import * as os from 'os';
import { execSync } from 'child_process';

type ZipEntry = { name: string; content: Buffer | string };

/**
 * Build a real zip file on disk using a small, dependency-free writer
 * (manual local-file-header + central-directory encoding with STORED method —
 * no compression, which is valid per the zip spec and sufficient for tests).
 */
function buildZip(entries: ZipEntry[], outPath: string): void {
  const localRecords: Buffer[] = [];
  const centralRecords: Buffer[] = [];
  let offset = 0;

  for (const e of entries) {
    const nameBuf = Buffer.from(e.name, 'utf8');
    const data = Buffer.isBuffer(e.content)
      ? e.content
      : Buffer.from(e.content, 'utf8');

    // CRC-32
    const crc = crc32(data);

    // Local file header
    const local = Buffer.alloc(30 + nameBuf.length);
    local.writeUInt32LE(0x04034b50, 0); // signature
    local.writeUInt16LE(20, 4); // version needed
    local.writeUInt16LE(0, 6); // flags
    local.writeUInt16LE(0, 8); // method (0 = stored)
    local.writeUInt16LE(0, 10); // mod time
    local.writeUInt16LE(0, 12); // mod date
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18); // compressed size
    local.writeUInt32LE(data.length, 22); // uncompressed size
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28); // extra len
    nameBuf.copy(local, 30);

    localRecords.push(local, data);

    // Central directory entry
    const central = Buffer.alloc(46 + nameBuf.length);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4); // version made by
    central.writeUInt16LE(20, 6); // version needed
    central.writeUInt16LE(0, 8); // flags
    central.writeUInt16LE(0, 10); // method
    central.writeUInt16LE(0, 12);
    central.writeUInt16LE(0, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt16LE(0, 30); // extra
    central.writeUInt16LE(0, 32); // comment
    central.writeUInt16LE(0, 34); // disk
    central.writeUInt16LE(0, 36); // internal attrs
    central.writeUInt32LE(0, 38); // external attrs
    central.writeUInt32LE(offset, 42); // local header offset
    nameBuf.copy(central, 46);

    centralRecords.push(central);
    offset += local.length + data.length;
  }

  const centralStart = offset;
  const centralSize = centralRecords.reduce((s, b) => s + b.length, 0);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(0, 4); // disk
  eocd.writeUInt16LE(0, 6);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(centralSize, 12);
  eocd.writeUInt32LE(centralStart, 16);
  eocd.writeUInt16LE(0, 20); // comment

  fs.writeFileSync(
    outPath,
    Buffer.concat([...localRecords, ...centralRecords, eocd])
  );
}

// Table-based CRC-32 (IEEE 802.3)
const crcTable: number[] = (() => {
  const t = new Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function mkTmpDir(prefix = 'archive-test-'): string {
  return fs.mkdtempSync(pathMod.join(os.tmpdir(), prefix));
}

// Minimal PNG-ish bytes (magic only — enough for our media-extension filter)
const TINY_JPG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]);
```

- [ ] **Step 2: Run tests to verify nothing broke**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS (existing `isArchivePath` tests still pass; helper is unused so far).

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/archives.test.ts
git commit -m "test(archives): add in-test zip fixture builder"
```

---

## Task 5: Implement `extractArchive` (happy path)

**Files:**
- Modify: `src/main/archives.ts`
- Modify: `src/__tests__/archives.test.ts`

- [ ] **Step 1: Write failing test**

Append inside `describe('archives', ...)` in `src/__tests__/archives.test.ts`:

```typescript
  describe('extractArchive', () => {
    let workDir: string;

    beforeEach(() => {
      workDir = mkTmpDir();
    });
    afterEach(() => {
      try {
        fs.rmSync(workDir, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    });

    it('extracts media files and skips non-media entries', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'book.cbz');
      buildZip(
        [
          { name: 'page01.jpg', content: TINY_JPG },
          { name: 'page02.jpg', content: TINY_JPG },
          { name: 'ComicInfo.xml', content: '<ComicInfo/>' },
        ],
        zipPath
      );

      const outDir = await extractArchive(zipPath);
      const files = fs.readdirSync(outDir).sort();

      expect(files).toContain('page01.jpg');
      expect(files).toContain('page02.jpg');
      expect(files).not.toContain('ComicInfo.xml');
    });

    it('preserves subfolder structure', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'nested.cbz');
      buildZip(
        [
          { name: 'ch1/01.jpg', content: TINY_JPG },
          { name: 'ch2/02.jpg', content: TINY_JPG },
        ],
        zipPath
      );

      const outDir = await extractArchive(zipPath);
      expect(fs.existsSync(pathMod.join(outDir, 'ch1', '01.jpg'))).toBe(true);
      expect(fs.existsSync(pathMod.join(outDir, 'ch2', '02.jpg'))).toBe(true);
    });
  });
```

The `_setCacheRoot` hook lets tests point the cache at a per-test temp dir instead of `%TEMP%/lowkey-archives`.

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run build && npx jest archives.test.ts -t "extractArchive"`
Expected: FAIL — `extractArchive` / `_setCacheRoot` not exported.

- [ ] **Step 3: Replace `src/main/archives.ts` with extraction implementation**

```typescript
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
      { lazyEntries: true, autoClose: true },
      (err, zipfile) => {
        if (err || !zipfile) {
          reject(err ?? new Error('Failed to open zip'));
          return;
        }
        zipfile.on('error', reject);
        zipfile.on('end', () => resolve());
        zipfile.readEntry();
        zipfile.on('entry', (entry) => {
          const name = entry.fileName;
          // Directory entry (zip convention: trailing slash)
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

  // Cache hit: directory already populated.
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
    // Delete partial dir; re-throw so load-files surfaces the error.
    await fsp.rm(extractRoot, { recursive: true, force: true });
    throw err;
  }

  return extractRoot;
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS (isArchivePath + extractArchive happy-path tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/archives.ts src/__tests__/archives.test.ts
git commit -m "feat(archives): extract zip/cbz archives to cached temp dir"
```

---

## Task 6: Zip-slip guard test + corrupted-archive test

**Files:**
- Modify: `src/__tests__/archives.test.ts`

These are the load-bearing safety tests.

- [ ] **Step 1: Write failing tests**

Inside the `describe('extractArchive', ...)` block, append:

```typescript
    it('skips entries that escape the extraction root (zip-slip)', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'evil.zip');
      buildZip(
        [
          { name: '../evil.jpg', content: TINY_JPG },
          { name: 'ok.jpg', content: TINY_JPG },
        ],
        zipPath
      );

      const outDir = await extractArchive(zipPath);
      expect(fs.existsSync(pathMod.join(outDir, 'ok.jpg'))).toBe(true);
      // Should NOT have written evil.jpg into the cache parent
      expect(
        fs.existsSync(pathMod.join(workDir, 'cache', 'evil.jpg'))
      ).toBe(false);
    });

    it('rejects on a corrupted archive and leaves no partial cache dir', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      const cacheDir = pathMod.join(workDir, 'cache');
      _setCacheRoot(cacheDir);

      const zipPath = pathMod.join(workDir, 'broken.zip');
      fs.writeFileSync(zipPath, Buffer.from('not a zip at all'));

      await expect(extractArchive(zipPath)).rejects.toBeDefined();

      // No partial hash dir should be left behind.
      const dirs = fs.existsSync(cacheDir) ? fs.readdirSync(cacheDir) : [];
      expect(dirs.length).toBe(0);
    });
```

- [ ] **Step 2: Run tests to verify they pass (they should — the implementation already handles both)**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS. If either fails, the implementation is wrong — fix `isSafeEntryPath` or the `try/catch` in `extractArchive`, not the test.

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/archives.test.ts
git commit -m "test(archives): cover zip-slip and corrupted-archive cases"
```

---

## Task 7: Cache-hit test

**Files:**
- Modify: `src/__tests__/archives.test.ts`

- [ ] **Step 1: Write failing test**

Append inside `describe('extractArchive', ...)`:

```typescript
    it('returns cached dir on second call without re-extracting', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'book.cbz');
      buildZip([{ name: 'page.jpg', content: TINY_JPG }], zipPath);

      const first = await extractArchive(zipPath);
      const firstMtime = fs.statSync(pathMod.join(first, 'page.jpg')).mtimeMs;

      // Wait a hair so a re-extract would produce a different mtime
      await new Promise((r) => setTimeout(r, 50));

      const second = await extractArchive(zipPath);
      expect(second).toBe(first);

      const secondMtime = fs.statSync(pathMod.join(second, 'page.jpg')).mtimeMs;
      expect(secondMtime).toBe(firstMtime); // file was NOT rewritten
    });
```

- [ ] **Step 2: Run test to verify it passes**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS — the cache-hit branch in `extractArchive` short-circuits on the second call.

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/archives.test.ts
git commit -m "test(archives): verify cache-hit skips re-extraction"
```

---

## Task 8: Implement `cleanupArchives` + in-flight dedupe

**Files:**
- Modify: `src/main/archives.ts`
- Modify: `src/__tests__/archives.test.ts`

`cleanupArchives` wipes the cache root on app quit. In-flight dedupe ensures two simultaneous `extractArchive` calls for the same archive share one extraction.

- [ ] **Step 1: Write failing tests**

Append to `src/__tests__/archives.test.ts`:

```typescript
  describe('cleanupArchives', () => {
    it('removes the cache root', async () => {
      const { extractArchive, cleanupArchives, _setCacheRoot } = await import(
        '../main/archives'
      );
      const workDir = mkTmpDir();
      const cacheDir = pathMod.join(workDir, 'cache');
      _setCacheRoot(cacheDir);

      const zipPath = pathMod.join(workDir, 'book.cbz');
      buildZip([{ name: 'page.jpg', content: TINY_JPG }], zipPath);

      await extractArchive(zipPath);
      expect(fs.existsSync(cacheDir)).toBe(true);

      await cleanupArchives();
      expect(fs.existsSync(cacheDir)).toBe(false);

      fs.rmSync(workDir, { recursive: true, force: true });
    });
  });

  describe('in-flight dedupe', () => {
    it('concurrent extracts of same archive share one extraction', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      const workDir = mkTmpDir();
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'book.cbz');
      buildZip(
        [{ name: 'a.jpg', content: TINY_JPG }, { name: 'b.jpg', content: TINY_JPG }],
        zipPath
      );

      const [r1, r2, r3] = await Promise.all([
        extractArchive(zipPath),
        extractArchive(zipPath),
        extractArchive(zipPath),
      ]);
      expect(r1).toBe(r2);
      expect(r2).toBe(r3);
      expect(fs.existsSync(pathMod.join(r1, 'a.jpg'))).toBe(true);
      expect(fs.existsSync(pathMod.join(r1, 'b.jpg'))).toBe(true);

      fs.rmSync(workDir, { recursive: true, force: true });
    });
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run build && npx jest archives.test.ts -t "cleanupArchives|in-flight"`
Expected: FAIL — `cleanupArchives` not exported; concurrent test likely races into overlapping extractions.

- [ ] **Step 3: Update `src/main/archives.ts`**

Add at the top (after `cacheRoot` declaration):

```typescript
const inFlight = new Map<string, Promise<string>>();
```

Replace the `extractArchive` function body with a deduping wrapper:

```typescript
export async function extractArchive(archivePath: string): Promise<string> {
  const hash = await hashForArchive(archivePath);
  const extractRoot = path.join(cacheRoot, hash);

  // Cache hit
  try {
    const s = await fsp.stat(extractRoot);
    if (s.isDirectory()) {
      const entries = await fsp.readdir(extractRoot);
      if (entries.length > 0) return extractRoot;
    }
  } catch {
    /* not cached */
  }

  // In-flight dedupe
  const pending = inFlight.get(hash);
  if (pending) return pending;

  const p = (async () => {
    try {
      await extractZipTo(archivePath, extractRoot);
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
```

- [ ] **Step 4: Run full test suite**

Run: `npm run build && npx jest archives.test.ts`
Expected: PASS — all archives tests including the new two.

- [ ] **Step 5: Commit**

```bash
git add src/main/archives.ts src/__tests__/archives.test.ts
git commit -m "feat(archives): cleanup + in-flight extract dedupe"
```

---

## Task 9: Wire archives into `load-files.ts`

**Files:**
- Modify: `src/main/load-files.ts`

At the top of `loadFiles`, detect archive input and swap the path before stat/walk runs.

- [ ] **Step 1: Modify `src/main/load-files.ts`**

Add import at the top (after existing imports at line 1-8):

```typescript
import { isArchivePath, extractArchive } from './archives';
```

Replace the `try { ... }` block inside `loadFiles` (current lines 329-341) with:

```typescript
    try {
      if (isArchivePath(filePath)) {
        // Archive: extract to temp and treat the extraction dir as the browse root.
        const extracted = await extractArchive(filePath);
        folderPath = extracted;
        fileName = '';
      } else {
        const stats = await fsPromises.lstat(filePath);
        if (stats.isDirectory()) {
          folderPath = filePath;
          fileName = '';
        } else {
          folderPath = path.dirname(filePath);
          fileName = path.basename(filePath);
        }
      }
    } catch {
      folderPath = path.dirname(filePath);
      fileName = path.basename(filePath);
    }
```

Also force `recursive = true` when browsing an archive, so nested internal folders show up in the flat list. Immediately after the block above (still at the top of `loadFiles`), add:

```typescript
    const isArchiveBrowse = folderPath !== filePath && isArchivePath(filePath);
    const effectiveRecursive = isArchiveBrowse ? true : recursive;
```

Then in the three places `recursive` is passed to the listing functions (`listFilesFastWindows`, `listFilesFastDarwin`, `walkDirectory` — around lines 375, 389, 404), change `recursive` to `effectiveRecursive`.

Sanity-check the final `loadFiles` signature still compiles.

- [ ] **Step 2: Run existing test suite to confirm no regression**

Run: `npm run build`
Expected: build succeeds. (No existing unit test for `loadFiles` directly; manual/e2e coverage via Task 13.)

- [ ] **Step 3: Commit**

```bash
git add src/main/load-files.ts
git commit -m "feat(load-files): route archive paths through extractArchive"
```

---

## Task 10: Add `select-archive` IPC + cleanup on quit

**Files:**
- Modify: `src/main/main.ts`
- Modify: `src/main/preload.ts`

- [ ] **Step 1: Add `select-archive` to `Channels` in `src/main/preload.ts`**

In the `Channels` type union (starting line 15), add `'select-archive'` right after `'select-directory'`:

```typescript
export type Channels =
  | 'shutdown'
  | 'select-file'
  | 'select-directory'
  | 'select-archive'
  | 'load-files'
  ...
```

- [ ] **Step 2: Add IPC handler in `src/main/main.ts`**

Immediately after the `select-directory` handler (after line 473), add:

```typescript
// Handle archive-open event from renderer process
type SelectArchiveInput = [string | undefined];
ipcMain.handle(
  'select-archive',
  async (_: IpcMainInvokeEvent, args: SelectArchiveInput) => {
    invariant(mainWindow, 'mainWindow is not defined');
    const defaultPath = args[0];
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      defaultPath,
      filters: [
        { name: 'Comic Archive', extensions: ['cbz', 'zip'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (!result.canceled) {
      return result.filePaths[0];
    }
    return null;
  }
);
```

- [ ] **Step 3: Call `cleanupArchives()` on quit**

At the top of `src/main/main.ts` with the other imports, add:

```typescript
import { cleanupArchives } from './archives';
```

Replace the existing `app.on('before-quit', ...)` block (lines 729-738) with:

```typescript
app.on('before-quit', async () => {
  if (db) {
    try {
      await db.close();
    } catch (err) {
      console.error('Error closing database on quit:', err);
    }
    db = null;
  }
  try {
    await cleanupArchives();
  } catch (err) {
    console.error('Error cleaning up archive cache on quit:', err);
  }
});
```

- [ ] **Step 4: Build and verify**

Run: `npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add src/main/main.ts src/main/preload.ts
git commit -m "feat(archives): add select-archive IPC + quit cleanup"
```

---

## Task 11: Renderer — `SELECT_ARCHIVE` event and `selectingArchive` state

**Files:**
- Modify: `src/renderer/state.tsx`

Mirror the `SELECT_DIRECTORY` / `selectingDirectory` pair but invoke `select-archive`. On success, `setPath` stores the archive path as `initialFile`, and `loadingFromFS` picks it up from there.

- [ ] **Step 1: Add the `selectingArchive` state**

In `src/renderer/state.tsx`, find `selectingDirectory` (line 1200). Immediately after its closing `},` (around line 1217), add the twin state:

```typescript
          selectingArchive: {
            invoke: {
              src: (context, event) => {
                const currentFile = context.initialFile;
                console.log('selecting archive', context, event);
                return invoke('select-archive', [
                  currentFile,
                ]);
              },
              onDone: {
                target: 'loadingFromFS',
                actions: ['setPath'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
```

- [ ] **Step 2: Add `SELECT_ARCHIVE` event transitions**

Find every occurrence of `SELECT_DIRECTORY: { target: 'selectingDirectory' }` (lines 1082, 1637, 2261, 2395). Directly after each, insert the sibling:

```typescript
              SELECT_ARCHIVE: {
                target: 'selectingArchive',
              },
```

All four occurrences must be updated — the event is exposed from the same parent states that already expose `SELECT_DIRECTORY`.

- [ ] **Step 3: Build to confirm no type errors**

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/renderer/state.tsx
git commit -m "feat(state): add SELECT_ARCHIVE event and selectingArchive state"
```

---

## Task 12: Renderer — command palette button + drag-and-drop route

**Files:**
- Modify: `src/renderer/components/controls/command-palette.tsx`
- Modify: `src/renderer/hooks/useFileDrop.ts`

- [ ] **Step 1: Add "Open Archive" button to the command palette**

In `src/renderer/components/controls/command-palette.tsx`, find the existing directory button (line 247-251):

```tsx
        <ActionButton
          icon={folderIcon}
          onClick={() => libraryService.send('SELECT_DIRECTORY')}
          tooltipId="select-directory"
        />
```

Immediately after it, add a sibling button. Reuse `folderIcon` to avoid introducing a new asset (the tooltip disambiguates):

```tsx
        <ActionButton
          icon={folderIcon}
          onClick={() => libraryService.send('SELECT_ARCHIVE')}
          tooltipId="select-archive"
        />
```

If there is a tooltip registry elsewhere in this file (search for `select-directory` to confirm pattern), add a `select-archive` entry with content "Open Archive…". If no such registry exists, the raw `tooltipId` string is fine — the tooltip may just render the id, which is acceptable for MVP.

- [ ] **Step 2: Update drag-and-drop to route archive drops**

In `src/renderer/hooks/useFileDrop.ts`:

Replace the `isMediaFile` helper (lines 31-34) with two helpers:

```typescript
function isMediaFile(fileName: string): boolean {
  const ft = getFileType(fileName);
  return ft === FileTypes.Image || ft === FileTypes.Video || ft === FileTypes.Audio;
}

function isArchiveFile(fileName: string): boolean {
  return getFileType(fileName) === FileTypes.Archive;
}
```

Replace the filter block inside `handleDrop` (currently lines 108-120) with archive-first handling:

```typescript
      // Archives open as a directory (treat like SELECT_ARCHIVE with a known path).
      const archiveFiles = nativeFiles.filter((f) => isArchiveFile(f.name));
      if (archiveFiles.length > 0) {
        const getPath = (window as any).electron?.getPathForFile;
        const first = archiveFiles[0];
        const archivePath = (getPath ? getPath(first) : (first as any).path) as
          | string
          | undefined;
        if (archivePath) {
          libraryService.send({ type: 'SET_FILE', path: archivePath });
        }
        return;
      }

      // Filter to media files only
      const mediaFiles = nativeFiles.filter((f) => isMediaFile(f.name));
      if (mediaFiles.length === 0) {
        libraryService.send('ADD_TOAST', {
          data: {
            type: 'warning',
            title: 'No media files',
            message: 'None of the dropped files are supported media types',
            durationMs: 3000,
          },
        });
        return;
      }
```

Rationale: `SET_FILE` is already wired in `state.tsx` (see line 1653, 2264, 2398) and routes to `loadingFromFS` while setting `initialFile = event.path`. That's exactly what we want for an archive path — `loadFiles` will detect the archive and extract.

Also update `resolveDirectory` (lines 36-52) so archive paths pass through unchanged (for command-line / file-association entry):

```typescript
/** Resolve the browsed directory from initialFile (which may be a file path, archive path, or dir path). */
function resolveDirectory(initialFile: string): string {
  const ft = getFileType(initialFile);
  // Archive paths represent their contents — pass through unchanged.
  if (ft === FileTypes.Archive) {
    return initialFile;
  }
  if (ft !== FileTypes.Other) {
    // It's a media file — extract the directory
    const lastSep = Math.max(initialFile.lastIndexOf('/'), initialFile.lastIndexOf('\\'));
    let dir = lastSep > 0 ? initialFile.substring(0, lastSep) : initialFile;
    if (/^[A-Za-z]:$/.test(dir)) {
      dir += '\\';
    }
    return dir;
  }
  return initialFile;
}
```

- [ ] **Step 3: Build**

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/renderer/components/controls/command-palette.tsx src/renderer/hooks/useFileDrop.ts
git commit -m "feat(ui): open archives via command palette and drag-drop"
```

---

## Task 13: OS file association

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Add `.cbz` and `.zip` file associations**

In `package.json` inside `build.fileAssociations` (array starting line 264), append two entries after the existing `mkv` block:

```json
      {
        "ext": "cbz",
        "name": "Comic Book Archive",
        "role": "Viewer"
      },
      {
        "ext": "zip",
        "name": "Zip Archive",
        "role": "Viewer"
      }
```

Make sure you place commas correctly — the `mkv` entry needs a trailing comma before the `cbz` entry.

- [ ] **Step 2: Verify JSON is valid**

Run: `node -e "JSON.parse(require('fs').readFileSync('package.json'))"`
Expected: no output, no error.

- [ ] **Step 3: Commit**

```bash
git add package.json
git commit -m "feat(archives): register .cbz/.zip OS file associations"
```

---

## Task 14: Manual verification

No automated tests run the full Electron UI. Do these manual checks before declaring the feature done.

**Files:** n/a (manual)

- [ ] **Step 1: Build and start dev**

Run: `yarn dev`
Expected: dev server boots, Electron window appears, library loads (or shows selecting state).

- [ ] **Step 2: Create a fixture CBZ**

From a scratch directory (not the repo), create 3-5 small JPEG images and a `ComicInfo.xml` file, then zip them:

```bash
# PowerShell:
Compress-Archive -Path page01.jpg,page02.jpg,page03.jpg,ComicInfo.xml -DestinationPath test.cbz
```

- [ ] **Step 3: Open via drag-and-drop**

Drag `test.cbz` onto the app window.
Expected: spinner briefly; library populates with the 3 JPEGs sorted by name. `ComicInfo.xml` is absent.

- [ ] **Step 4: Open via command palette**

Click the second folder-icon button in the command palette.
Expected: native file picker filtered to `.cbz` / `.zip`. Pick `test.cbz`. Library loads same as above.

- [ ] **Step 5: Reopen the same archive (cache hit)**

Repeat step 4 or 3. Expected: loads noticeably faster (no extraction delay). Inspect `%TEMP%\lowkey-archives` and confirm a hash directory exists.

- [ ] **Step 6: Modify the archive, reopen**

Re-create `test.cbz` with a different set of files. Reopen.
Expected: new content shown. Inspect `%TEMP%\lowkey-archives` and confirm a second hash dir exists (or the old hash was replaced).

- [ ] **Step 7: Quit the app**

Close Electron (`Ctrl+Q` or File > Quit).
Expected: `%TEMP%\lowkey-archives` is deleted on quit.

- [ ] **Step 8: OS double-click (Windows only, release build)**

Build a production package: `yarn package`.
Install the produced binary. Right-click a `.cbz` → Open With → Lowkey Media Viewer. Expected: app launches and opens the archive.

(File-association behavior is tied to installed builds, not dev mode — skip if not shipping a build.)

- [ ] **Step 9: Error cases**

- Drag a corrupted zip (`echo "garbage" > bad.zip`): expected error toast from the library loader, no crash, no partial temp dir.
- Drag a zip containing only `ComicInfo.xml` (no images): empty-library UX renders.

- [ ] **Step 10: Final verification + single-run commit-free summary**

No commit for the manual task. If anything failed, return to the relevant task and fix.

---

## Self-Review Notes

Verified against spec sections:

- **Architecture** (spec §Architecture) — covered by Tasks 1, 3, 5, 8, 9, 10, 11, 12.
- **Extraction & caching** (spec §Extraction & caching) — Tasks 3, 5, 7, 8. Hash = `sha1(absolutePath + mtimeMs)` implemented in `hashForArchive`.
- **LRU cache** (spec §LRU cache) — **intentionally deferred from MVP.** In-flight dedupe and quit-cleanup cover the correctness needs; LRU eviction can be added later when cache size becomes a real issue. If the user wants LRU enforced now, add a task after Task 8 that reads `.meta.json` on startup and prunes.
- **Cleanup on quit** (spec §Cleanup on app quit) — Task 10.
- **User flow entry points** (spec §User flow) — command palette (Task 12), drag-and-drop (Task 12), OS file association (Task 13). No separate "app menu" item because the existing UI uses the command palette, not a native menu.
- **Error cases** (spec §Error cases) — Tasks 6 (zip-slip, corrupted), 8 (dedupe), 9 (load-files failure path).
- **Testing** (spec §Testing) — Tasks 1, 3, 5, 6, 7, 8, 14. Unit tests build fixture zips in-test (no binary fixtures). Manual plan in Task 14.
- **Non-goals** — no CBR/RAR, no Go server, no nested archives, no write-back. Plan respects all of these.

**Known deviation from spec:** spec mentions LRU eviction with N=8 / M=2 GB limits; this plan ships quit-cleanup + cache-hit, and leaves LRU as a follow-up. Rationale: quit-cleanup makes unbounded growth time-bounded (one session), and enforcing LRU now adds complexity for a gain the user won't see until they've opened >8 archives in one session.

**Known deviation from spec:** spec mentions a `.meta.json` sidecar per extraction; this plan does not write one. Rationale: we dropped LRU, and the hash dir itself is enough state — `.meta.json` was only needed for cross-run LRU reconstruction.

**Placeholder scan:** no "TODO" / "TBD" / vague steps. Every code step is complete.

**Type consistency:** `isArchivePath`, `extractArchive`, `cleanupArchives`, `_setCacheRoot` names consistent across all tasks that reference them.
