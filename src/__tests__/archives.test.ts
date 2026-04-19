/**
 * @jest-environment node
 */
import { isArchivePath } from '../main/archives';
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

      const dirs = fs.existsSync(cacheDir) ? fs.readdirSync(cacheDir) : [];
      expect(dirs.length).toBe(0);
    });

    it('returns cached dir on second call without re-extracting', async () => {
      const { extractArchive, _setCacheRoot } = await import('../main/archives');
      _setCacheRoot(pathMod.join(workDir, 'cache'));

      const zipPath = pathMod.join(workDir, 'book.cbz');
      buildZip([{ name: 'page.jpg', content: TINY_JPG }], zipPath);

      const first = await extractArchive(zipPath);
      const firstMtime = fs.statSync(pathMod.join(first, 'page.jpg')).mtimeMs;

      await new Promise((r) => setTimeout(r, 50));

      const second = await extractArchive(zipPath);
      expect(second).toBe(first);

      const secondMtime = fs.statSync(pathMod.join(second, 'page.jpg')).mtimeMs;
      expect(secondMtime).toBe(firstMtime);
    });
  });

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
});
