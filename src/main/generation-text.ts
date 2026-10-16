/*
 * Embedded generation text — the metadata AI image tools leave inside the
 * files they write. This module only EXTRACTS it: every text record a file
 * carries comes back as a keyword → text map and the renderer decides what
 * it means (ComfyUI `prompt` / `workflow` JSON, a WebUI `parameters` block,
 * and so on). Keeping the backend dumb is what lets the Go server return
 * the identical shape and the renderer parse both the same way.
 *
 * Where the text lives, by container:
 *   PNG   tEXt / zTXt / iTXt chunks (ComfyUI: prompt, workflow; WebUI:
 *         parameters; NovelAI: Comment …)
 *   WebP  the RIFF `EXIF` chunk → a TIFF directory. ComfyUI stores
 *         "prompt:{…}" in Model (0x0110) and "workflow:{…}" in Make
 *         (0x010f); WebUI stores parameters in UserComment (0x9286)
 *   JPEG  APP1 Exif (same TIFF layout) and COM comment segments
 *   video (mp4 / webm / mkv / mov) container tags via ffprobe — the
 *         VideoHelperSuite nodes put {"prompt":…,"workflow":…} in `comment`
 */
import fs from 'fs';
import zlib from 'zlib';
import { exec } from 'child_process';
import { promisify } from 'util';

const execPromisified = promisify(exec);

export type TextChunks = Record<string, string>;

const MAX_TEXT = 64 * 1024 * 1024;   // a workflow JSON is big, but not this big
const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv|avi)$/i;

/** Read `len` bytes at `pos` from an open handle (short reads are fine). */
async function readAt(fh: fs.promises.FileHandle, pos: number, len: number): Promise<Buffer> {
  const buf = Buffer.alloc(len);
  const { bytesRead } = await fh.read(buf, 0, len, pos);
  return bytesRead === len ? buf : buf.subarray(0, bytesRead);
}

/* ---- PNG ------------------------------------------------------------- */

const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

async function readPngText(fh: fs.promises.FileHandle, size: number): Promise<TextChunks> {
  const out: TextChunks = {};
  let pos = PNG_SIG.length;
  while (pos + 8 <= size) {
    const head = await readAt(fh, pos, 8);
    if (head.length < 8) break;
    const len = head.readUInt32BE(0);
    const type = head.toString('latin1', 4, 8);
    const dataPos = pos + 8;
    if (type === 'IEND') break;
    if ((type === 'tEXt' || type === 'zTXt' || type === 'iTXt') && len <= MAX_TEXT) {
      const data = await readAt(fh, dataPos, len);
      const entry = decodePngTextChunk(type, data);
      if (entry && entry.text) out[entry.keyword] = entry.text;
    }
    pos = dataPos + len + 4;   // data + CRC — IDAT and friends are skipped, not read
  }
  return out;
}

function decodePngTextChunk(type: string, data: Buffer): { keyword: string; text: string } | null {
  const nul = data.indexOf(0);
  if (nul < 0) return null;
  const keyword = data.toString('latin1', 0, nul);
  try {
    if (type === 'tEXt') return { keyword, text: data.toString('latin1', nul + 1) };
    if (type === 'zTXt') {
      // keyword \0 compression-method(1) compressed-text
      return { keyword, text: zlib.inflateSync(data.subarray(nul + 2)).toString('latin1') };
    }
    // iTXt: keyword \0 compression-flag(1) compression-method(1) language \0 translated-keyword \0 text
    const compressed = data[nul + 1] === 1;
    let p = nul + 3;
    p = data.indexOf(0, p) + 1;           // skip language tag
    p = data.indexOf(0, p) + 1;           // skip translated keyword
    const body = data.subarray(p);
    const text = compressed ? zlib.inflateSync(body).toString('utf8') : body.toString('utf8');
    return { keyword, text };
  } catch {
    return null;
  }
}

/* ---- TIFF / Exif ------------------------------------------------------ */

const TAG_IMAGE_DESCRIPTION = 0x010e;
const TAG_MAKE = 0x010f;
const TAG_MODEL = 0x0110;
const TAG_EXIF_IFD = 0x8769;
const TAG_USER_COMMENT = 0x9286;
const TYPE_SIZE: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };

/** The text-bearing tags of a TIFF blob (IFD0 + the Exif sub-IFD), keyed
 * by a readable name. Tolerates truncated or malformed directories. */
export function readTiffText(tiff: Buffer): TextChunks {
  const out: TextChunks = {};
  if (tiff.length < 8) return out;
  const le = tiff[0] === 0x49 && tiff[1] === 0x49;
  if (!le && !(tiff[0] === 0x4d && tiff[1] === 0x4d)) return out;
  const u16 = (p: number) => (le ? tiff.readUInt16LE(p) : tiff.readUInt16BE(p));
  const u32 = (p: number) => (le ? tiff.readUInt32LE(p) : tiff.readUInt32BE(p));
  const visited = new Set<number>();

  const walk = (ifd: number) => {
    if (ifd < 8 || ifd + 2 > tiff.length || visited.has(ifd)) return;
    visited.add(ifd);
    const n = u16(ifd);
    for (let i = 0; i < n; i++) {
      const e = ifd + 2 + i * 12;
      if (e + 12 > tiff.length) return;
      const tag = u16(e);
      const type = u16(e + 2);
      const count = u32(e + 4);
      const unit = TYPE_SIZE[type] ?? 1;
      const byteLen = count * unit;
      const valuePos = byteLen <= 4 ? e + 8 : u32(e + 8);
      if (tag === TAG_EXIF_IFD) {
        walk(u32(e + 8));
        continue;
      }
      if (valuePos + byteLen > tiff.length) continue;
      const raw = tiff.subarray(valuePos, valuePos + byteLen);
      if (tag === TAG_USER_COMMENT) {
        const text = decodeUserComment(raw, le);
        if (text) out.UserComment = text;
      } else if (tag === TAG_IMAGE_DESCRIPTION || tag === TAG_MAKE || tag === TAG_MODEL) {
        const text = raw.toString('utf8').replace(/\0+$/, '');
        if (text) out[tag === TAG_IMAGE_DESCRIPTION ? 'ImageDescription' : tag === TAG_MAKE ? 'Make' : 'Model'] = text;
      }
    }
  };
  walk(u32(4));
  return out;
}

/** UserComment opens with an 8-byte character code. */
function decodeUserComment(raw: Buffer, le: boolean): string {
  const code = raw.toString('latin1', 0, 8);
  const body = raw.subarray(8);
  let text: string;
  if (code.startsWith('UNICODE')) {
    // The spec says platform order; WebUI (Pillow) writes big-endian
    // regardless, so sniff: real text has zero high bytes in BE.
    const be = body.length >= 2 && body[0] === 0 && body[1] !== 0;
    text = be || !le ? swap16(body).toString('utf16le') : body.toString('utf16le');
  } else if (code.startsWith('ASCII') || code.startsWith('JIS') || /^\0+$/.test(code)) {
    text = body.toString('utf8');
  } else {
    text = raw.toString('utf8');
  }
  return text.replace(/\0+$/, '');
}

function swap16(b: Buffer): Buffer {
  const out = Buffer.from(b.subarray(0, b.length - (b.length % 2)));
  out.swap16();
  return out;
}

/* ---- WebP -------------------------------------------------------------- */

async function readWebpText(fh: fs.promises.FileHandle, size: number): Promise<TextChunks> {
  const out: TextChunks = {};
  let pos = 12;   // RIFF size WEBP
  while (pos + 8 <= size) {
    const head = await readAt(fh, pos, 8);
    if (head.length < 8) break;
    const fourcc = head.toString('latin1', 0, 4);
    const len = head.readUInt32LE(4);
    if (fourcc === 'EXIF' && len <= MAX_TEXT) {
      let data = await readAt(fh, pos + 8, len);
      if (data.toString('latin1', 0, 6) === 'Exif\0\0') data = data.subarray(6);
      Object.assign(out, readTiffText(data));
    } else if (fourcc === 'XMP ' && len <= MAX_TEXT) {
      const xmp = (await readAt(fh, pos + 8, len)).toString('utf8');
      const desc = /<dc:description>[\s\S]*?<rdf:li[^>]*>([\s\S]*?)<\/rdf:li>/.exec(xmp)?.[1];
      if (desc) out.XMPDescription = desc;
    }
    pos += 8 + len + (len & 1);
  }
  return out;
}

/* ---- JPEG -------------------------------------------------------------- */

async function readJpegText(fh: fs.promises.FileHandle, size: number): Promise<TextChunks> {
  const out: TextChunks = {};
  let pos = 2;
  const comments: string[] = [];
  while (pos + 4 <= size) {
    const head = await readAt(fh, pos, 4);
    if (head.length < 4 || head[0] !== 0xff) break;
    const marker = head[1];
    if (marker === 0xd9 || marker === 0xda) break;   // EOI / start of scan: no more headers
    if (marker === 0xff) { pos += 1; continue; }
    const len = head.readUInt16BE(2);
    if (len < 2) break;
    const body = len - 2;
    if (marker === 0xe1 && body <= MAX_TEXT) {
      const data = await readAt(fh, pos + 4, body);
      if (data.toString('latin1', 0, 6) === 'Exif\0\0') Object.assign(out, readTiffText(data.subarray(6)));
    } else if (marker === 0xfe && body <= MAX_TEXT) {
      const text = (await readAt(fh, pos + 4, body)).toString('utf8').replace(/\0+$/, '');
      if (text) comments.push(text);
    }
    pos += 2 + len;
  }
  if (comments.length) out.Comment = comments.join('\n');
  return out;
}

/* ---- video containers -------------------------------------------------- */

async function readVideoText(filePath: string, ffprobePath: string): Promise<TextChunks> {
  const out: TextChunks = {};
  try {
    const { stdout } = await execPromisified(
      `"${ffprobePath}" -v error -show_entries format_tags -of json "${filePath}"`,
      { maxBuffer: MAX_TEXT }
    );
    const tags = JSON.parse(stdout)?.format?.tags ?? {};
    for (const [key, value] of Object.entries(tags)) {
      if (typeof value === 'string' && value) out[key] = value;
    }
  } catch {
    // no ffprobe, or not a container it understands — nothing to show
  }
  return out;
}

/* ---- entry point ------------------------------------------------------- */

/**
 * Every embedded text record in the file, or undefined when it has none.
 * Never throws: a file we can't open or don't understand just has no
 * generation text.
 */
export async function readGenerationText(
  filePath: string,
  ffprobePath?: string
): Promise<TextChunks | undefined> {
  let chunks: TextChunks = {};
  if (VIDEO_EXT.test(filePath)) {
    if (ffprobePath) chunks = await readVideoText(filePath, ffprobePath);
  } else {
    let fh: fs.promises.FileHandle | null = null;
    try {
      fh = await fs.promises.open(filePath, 'r');
      const { size } = await fh.stat();
      const head = await readAt(fh, 0, 12);
      if (head.subarray(0, 8).equals(PNG_SIG)) chunks = await readPngText(fh, size);
      else if (head.toString('latin1', 0, 4) === 'RIFF' && head.toString('latin1', 8, 12) === 'WEBP') chunks = await readWebpText(fh, size);
      else if (head[0] === 0xff && head[1] === 0xd8) chunks = await readJpegText(fh, size);
    } catch {
      return undefined;
    } finally {
      await fh?.close().catch(() => undefined);
    }
  }
  return Object.keys(chunks).length ? chunks : undefined;
}
