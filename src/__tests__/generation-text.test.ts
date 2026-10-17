import fs from 'fs';
import os from 'os';
import path from 'path';
import zlib from 'zlib';
import { readGenerationText, readTiffText } from '../main/generation-text';

/* Synthetic files built byte by byte, so the walkers are tested against the
 * exact container layouts rather than against whatever a library emits. */

const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'latin1'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}
function png(chunks: Buffer[]): Buffer {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(1, 0);
  ihdr.writeUInt32BE(1, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr),
    ...chunks,
    pngChunk('IDAT', zlib.deflateSync(Buffer.alloc(5))),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

/** A little-endian TIFF with IFD0 {Make, Model, ExifIFD} → Exif {UserComment}. */
function tiff(fields: { make?: string; model?: string; userComment?: Buffer }): Buffer {
  const entries0: Array<[number, number, Buffer]> = [];
  if (fields.make) entries0.push([0x010f, 2, Buffer.from(fields.make + '\0', 'utf8')]);
  if (fields.model) entries0.push([0x0110, 2, Buffer.from(fields.model + '\0', 'utf8')]);
  const exifEntries: Array<[number, number, Buffer]> = fields.userComment ? [[0x9286, 7, fields.userComment]] : [];
  const ifd0Pos = 8;
  const ifd0Len = 2 + 12 * (entries0.length + (exifEntries.length ? 1 : 0)) + 4;
  const exifPos = ifd0Pos + ifd0Len;
  const exifLen = exifEntries.length ? 2 + 12 * exifEntries.length + 4 : 0;
  let dataPos = exifPos + exifLen;
  const data: Buffer[] = [];
  const entry = (tag: number, type: number, value: Buffer) => {
    const e = Buffer.alloc(12);
    e.writeUInt16LE(tag, 0);
    e.writeUInt16LE(type, 2);
    e.writeUInt32LE(value.length, 4);
    if (value.length <= 4) value.copy(e, 8);
    else {
      e.writeUInt32LE(dataPos, 8);
      data.push(value);
      dataPos += value.length;
    }
    return e;
  };
  const head = Buffer.from([0x49, 0x49, 0x2a, 0, 8, 0, 0, 0]);
  const ifd0 = [Buffer.alloc(2), ...entries0.map(([t, ty, v]) => entry(t, ty, v))];
  if (exifEntries.length) {
    const e = Buffer.alloc(12);
    e.writeUInt16LE(0x8769, 0);
    e.writeUInt16LE(4, 2);
    e.writeUInt32LE(1, 4);
    e.writeUInt32LE(exifPos, 8);
    ifd0.push(e);
  }
  ifd0[0].writeUInt16LE(ifd0.length - 1);
  ifd0.push(Buffer.alloc(4));
  const exif = exifEntries.length ? [Buffer.alloc(2), ...exifEntries.map(([t, ty, v]) => entry(t, ty, v)), Buffer.alloc(4)] : [];
  if (exif.length) exif[0].writeUInt16LE(exifEntries.length);
  return Buffer.concat([head, ...ifd0, ...exif, ...data]);
}

function webp(exif: Buffer): Buffer {
  const chunk = (fourcc: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32LE(data.length);
    return Buffer.concat([Buffer.from(fourcc, 'latin1'), len, data, data.length & 1 ? Buffer.alloc(1) : Buffer.alloc(0)]);
  };
  const body = Buffer.concat([Buffer.from('WEBP', 'latin1'), chunk('VP8X', Buffer.alloc(10)), chunk('EXIF', exif), chunk('VP8 ', Buffer.alloc(3))]);
  const size = Buffer.alloc(4);
  size.writeUInt32LE(body.length);
  return Buffer.concat([Buffer.from('RIFF', 'latin1'), size, body]);
}

function jpeg(exif: Buffer | null, comment: string | null): Buffer {
  const seg = (marker: number, data: Buffer) => {
    const len = Buffer.alloc(2);
    len.writeUInt16BE(data.length + 2);
    return Buffer.concat([Buffer.from([0xff, marker]), len, data]);
  };
  const parts = [Buffer.from([0xff, 0xd8])];
  if (exif) parts.push(seg(0xe1, Buffer.concat([Buffer.from('Exif\0\0', 'latin1'), exif])));
  if (comment) parts.push(seg(0xfe, Buffer.from(comment, 'utf8')));
  parts.push(seg(0xdb, Buffer.alloc(65)), Buffer.from([0xff, 0xda, 0, 2]), Buffer.alloc(20), Buffer.from([0xff, 0xd9]));
  return Buffer.concat(parts);
}

let dir: string;
beforeAll(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gen-text-')); });
afterAll(() => { fs.rmSync(dir, { recursive: true, force: true }); });
const write = (name: string, buf: Buffer) => {
  const p = path.join(dir, name);
  fs.writeFileSync(p, buf);
  return p;
};

test('PNG: tEXt, zTXt and iTXt chunks each decode by keyword; IDAT is skipped', async () => {
  const prompt = JSON.stringify({ '3': { class_type: 'KSampler', inputs: { seed: 1 } } });
  const file = write('comfy.png', png([
    pngChunk('tEXt', Buffer.from(`prompt\0${prompt}`, 'latin1')),
    pngChunk('zTXt', Buffer.concat([Buffer.from('workflow\0\0', 'latin1'), zlib.deflateSync(Buffer.from('{"nodes":[]}'))])),
    pngChunk('iTXt', Buffer.from('parameters\0\0\0en\0\0über — Steps: 1', 'utf8')),
  ]));
  expect(await readGenerationText(file)).toEqual({
    prompt,
    workflow: '{"nodes":[]}',
    parameters: 'über — Steps: 1',
  });
});

test('PNG: the loki-retouch record (iTXt keyed `loki`, as the png crate writes it) comes back verbatim', async () => {
  const record = JSON.stringify({ loki: 1, tool: 'loki-retouch', version: '0.1.0', prompt: 'über — make it night', seed: 3, inputs: [{ tag: '<image1>', path: 'photo.jpg' }] });
  // keyword \0 compression-flag(0) method(0) language \0 translated-keyword \0 text — the png crate leaves language/translation empty
  const file = write('retouch.png', png([pngChunk('iTXt', Buffer.from(`loki\0\0\0\0\0${record}`, 'utf8'))]));
  expect(await readGenerationText(file)).toEqual({ loki: record });
});

test('PNG with no text chunks, and non-image junk, report nothing', async () => {
  expect(await readGenerationText(write('plain.png', png([])))).toBeUndefined();
  expect(await readGenerationText(write('junk.bin', Buffer.from('not an image at all')))).toBeUndefined();
  expect(await readGenerationText(path.join(dir, 'missing.png'))).toBeUndefined();
});

test('WebP: ComfyUI Exif Make/Model fields and a WebUI UNICODE UserComment', async () => {
  const uc = Buffer.concat([Buffer.from('UNICODE\0', 'latin1'), Buffer.from('a cat\nSteps: 20, Seed: 5', 'utf16le').swap16()]);
  const file = write('comfy.webp', webp(tiff({ make: 'workflow:{"nodes":[]}', model: 'prompt:{"1":{}}', userComment: uc })));
  expect(await readGenerationText(file)).toEqual({
    Make: 'workflow:{"nodes":[]}',
    Model: 'prompt:{"1":{}}',
    UserComment: 'a cat\nSteps: 20, Seed: 5',
  });
});

test('JPEG: APP1 Exif and COM comment segments', async () => {
  const uc = Buffer.concat([Buffer.from('ASCII\0\0\0', 'latin1'), Buffer.from('prompt text\nSteps: 9', 'latin1')]);
  const file = write('webui.jpg', jpeg(tiff({ userComment: uc }), 'made with love'));
  expect(await readGenerationText(file)).toEqual({ UserComment: 'prompt text\nSteps: 9', Comment: 'made with love' });
});

test('TIFF walker survives a truncated directory', () => {
  const t = tiff({ make: 'workflow:{}', model: 'prompt:{}' });
  expect(readTiffText(t.subarray(0, 20))).toEqual({});
  expect(readTiffText(Buffer.from('nope'))).toEqual({});
});
