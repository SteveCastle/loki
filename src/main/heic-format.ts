// Byte-level helpers for HEIC/HEIF display conversion (see heic.ts). Pure
// functions over Buffers — no Electron, no fs — so they unit-test directly.
//
// ffmpeg (8.0+) decodes HEIC fine, including the 512×512 tile grids every
// iPhone photo is stored as, but assembling the grid drops the image's ICC
// profile. iPhone photos are Display P3, so a JPEG without the profile renders
// visibly washed out. We read the profile (and the primary image's declared
// size, for a sanity check) straight out of the HEIF item properties and put
// the profile back into the JPEG ffmpeg produced.

export const HEIC_EXT_RE = /\.(heic|heif)$/i;

export function isHeicPath(filePath: string): boolean {
  return HEIC_EXT_RE.test(filePath);
}

export interface HeicInfo {
  /** ICC profile of the primary image, or null when it declares none. */
  icc: Buffer | null;
  /** Display size of the primary image (after irot), 0 when undeclared. */
  width: number;
  height: number;
  /** A clean-aperture crop applies, so the decoded size won't match ispe. */
  cropped: boolean;
}

interface Box {
  type: string;
  /** Offset of the payload (after size/type/largesize). */
  body: number;
  end: number;
}

// ISOBMFF boxes laid end to end inside [start, end). Stops at the first
// malformed header rather than throwing: a truncated read just yields fewer
// boxes.
export function* isoBoxes(b: Buffer, start: number, end: number): Generator<Box> {
  let p = start;
  while (p + 8 <= end) {
    let size = b.readUInt32BE(p);
    const type = b.toString('latin1', p + 4, p + 8);
    let body = p + 8;
    if (size === 1) {
      if (p + 16 > end) return;
      size = Number(b.readBigUInt64BE(p + 8));
      body = p + 16;
    } else if (size === 0) {
      size = end - p;
    }
    if (size < body - p || p + size > end) return;
    yield { type, body, end: p + size };
    p += size;
  }
}

function child(b: Buffer, parent: Box, type: string, skip = 0): Box | undefined {
  for (const box of isoBoxes(b, parent.body + skip, parent.end)) {
    if (box.type === type) return box;
  }
  return undefined;
}

/**
 * Reads the primary item's ICC profile and display size from a buffer that
 * holds the top-level `meta` box (with or without the boxes around it).
 * Returns null when there is no usable meta box.
 */
export function parseHeicMeta(b: Buffer): HeicInfo | null {
  let meta: Box | undefined;
  for (const box of isoBoxes(b, 0, b.length)) {
    if (box.type === 'meta') meta = box;
  }
  if (!meta) return null;
  // meta, pitm and ipma are FullBoxes: 1 byte version + 3 bytes flags.
  const pitm = child(b, meta, 'pitm', 4);
  const iprp = child(b, meta, 'iprp', 4);
  if (!pitm || !iprp) return null;
  const primary =
    b[pitm.body] === 0 ? b.readUInt16BE(pitm.body + 4) : b.readUInt32BE(pitm.body + 4);

  const ipco = child(b, iprp, 'ipco');
  const ipma = child(b, iprp, 'ipma');
  if (!ipco || !ipma) return null;
  const props = [...isoBoxes(b, ipco.body, ipco.end)];

  // ipma: entry_count, then per item its id and a list of 1-based ipco
  // indices (7 or 15 bits, the top bit flags the property as essential).
  const version = b[ipma.body];
  const wideIndex = (b[ipma.body + 3] & 1) === 1;
  let p = ipma.body + 4;
  const entries = b.readUInt32BE(p);
  p += 4;
  let indices: number[] = [];
  for (let i = 0; i < entries && p < ipma.end; i++) {
    const item = version < 1 ? b.readUInt16BE(p) : b.readUInt32BE(p);
    p += version < 1 ? 2 : 4;
    const n = b[p];
    p += 1;
    const list: number[] = [];
    for (let j = 0; j < n; j++) {
      if (wideIndex) {
        list.push(b.readUInt16BE(p) & 0x7fff);
        p += 2;
      } else {
        list.push(b[p] & 0x7f);
        p += 1;
      }
    }
    if (item === primary) indices = list;
  }

  const info: HeicInfo = { icc: null, width: 0, height: 0, cropped: false };
  let quarterTurns = 0;
  for (const index of indices) {
    const prop = props[index - 1];
    if (!prop) continue;
    if (prop.type === 'colr') {
      const kind = b.toString('latin1', prop.body, prop.body + 4);
      if (kind === 'prof' || kind === 'rICC') {
        info.icc = Buffer.from(b.subarray(prop.body + 4, prop.end));
      }
    } else if (prop.type === 'ispe') {
      info.width = b.readUInt32BE(prop.body + 4);
      info.height = b.readUInt32BE(prop.body + 8);
    } else if (prop.type === 'irot') {
      quarterTurns = b[prop.body] & 3;
    } else if (prop.type === 'clap') {
      info.cropped = true;
    }
  }
  if (quarterTurns % 2 === 1) {
    [info.width, info.height] = [info.height, info.width];
  }
  return info;
}

const SOI = 0xd8;
const SOS = 0xda;
const APP0 = 0xe0;
const APP2 = 0xe2;
const ICC_TAG = Buffer.from('ICC_PROFILE\0', 'latin1');
// Segment length is a u16 that counts itself, the tag and the seq/count bytes.
const ICC_CHUNK = 0xffff - 2 - ICC_TAG.length - 2;

interface Segment {
  marker: number;
  start: number;
  /** Offset of the segment payload (after the length field). */
  body: number;
  end: number;
}

// JPEG marker segments up to (not including) the entropy-coded data.
function* jpegSegments(b: Buffer): Generator<Segment> {
  if (b.length < 4 || b[0] !== 0xff || b[1] !== SOI) return;
  let p = 2;
  while (p + 4 <= b.length && b[p] === 0xff) {
    const marker = b[p + 1];
    const len = b.readUInt16BE(p + 2);
    yield { marker, start: p, body: p + 4, end: p + 2 + len };
    if (marker === SOS) return;
    p += 2 + len;
  }
}

/** Pixel size from the JPEG's SOF header, or null when there isn't one. */
export function jpegSize(b: Buffer): { width: number; height: number } | null {
  for (const s of jpegSegments(b)) {
    const m = s.marker;
    // SOF0–SOF15, minus DHT (C4), JPG (C8) and DAC (CC).
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
      return { height: b.readUInt16BE(s.body + 1), width: b.readUInt16BE(s.body + 3) };
    }
  }
  return null;
}

function jpegHasIcc(b: Buffer): boolean {
  for (const s of jpegSegments(b)) {
    if (s.marker === APP2 && b.subarray(s.body, s.body + ICC_TAG.length).equals(ICC_TAG)) {
      return true;
    }
  }
  return false;
}

/**
 * The JPEG with `icc` embedded as APP2 ICC_PROFILE segment(s), placed after
 * SOI/APP0 where readers expect it. A JPEG that already carries a profile is
 * returned unchanged.
 */
export function withJpegIcc(jpeg: Buffer, icc: Buffer): Buffer {
  if (!icc.length || jpegHasIcc(jpeg)) return jpeg;
  const count = Math.ceil(icc.length / ICC_CHUNK);
  if (count > 255) return jpeg;
  let insertAt = 2;
  for (const s of jpegSegments(jpeg)) {
    if (s.marker === APP0 && s.start === 2) insertAt = s.end;
    break;
  }
  const segments: Buffer[] = [];
  for (let i = 0; i < count; i++) {
    const data = icc.subarray(i * ICC_CHUNK, (i + 1) * ICC_CHUNK);
    const header = Buffer.alloc(4);
    header[0] = 0xff;
    header[1] = APP2;
    header.writeUInt16BE(2 + ICC_TAG.length + 2 + data.length, 2);
    segments.push(header, ICC_TAG, Buffer.from([i + 1, count]), data);
  }
  return Buffer.concat([jpeg.subarray(0, insertAt), ...segments, jpeg.subarray(insertAt)]);
}
