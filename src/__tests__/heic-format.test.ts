import fs from 'fs';
import {
  isHeicPath,
  jpegSize,
  parseHeicMeta,
  withJpegIcc,
} from '../main/heic-format';

function box(type: string, ...payload: Buffer[]): Buffer {
  const body = Buffer.concat(payload);
  const header = Buffer.alloc(8);
  header.writeUInt32BE(8 + body.length, 0);
  header.write(type, 4, 'latin1');
  return Buffer.concat([header, body]);
}

const full = (version: number, flags = 0) =>
  Buffer.from([version, (flags >> 16) & 0xff, (flags >> 8) & 0xff, flags & 0xff]);
const u16 = (n: number) => {
  const b = Buffer.alloc(2);
  b.writeUInt16BE(n);
  return b;
};
const u32 = (n: number) => {
  const b = Buffer.alloc(4);
  b.writeUInt32BE(n);
  return b;
};

const ICC = Buffer.from('fake-display-p3-profile');

// A meta box shaped like an iPhone photo: primary item 49 (the grid) with
// colr/ispe/irot, plus a second item (a depth map) with its own ICC that must
// not be picked up.
function heicMeta({ rotate = 0, wide = false } = {}): Buffer {
  const ipco = box(
    'ipco',
    box('colr', Buffer.from('prof'), ICC),
    box('ispe', full(0), u32(4032), u32(3024)),
    box('irot', Buffer.from([rotate])),
    box('colr', Buffer.from('prof'), Buffer.from('depth-profile'))
  );
  const idx = (i: number) => (wide ? u16(0x8000 | i) : Buffer.from([0x80 | i]));
  const ipma = box(
    'ipma',
    full(0, wide ? 1 : 0),
    u32(2),
    u16(62), Buffer.from([1]), idx(4),
    u16(49), Buffer.from([3]), idx(1), idx(2), idx(3)
  );
  return Buffer.concat([
    box('ftyp', Buffer.from('heic'), u32(0), Buffer.from('mif1heic')),
    box('meta', full(0), box('hdlr', full(0), Buffer.alloc(20)), box('pitm', full(0), u16(49)), box('iprp', ipco, ipma)),
    box('mdat', Buffer.alloc(16)),
  ]);
}

// SOI, APP0 (JFIF), SOF0 (h=3024, w=4032), SOS, EOI.
function jpeg(): Buffer {
  const app0 = Buffer.concat([Buffer.from([0xff, 0xe0]), u16(16), Buffer.from('JFIF\0'), Buffer.alloc(9)]);
  const sof0 = Buffer.concat([Buffer.from([0xff, 0xc0]), u16(17), Buffer.from([8]), u16(3024), u16(4032), Buffer.from([3]), Buffer.alloc(9)]);
  const sos = Buffer.concat([Buffer.from([0xff, 0xda]), u16(8), Buffer.alloc(6)]);
  return Buffer.concat([Buffer.from([0xff, 0xd8]), app0, sof0, sos, Buffer.from([0x12, 0x34, 0xff, 0xd9])]);
}

describe('heic-format', () => {
  it('recognises .heic and .heif in any case', () => {
    expect(isHeicPath('C:\\Photos\\IMG_0001.HEIC')).toBe(true);
    expect(isHeicPath('/x/y.heif')).toBe(true);
    expect(isHeicPath('/x/heic.jpg')).toBe(false);
  });

  it("reads the primary item's ICC and size, not another item's", () => {
    const info = parseHeicMeta(heicMeta());
    expect(info).not.toBeNull();
    expect(info!.icc).toEqual(ICC);
    expect([info!.width, info!.height]).toEqual([4032, 3024]);
    expect(info!.cropped).toBe(false);
  });

  it('swaps the size for a quarter-turn irot', () => {
    const info = parseHeicMeta(heicMeta({ rotate: 3 }));
    expect([info!.width, info!.height]).toEqual([3024, 4032]);
  });

  it('handles 15-bit ipma property indices', () => {
    expect(parseHeicMeta(heicMeta({ wide: true }))!.icc).toEqual(ICC);
  });

  it('returns null without a meta box and survives truncation', () => {
    expect(parseHeicMeta(box('ftyp', Buffer.from('heic')))).toBeNull();
    const meta = heicMeta();
    for (let n = 0; n < meta.length; n += 7) {
      expect(() => parseHeicMeta(meta.subarray(0, n))).not.toThrow();
    }
  });

  it('reads the JPEG size from SOF', () => {
    expect(jpegSize(jpeg())).toEqual({ width: 4032, height: 3024 });
    expect(jpegSize(Buffer.from('not a jpeg'))).toBeNull();
  });

  it('embeds the ICC profile after APP0 and keeps the image intact', () => {
    const src = jpeg();
    const out = withJpegIcc(src, ICC);
    // APP2 right after SOI(2) + APP0(18)
    expect([out[20], out[21]]).toEqual([0xff, 0xe2]);
    expect(out.subarray(24, 36).toString('latin1')).toBe('ICC_PROFILE\0');
    expect([out[36], out[37]]).toEqual([1, 1]);
    expect(out.subarray(38, 38 + ICC.length)).toEqual(ICC);
    expect(jpegSize(out)).toEqual({ width: 4032, height: 3024 });
    expect(out.subarray(out.length - 4)).toEqual(src.subarray(src.length - 4));
    // Idempotent: a JPEG that has a profile isn't given a second one.
    expect(withJpegIcc(out, ICC)).toBe(out);
  });

  it('splits a large profile across numbered APP2 segments', () => {
    const big = Buffer.alloc(150_000, 7);
    const out = withJpegIcc(jpeg(), big);
    expect(out.length).toBe(jpeg().length + big.length + 3 * (4 + 12 + 2));
  });

  // Opt-in check against a real photo: LOKI_HEIC_SAMPLE=<path to .heic>
  const sample = process.env.LOKI_HEIC_SAMPLE;
  (sample ? it : it.skip)('parses a real HEIC', () => {
    const info = parseHeicMeta(fs.readFileSync(sample!));
    expect(info).not.toBeNull();
    expect(info!.width).toBeGreaterThan(0);
    expect(info!.icc?.length).toBeGreaterThan(0);
  });
});
