/**
 * Tests for the SRT→VTT conversion helper. The helper produces a string
 * (callers wrap it in a Blob URL). Conversion rules:
 *   - SRT cue timestamps use commas as the millisecond separator;
 *     VTT uses periods. Replace `,` with `.` only inside timestamp lines.
 *   - SRT files have no header. Prepend `WEBVTT\n\n`.
 *   - VTT input passes through unchanged except for an enforced trailing
 *     newline so callers can always concatenate cues safely.
 *   - Empty / malformed input still produces a parseable VTT (header only).
 */
import { srtToVtt, toVttString } from '../renderer/components/media-viewers/subtitle-loader';

describe('srtToVtt', () => {
  it('prepends WEBVTT header', () => {
    expect(srtToVtt('1\n00:00:01,000 --> 00:00:02,000\nHello\n')).toMatch(/^WEBVTT\n\n/);
  });

  it('replaces comma millisecond separator with period in timestamps', () => {
    const out = srtToVtt('1\n00:00:01,500 --> 00:00:02,750\nHi\n');
    expect(out).toContain('00:00:01.500 --> 00:00:02.750');
  });

  it('does not replace commas in cue text', () => {
    const out = srtToVtt('1\n00:00:01,000 --> 00:00:02,000\nHello, world\n');
    expect(out).toContain('Hello, world');
  });

  it('preserves multiple cues', () => {
    const srt = [
      '1',
      '00:00:01,000 --> 00:00:02,000',
      'First',
      '',
      '2',
      '00:00:03,000 --> 00:00:04,000',
      'Second',
      '',
    ].join('\n');
    const out = srtToVtt(srt);
    expect(out).toContain('First');
    expect(out).toContain('Second');
    expect(out).toContain('00:00:03.000 --> 00:00:04.000');
  });

  it('handles empty input as an empty VTT', () => {
    expect(srtToVtt('')).toBe('WEBVTT\n\n');
  });

  it('strips a leading UTF-8 BOM so it does not corrupt the first cue', () => {
    const srt = '﻿1\n00:00:01,000 --> 00:00:02,000\nHello\n';
    const out = srtToVtt(srt);
    expect(out.startsWith('WEBVTT\n\n')).toBe(true);
    // The BOM must not survive into the cue body.
    expect(out).not.toContain('﻿');
  });

  it('normalizes CRLF line endings to LF', () => {
    const srt = '1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n';
    const out = srtToVtt(srt);
    expect(out).not.toContain('\r');
    expect(out).toContain('00:00:01.000 --> 00:00:02.000');
  });

  it('normalizes lone CR line endings to LF', () => {
    const srt = '1\r00:00:01,000 --> 00:00:02,000\rHello\r';
    const out = srtToVtt(srt);
    expect(out).not.toContain('\r');
    expect(out).toContain('00:00:01.000 --> 00:00:02.000');
  });
});

describe('toVttString', () => {
  it('passes VTT input through with header preserved', () => {
    const vtt = 'WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHi\n';
    expect(toVttString(vtt, 'vtt')).toContain('WEBVTT');
    expect(toVttString(vtt, 'vtt')).toContain('00:00:01.000 --> 00:00:02.000');
  });

  it('converts SRT input via srtToVtt', () => {
    expect(toVttString('1\n00:00:01,000 --> 00:00:02,000\nHi\n', 'srt')).toMatch(
      /^WEBVTT\n\n/
    );
  });

  it('adds a WEBVTT header to VTT input that is missing one', () => {
    const noHeader = '1\n00:00:01.000 --> 00:00:02.000\nHi\n';
    expect(toVttString(noHeader, 'vtt')).toMatch(/^WEBVTT\n\n/);
  });

  it('strips BOM and normalizes CRLF in a VTT input that has both', () => {
    const vtt = '﻿WEBVTT\r\n\r\n1\r\n00:00:01.000 --> 00:00:02.000\r\nHi\r\n';
    const out = toVttString(vtt, 'vtt');
    expect(out.startsWith('WEBVTT')).toBe(true);
    expect(out).not.toContain('\r');
    expect(out).not.toContain('﻿');
  });
});
