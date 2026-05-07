/**
 * Tests the sidecar lookup helper. We separate the lookup from the IPC
 * handler so the lookup can be tested without an Electron context.
 *
 * Lookup precedence: `<basename>.srt` first, then `<basename>.vtt`.
 * Returns null when neither exists or when both are unreadable.
 */
import { findSidecarSubtitle } from '../main/subtitles';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

describe('findSidecarSubtitle', () => {
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'subtitle-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it('returns null when no sidecar exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    expect(await findSidecarSubtitle(video)).toBeNull();
  });

  it('returns srt content when only .srt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.srt'), 'srt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'srt-content' });
  });

  it('returns vtt content when only .vtt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.vtt'), 'vtt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'vtt', content: 'vtt-content' });
  });

  it('prefers .srt when both .srt and .vtt exist', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.srt'), 'srt-content');
    fs.writeFileSync(path.join(tmp, 'movie.vtt'), 'vtt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'srt-content' });
  });

  it('handles paths with multiple dots in basename', async () => {
    const video = path.join(tmp, 'movie.s01e02.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.s01e02.srt'), 'ok');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'ok' });
  });

  it('returns null when video path is empty', async () => {
    expect(await findSidecarSubtitle('')).toBeNull();
  });
});
