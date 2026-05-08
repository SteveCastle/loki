/**
 * Tests the sidecar lookup helper. We separate the lookup from the IPC
 * handler so the lookup can be tested without an Electron context.
 *
 * Lookup precedence (for video at `dir/movie.mp4`):
 *   1. `dir/movie.srt`
 *   2. `dir/movie.mp4.srt`
 *   3. `dir/movie.vtt`
 *   4. `dir/movie.mp4.vtt`
 * The basename-only form (#1, #3) is tried before the video-extension-included
 * form (#2, #4) for each subtitle extension. Returns null when nothing matches.
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

  it('returns srt content when only <basename><videoExt>.srt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.mp4.srt'), 'full-name-srt');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'full-name-srt' });
  });

  it('returns vtt content when only <basename><videoExt>.vtt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.mp4.vtt'), 'full-name-vtt');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'vtt', content: 'full-name-vtt' });
  });

  it('prefers <basename>.srt over <basename><videoExt>.srt', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.srt'), 'short');
    fs.writeFileSync(path.join(tmp, 'movie.mp4.srt'), 'long');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'short' });
  });

  it('prefers <basename><videoExt>.srt over <basename>.vtt', async () => {
    // Both exist; precedence is per-extension (srt before vtt) regardless of
    // basename form. .srt wins even when only the longer .srt form is present.
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.mp4.srt'), 'long-srt');
    fs.writeFileSync(path.join(tmp, 'movie.vtt'), 'short-vtt');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'long-srt' });
  });
});
