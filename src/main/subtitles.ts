/**
 * Sidecar subtitle file lookup + IPC handler.
 *
 * For `dir/movie.mp4` we look for, in order:
 *   1. `dir/movie.srt`        — basename only
 *   2. `dir/movie.mp4.srt`    — full filename including the video extension
 *   3. `dir/movie.vtt`
 *   4. `dir/movie.mp4.vtt`
 * Exact match only — no language-suffix matching by design.
 */
import { ipcMain } from 'electron';
import * as fs from 'fs';
import * as path from 'path';

export type SubtitleSidecar = {
  ext: 'srt' | 'vtt';
  content: string;
};

const EXTS: Array<'srt' | 'vtt'> = ['srt', 'vtt'];

export async function findSidecarSubtitle(
  videoPath: string
): Promise<SubtitleSidecar | null> {
  if (!videoPath) return null;
  const dir = path.dirname(videoPath);
  const videoExt = path.extname(videoPath);
  const base = path.basename(videoPath, videoExt);
  const fullName = path.basename(videoPath); // basename + extension, e.g. "movie.mp4"

  for (const candidate of EXTS) {
    const candidatePaths = [
      path.join(dir, `${base}.${candidate}`),
      path.join(dir, `${fullName}.${candidate}`),
    ];
    for (const p of candidatePaths) {
      try {
        const content = await fs.promises.readFile(p, 'utf-8');
        return { ext: candidate, content };
      } catch (err: any) {
        if (err && err.code === 'ENOENT') continue;
        // Other read errors (permission, etc.): log and treat as not found.
        console.warn('[subtitles] failed to read', p, err);
      }
    }
  }
  return null;
}

export function registerSubtitleHandlers(): void {
  ipcMain.handle('find-subtitle', async (_event, args: unknown[]) => {
    const videoPath = Array.isArray(args) ? (args[0] as string) : '';
    return findSidecarSubtitle(videoPath);
  });
}
