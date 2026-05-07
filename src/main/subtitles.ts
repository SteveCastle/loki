/**
 * Sidecar subtitle file lookup + IPC handler.
 *
 * For a given video path we check for `<basename>.srt` first, then
 * `<basename>.vtt` in the same directory. Exact basename only — no
 * language-suffix matching by design.
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
  const ext = path.extname(videoPath);
  const base = path.basename(videoPath, ext);

  for (const candidate of EXTS) {
    const p = path.join(dir, `${base}.${candidate}`);
    try {
      const content = await fs.promises.readFile(p, 'utf-8');
      return { ext: candidate, content };
    } catch (err: any) {
      if (err && err.code === 'ENOENT') continue;
      // Other read errors (permission, etc.): log and treat as not found.
      console.warn('[subtitles] failed to read', p, err);
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
