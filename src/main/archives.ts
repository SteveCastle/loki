import * as path from 'path';

export function isArchivePath(p: string): boolean {
  const ext = path.extname(p).toLowerCase();
  return ext === '.cbz' || ext === '.zip';
}
