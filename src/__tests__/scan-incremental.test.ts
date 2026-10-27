/**
 * Streamed folder scans: per-batch work must be proportional to the batch,
 * not the library, and give exactly what a full recompute would.
 */
import filter from '../renderer/filter';
import { appendUnique, _resetScanIndex } from '../renderer/scan-index';

const item = (p: string) => ({ path: p, mtimeMs: 0 } as any);
const exts = ['jpg', 'mp4', 'png', 'mp3', 'webm'];
const batchOf = (start: number, n: number) =>
  Array.from({ length: n }, (_, i) => item(`C:/lib/f${start + i}.${exts[(start + i) % exts.length]}`));

beforeEach(() => _resetScanIndex());

test('appendUnique de-dupes case-insensitively across and within batches', () => {
  let lib = appendUnique([], [item('C:/A.jpg'), item('c:/a.JPG'), item('C:/b.jpg')]);
  expect(lib.map((i) => i.path)).toEqual(['C:/A.jpg', 'C:/b.jpg']);
  lib = appendUnique(lib, [item('C:/B.JPG'), item('C:/c.jpg')]);
  expect(lib.map((i) => i.path)).toEqual(['C:/A.jpg', 'C:/b.jpg', 'C:/c.jpg']);
});

test('appendUnique rebuilds its index for a library it did not produce', () => {
  appendUnique([], [item('/x.jpg')]);
  const foreign = [item('/y.jpg')];
  const lib = appendUnique(foreign, [item('/x.jpg'), item('/Y.JPG')]);
  expect(lib.map((i) => i.path)).toEqual(['/y.jpg', '/x.jpg']);
});

test.each(['all', 'video', 'static'] as const)(
  'incremental stream filter matches a full recompute (%s)',
  (filters) => {
    let lib: any[] = [];
    let id = 0;
    for (let b = 0; b < 20; b++) {
      lib = appendUnique(lib, batchOf(b * 50, 50));
      const incremental = filter(`load-${++id}`, '', lib, filters, 'stream');
      const full = lib.filter((_, i) => true);
      // A fresh key + a non-extending call forces the full path for comparison.
      filter(`reset-${id}`, '', [item('/z.jpg')], filters, 'name');
      const recomputed = filter(`full-${id}`, '', full, filters, 'stream');
      expect(incremental.map((i: any) => i.path)).toEqual(recomputed.map((i: any) => i.path));
    }
  }
);

test('a 200k-file scan in 500-file batches stays linear', () => {
  let lib: any[] = [];
  const start = Date.now();
  for (let b = 0; b < 400; b++) {
    lib = appendUnique(lib, batchOf(b * 500, 500));
    filter(`scan-${b}`, '', lib, 'video', 'stream');
  }
  const ms = Date.now() - start;
  expect(lib.length).toBe(200000);
  // The old per-batch Set rebuild + full type filter took tens of seconds here.
  expect(ms).toBeLessThan(5000);
});
