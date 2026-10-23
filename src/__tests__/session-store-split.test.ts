/**
 * @jest-environment node
 *
 * The main-process session store keeps the item lists (library, previous)
 * in their own files so a cursor move rewrites only the small session.json
 * instead of re-stringifying a possibly million-item library.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';

let dir: string;

jest.mock('electron', () => ({
  app: { getPath: () => dir, on: jest.fn() },
  ipcMain: { handle: jest.fn() },
}));

function freshStore() {
  let store: any;
  jest.isolateModules(() => {
    // eslint-disable-next-line global-require
    store = require('../main/sessionStore').getSessionStore();
  });
  return store;
}

const lib = (n: number) => ({
  library: Array.from({ length: n }, (_, i) => ({ path: `/m/${i}.jpg`, mtimeMs: i })),
  initialFile: '/m/0.jpg',
});

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'session-store-'));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test('cursor writes do not rewrite the library file', async () => {
  const store = freshStore();
  await store.init();
  store.set('library', lib(1000));
  await store.flush();
  const libFile = path.join(dir, 'session-library.json');
  const before = fs.statSync(libFile).mtimeMs;
  const writes = jest.spyOn(fs.promises, 'writeFile');

  store.set('cursor', { cursor: 5 });
  await store.flush();

  const written = writes.mock.calls.map((c) => String(c[0]));
  expect(written.some((f) => f.includes('session-library'))).toBe(false);
  expect(written.some((f) => f.endsWith('session.json.tmp'))).toBe(true);
  expect(fs.statSync(libFile).mtimeMs).toBe(before);
  const small = JSON.parse(fs.readFileSync(path.join(dir, 'session.json'), 'utf-8'));
  expect(small.cursor).toEqual({ cursor: 5 });
  expect(small.library).toBeUndefined();
  writes.mockRestore();
});

test('round-trips every key across a restart', async () => {
  const store = freshStore();
  await store.init();
  store.setMany({
    library: lib(3),
    previous: { previousLibrary: lib(2).library, previousCursor: 1 },
    cursor: { cursor: 2 },
  });
  await store.flush();

  const again = freshStore();
  await again.init();
  expect(again.get('library')).toEqual(lib(3));
  expect(again.get('previous')).toEqual({ previousLibrary: lib(2).library, previousCursor: 1 });
  expect(again.get('cursor')).toEqual({ cursor: 2 });
});

test('migrates a pre-split session.json with inline lists', async () => {
  fs.writeFileSync(
    path.join(dir, 'session.json'),
    JSON.stringify({ library: lib(4), cursor: { cursor: 3 }, query: null, previous: null, version: 1 })
  );
  const store = freshStore();
  await store.init();
  expect(store.get('library')).toEqual(lib(4));

  store.set('cursor', { cursor: 1 });
  await store.flush();
  const small = JSON.parse(fs.readFileSync(path.join(dir, 'session.json'), 'utf-8'));
  expect(small.library).toBeUndefined();

  const again = freshStore();
  await again.init();
  expect(again.get('library')).toEqual(lib(4));
  expect(again.get('cursor')).toEqual({ cursor: 1 });
});

test('clear empties the large files too', async () => {
  const store = freshStore();
  await store.init();
  store.set('library', lib(3));
  await store.flush();
  store.clear();
  await store.flush();
  const again = freshStore();
  await again.init();
  expect(again.get('library')).toBeNull();
});
