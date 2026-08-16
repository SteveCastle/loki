/**
 * filePathFromArgv resolves "what was this (second) launch asked to open" from
 * an argv that arrives with Chromium switches mixed in. It backs the
 * single-instance forwarding in main.ts: a losing instance quits and the
 * running one receives this path over the 'open-path' channel.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import { filePathFromArgv } from '../main/file-handling';

describe('filePathFromArgv', () => {
  let dir: string;
  let file: string;

  beforeAll(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'loki-argv-'));
    file = path.join(dir, 'clip.mp4');
    fs.writeFileSync(file, 'x');
  });

  afterAll(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('picks the file argument from a packaged-style argv', () => {
    expect(filePathFromArgv(['C:\\app\\viewer.exe', file], 1)).toBe(file);
  });

  it('ignores switches appended after the file', () => {
    expect(
      filePathFromArgv(
        ['C:\\app\\viewer.exe', file, '--allow-file-access-from-files'],
        1
      )
    ).toBe(file);
  });

  it('accepts a directory argument (folder opens are supported)', () => {
    expect(filePathFromArgv(['C:\\app\\viewer.exe', dir], 1)).toBe(dir);
  });

  it('rejects paths that do not exist', () => {
    const ghost = path.join(dir, 'does-not-exist.mp4');
    expect(filePathFromArgv(['C:\\app\\viewer.exe', ghost], 1)).toBe('');
  });

  it('startIndex skips the dev-mode app path when no file was given', () => {
    // dev argv is [electron, appPath, ...]; appPath is a real directory and
    // must not be mistaken for something the user opened.
    expect(filePathFromArgv(['C:\\tools\\electron.exe', dir], 2)).toBe('');
  });

  it('still finds a real file after the dev-mode app path', () => {
    expect(filePathFromArgv(['C:\\tools\\electron.exe', dir, file], 2)).toBe(
      file
    );
  });

  it('returns empty for a bare relaunch with no arguments', () => {
    expect(filePathFromArgv(['C:\\app\\viewer.exe'], 1)).toBe('');
  });
});
