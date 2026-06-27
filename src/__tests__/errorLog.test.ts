import os from 'os';
import path from 'path';

// Mock electron's app so errorLog resolves its path to a temp dir instead of a
// real userData location. Must be declared before importing errorLog.
const TMP_DIR = path.join(os.tmpdir(), 'loki-errorlog-test');
jest.mock('electron', () => ({
  app: { getPath: () => TMP_DIR },
}));

import fs from 'fs';
import {
  serializeError,
  logEvent,
  getAppLogPath,
  flushAppLog,
} from '../main/errorLog';

describe('serializeError', () => {
  it('extracts message, stack, name, and code from an Error', () => {
    const err = Object.assign(new Error('database is locked'), {
      code: 'SQLITE_BUSY',
    });
    const out = serializeError(err) as Record<string, unknown>;
    expect(out.name).toBe('Error');
    expect(out.message).toBe('database is locked');
    expect(out.code).toBe('SQLITE_BUSY');
    expect(typeof out.stack).toBe('string');
  });

  it('handles null/undefined and primitive values', () => {
    expect(serializeError(null)).toBeNull();
    expect(serializeError(undefined)).toBeNull();
    expect(serializeError('plain string')).toEqual({ value: 'plain string' });
    expect(serializeError(42)).toEqual({ value: '42' });
  });

  it('passes through plain objects', () => {
    expect(serializeError({ foo: 'bar' })).toEqual({ foo: 'bar' });
  });
});

describe('logEvent', () => {
  beforeAll(() => {
    fs.mkdirSync(TMP_DIR, { recursive: true });
    // Start from a clean file.
    try {
      fs.rmSync(getAppLogPath());
    } catch {
      // not there yet — fine
    }
  });

  afterAll(async () => {
    await flushAppLog();
  });

  it('appends a JSON line with the expected shape', async () => {
    logEvent({
      scope: 'ipc:load-files',
      message: 'load-files failed',
      data: { filePath: 'C:/x' },
      error: Object.assign(new Error('stalled'), { code: 'ETIMEDOUT' }),
    });
    await flushAppLog();

    const contents = fs.readFileSync(getAppLogPath(), 'utf8').trim();
    const lines = contents.split('\n').filter(Boolean);
    const last = JSON.parse(lines[lines.length - 1]);

    expect(last.scope).toBe('ipc:load-files');
    expect(last.message).toBe('load-files failed');
    expect(last.level).toBe('error');
    expect(last.source).toBe('electron');
    expect(last.data).toEqual({ filePath: 'C:/x' });
    expect(last.error.code).toBe('ETIMEDOUT');
    expect(typeof last.ts).toBe('string');
  });
});
