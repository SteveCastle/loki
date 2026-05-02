import { app } from 'electron';
import fs from 'fs';
import path from 'path';

// Append-only JSONL log of every DB query the Electron Database wrapper
// executes. One line per query, written from a single shared write stream.
// Each entry has: ts, source ("electron"), name (optional), sql (whitespace-
// collapsed), params, duration_ms, rows, error. The user runs the app for a
// while and then ships query-log.jsonl back; the slow queries become obvious.

let cachedLogPath: string | null = null;
let stream: fs.WriteStream | null = null;
let enabled = true;

function resolveLogPath(): string {
  if (cachedLogPath) return cachedLogPath;
  try {
    const dir = app.getPath('userData');
    cachedLogPath = path.join(dir, 'query-log.jsonl');
  } catch {
    // Fall back to cwd if we're called before app is ready (shouldn't happen in
    // practice — Database is constructed after app.whenReady — but be safe).
    cachedLogPath = path.join(process.cwd(), 'query-log.jsonl');
  }
  return cachedLogPath;
}

function getStream(): fs.WriteStream | null {
  if (stream) return stream;
  try {
    stream = fs.createWriteStream(resolveLogPath(), { flags: 'a' });
    stream.on('error', (e) => {
      console.error('[queryLog] write stream error', e);
      stream = null;
    });
    return stream;
  } catch (e) {
    console.error('[queryLog] failed to open log file', e);
    return null;
  }
}

function collapseWhitespace(s: string): string {
  return s.replace(/\s+/g, ' ').trim();
}

export interface QueryLogEntry {
  name?: string;
  sql: string;
  params?: unknown[];
  duration_ms: number;
  rows?: number | null;
  error?: string | null;
}

export function logQuery(entry: QueryLogEntry): void {
  if (!enabled) return;
  try {
    const s = getStream();
    if (!s) return;
    const line =
      JSON.stringify({
        ts: new Date().toISOString(),
        source: 'electron',
        name: entry.name,
        sql: collapseWhitespace(entry.sql),
        params: entry.params,
        duration_ms: Number(entry.duration_ms.toFixed(3)),
        rows: entry.rows ?? null,
        error: entry.error ?? null,
      }) + '\n';
    s.write(line);
  } catch {
    // Never let logging break the app.
  }
}

export function setQueryLogEnabled(v: boolean): void {
  enabled = v;
}

export function getQueryLogPath(): string {
  return resolveLogPath();
}

export function flushQueryLog(): Promise<void> {
  return new Promise((resolve) => {
    const s = stream;
    if (!s) {
      resolve();
      return;
    }
    s.end(() => {
      stream = null;
      resolve();
    });
  });
}
