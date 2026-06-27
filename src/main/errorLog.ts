import { app } from 'electron';
import fs from 'fs';
import path from 'path';

// Append-only JSONL log of app-level errors and data-load failures, written
// from the main process. This is deliberately separate from query-log.jsonl
// (which records every DB query): this file is for the things that strand the
// app — uncaught exceptions, unhandled rejections, IPC handler failures, and
// load lifecycle events (load-db / load-files start / timeout / error).
//
// The user reproduces the hang, then ships app-log.jsonl back; the last lines
// reveal which load never completed (e.g. a `load-files` start with no matching
// done, or a `load-db` timeout) instead of leaving us guessing.
//
// Lives at <userData>/app-log.jsonl. Logging must never throw.

export type LogLevel = 'error' | 'warn' | 'info';

let cachedLogPath: string | null = null;
let stream: fs.WriteStream | null = null;
let enabled = true;

function resolveLogPath(): string {
  if (cachedLogPath) return cachedLogPath;
  try {
    cachedLogPath = path.join(app.getPath('userData'), 'app-log.jsonl');
  } catch {
    // Called before app is ready (shouldn't happen, but never crash on it).
    cachedLogPath = path.join(process.cwd(), 'app-log.jsonl');
  }
  return cachedLogPath;
}

function getStream(): fs.WriteStream | null {
  if (stream) return stream;
  try {
    stream = fs.createWriteStream(resolveLogPath(), { flags: 'a' });
    stream.on('error', (e) => {
      // eslint-disable-next-line no-console
      console.error('[errorLog] write stream error', e);
      stream = null;
    });
    return stream;
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error('[errorLog] failed to open log file', e);
    return null;
  }
}

// Turn an arbitrary thrown value into a JSON-serializable shape. Errors don't
// serialize via JSON.stringify (message/stack are non-enumerable), so pull the
// useful fields out explicitly.
export function serializeError(err: unknown): unknown {
  if (err == null) return null;
  if (err instanceof Error) {
    const out: Record<string, unknown> = {
      name: err.name,
      message: err.message,
      stack: err.stack,
    };
    const code = (err as { code?: unknown }).code;
    if (code !== undefined) out.code = code;
    const errno = (err as { errno?: unknown }).errno;
    if (errno !== undefined) out.errno = errno;
    return out;
  }
  if (typeof err === 'object') {
    try {
      return JSON.parse(JSON.stringify(err));
    } catch {
      return { value: String(err) };
    }
  }
  return { value: String(err) };
}

export interface LogEntryInput {
  level?: LogLevel;
  /** Where it came from, e.g. 'main:uncaughtException', 'ipc:load-files'. */
  scope: string;
  message: string;
  data?: unknown;
  /** A thrown value; serialized into the entry. */
  error?: unknown;
}

export function logEvent(entry: LogEntryInput): void {
  if (!enabled) return;
  try {
    const s = getStream();
    if (!s) return;
    const line =
      JSON.stringify({
        ts: new Date().toISOString(),
        source: 'electron',
        level: entry.level ?? 'error',
        scope: entry.scope,
        message: entry.message,
        data: entry.data ?? null,
        error: entry.error !== undefined ? serializeError(entry.error) : null,
      }) + '\n';
    s.write(line);
  } catch {
    // Never let logging break the app.
  }
}

export function setAppLogEnabled(v: boolean): void {
  enabled = v;
}

export function getAppLogPath(): string {
  return resolveLogPath();
}

export function flushAppLog(): Promise<void> {
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

/**
 * Installs process-level handlers so nothing crashes or hangs silently. Safe to
 * call once at main startup. Logs to the file and still logs to console.
 */
export function installGlobalErrorHandlers(): void {
  process.on('uncaughtException', (error) => {
    // eslint-disable-next-line no-console
    console.error('Uncaught exception in main process:', error);
    logEvent({ scope: 'main:uncaughtException', message: 'Uncaught exception', error });
  });
  process.on('unhandledRejection', (reason) => {
    // eslint-disable-next-line no-console
    console.error('Unhandled promise rejection in main process:', reason);
    logEvent({
      scope: 'main:unhandledRejection',
      message: 'Unhandled promise rejection',
      error: reason,
    });
  });
}
