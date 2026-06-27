// Recovery helpers for transient SQLite lock contention.
//
// The desktop app and the Go media-server can hold the same `dream.sqlite`
// open at once. When the server holds a write lock while the app is starting,
// SQLite returns SQLITE_BUSY / SQLITE_LOCKED. These helpers let startup wait
// the lock out (busy_timeout, applied in database.ts) and retry instead of
// failing hard and stranding the renderer on a never-loading screen.

/**
 * True when an error represents a transient SQLite lock (SQLITE_BUSY /
 * SQLITE_LOCKED), which is safe to wait out and retry. Other errors (missing
 * table, disk full, corruption) are not retryable and must surface.
 */
export function isDatabaseLockedError(err: unknown): boolean {
  if (!err) return false;
  const code = (err as { code?: unknown }).code;
  if (code === 'SQLITE_BUSY' || code === 'SQLITE_LOCKED') return true;
  const message = (err as { message?: unknown }).message;
  if (typeof message === 'string') {
    return (
      message.includes('SQLITE_BUSY') ||
      message.includes('SQLITE_LOCKED') ||
      message.includes('database is locked') ||
      message.includes('database table is locked')
    );
  }
  return false;
}

export interface RetryOptions {
  /** Number of retries after the first attempt. */
  retries: number;
  /** Decides whether a thrown error is worth retrying. */
  isRetryable: (err: unknown) => boolean;
  /** Base backoff in ms; doubles each retry. Defaults to 250ms. */
  baseDelayMs?: number;
  /** Injectable sleep so tests stay deterministic. */
  sleep?: (ms: number) => Promise<void>;
  /** Optional hook for logging each retry. */
  onRetry?: (err: unknown, attempt: number, delayMs: number) => void;
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/**
 * Runs `factory`, retrying with exponential backoff while it throws a
 * retryable error. Re-throws immediately on a non-retryable error, and
 * re-throws the last error once retries are exhausted.
 */
export async function retryAsync<T>(
  factory: () => Promise<T>,
  options: RetryOptions
): Promise<T> {
  const {
    retries,
    isRetryable,
    baseDelayMs = 250,
    sleep = defaultSleep,
    onRetry,
  } = options;

  let attempt = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      return await factory();
    } catch (err) {
      if (attempt >= retries || !isRetryable(err)) {
        throw err;
      }
      const delayMs = baseDelayMs * 2 ** attempt;
      onRetry?.(err, attempt + 1, delayMs);
      attempt += 1;
      await sleep(delayMs);
    }
  }
}
