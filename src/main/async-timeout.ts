// Wrap a promise so it can't hang forever. Used to bound fallible async loads
// (DB open, filesystem scans) so a stalled operation surfaces as a rejection —
// which the renderer's XState machine can route through its onError transitions
// — instead of stranding the app on a permanent loading spinner.

export class TimeoutError extends Error {
  readonly code = 'ETIMEDOUT';

  constructor(label: string, ms: number) {
    super(`${label} timed out after ${ms}ms`);
    this.name = 'TimeoutError';
  }
}

/**
 * Rejects with TimeoutError if `promise` doesn't settle within `ms`. The
 * underlying work is NOT cancelled (JS can't cancel a promise) — callers that
 * own a cancellable resource (e.g. a child process) should pass `onTimeout` to
 * clean it up. The timer is always cleared so the process can exit cleanly.
 */
export function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  label: string,
  onTimeout?: () => void
): Promise<T> {
  if (!Number.isFinite(ms) || ms <= 0) return promise;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      try {
        onTimeout?.();
      } catch {
        // cleanup best-effort
      }
      reject(new TimeoutError(label, ms));
    }, ms);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}
