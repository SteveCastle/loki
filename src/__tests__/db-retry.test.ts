import {
  isDatabaseLockedError,
  retryAsync,
} from '../main/db-retry';

describe('isDatabaseLockedError', () => {
  it('detects SQLITE_BUSY errors by code', () => {
    const err = Object.assign(new Error('database is locked'), {
      code: 'SQLITE_BUSY',
    });
    expect(isDatabaseLockedError(err)).toBe(true);
  });

  it('detects SQLITE_LOCKED errors by code', () => {
    const err = Object.assign(new Error('database table is locked'), {
      code: 'SQLITE_LOCKED',
    });
    expect(isDatabaseLockedError(err)).toBe(true);
  });

  it('detects lock errors by message when code is absent', () => {
    expect(isDatabaseLockedError(new Error('SQLITE_BUSY: database is locked')))
      .toBe(true);
    expect(isDatabaseLockedError(new Error('database is locked'))).toBe(true);
  });

  it('does not treat unrelated errors as lock errors', () => {
    expect(isDatabaseLockedError(new Error('no such table: media'))).toBe(false);
    expect(
      isDatabaseLockedError(
        Object.assign(new Error('disk full'), { code: 'SQLITE_FULL' })
      )
    ).toBe(false);
    expect(isDatabaseLockedError(undefined)).toBe(false);
    expect(isDatabaseLockedError(null)).toBe(false);
  });
});

describe('retryAsync', () => {
  // No-op sleep keeps tests deterministic and fast.
  const noSleep = () => Promise.resolve();

  it('returns the result when the factory succeeds on the first try', async () => {
    let calls = 0;
    const result = await retryAsync(
      async () => {
        calls += 1;
        return 'ok';
      },
      { retries: 3, isRetryable: () => true, sleep: noSleep }
    );
    expect(result).toBe('ok');
    expect(calls).toBe(1);
  });

  it('retries a retryable failure and eventually succeeds', async () => {
    let calls = 0;
    const result = await retryAsync(
      async () => {
        calls += 1;
        if (calls < 3) {
          throw Object.assign(new Error('locked'), { code: 'SQLITE_BUSY' });
        }
        return 'recovered';
      },
      {
        retries: 5,
        isRetryable: isDatabaseLockedError,
        sleep: noSleep,
      }
    );
    expect(result).toBe('recovered');
    expect(calls).toBe(3);
  });

  it('does not retry a non-retryable failure', async () => {
    let calls = 0;
    await expect(
      retryAsync(
        async () => {
          calls += 1;
          throw new Error('no such table');
        },
        { retries: 5, isRetryable: isDatabaseLockedError, sleep: noSleep }
      )
    ).rejects.toThrow('no such table');
    expect(calls).toBe(1);
  });

  it('gives up after exhausting retries and throws the last error', async () => {
    let calls = 0;
    await expect(
      retryAsync(
        async () => {
          calls += 1;
          throw Object.assign(new Error(`busy ${calls}`), {
            code: 'SQLITE_BUSY',
          });
        },
        { retries: 3, isRetryable: () => true, sleep: noSleep }
      )
    ).rejects.toThrow('busy 4');
    // initial attempt + 3 retries
    expect(calls).toBe(4);
  });

  it('waits between retries using the injected sleep with growing delays', async () => {
    const delays: number[] = [];
    let calls = 0;
    await retryAsync(
      async () => {
        calls += 1;
        if (calls < 3) throw Object.assign(new Error('busy'), { code: 'SQLITE_BUSY' });
        return 'done';
      },
      {
        retries: 5,
        isRetryable: () => true,
        baseDelayMs: 100,
        sleep: (ms: number) => {
          delays.push(ms);
          return Promise.resolve();
        },
      }
    );
    // Two failures => two waits, exponential backoff from baseDelayMs.
    expect(delays).toEqual([100, 200]);
  });
});
