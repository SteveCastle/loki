import { withTimeout, TimeoutError } from '../main/async-timeout';

const delay = <T>(ms: number, value: T): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(value), ms));

const rejectAfter = (ms: number, err: Error): Promise<never> =>
  new Promise((_, reject) => setTimeout(() => reject(err), ms));

describe('withTimeout', () => {
  it('passes through the value when the promise settles in time', async () => {
    await expect(withTimeout(delay(5, 'ok'), 100, 'fast')).resolves.toBe('ok');
  });

  it('rejects with a TimeoutError when the promise hangs', async () => {
    const hang = new Promise(() => {}); // never settles
    await expect(withTimeout(hang, 10, 'load-files')).rejects.toBeInstanceOf(
      TimeoutError
    );
  });

  it('TimeoutError carries the label, duration, and ETIMEDOUT code', async () => {
    const hang = new Promise(() => {});
    await withTimeout(hang, 10, 'load-db').catch((e: TimeoutError) => {
      expect(e.message).toContain('load-db');
      expect(e.message).toContain('10ms');
      expect(e.code).toBe('ETIMEDOUT');
    });
  });

  it('invokes the onTimeout cleanup hook so callers can kill a child', async () => {
    const hang = new Promise(() => {});
    const onTimeout = jest.fn();
    await withTimeout(hang, 10, 'scan', onTimeout).catch(() => undefined);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it('does not call onTimeout when the promise settles in time', async () => {
    const onTimeout = jest.fn();
    await withTimeout(delay(5, 'ok'), 100, 'fast', onTimeout);
    expect(onTimeout).not.toHaveBeenCalled();
  });

  it('propagates the original rejection (not a timeout) when it loses the race', async () => {
    const boom = new Error('boom');
    await expect(withTimeout(rejectAfter(5, boom), 100, 'x')).rejects.toBe(boom);
  });

  it('disables the timeout for non-positive ms (returns the promise as-is)', async () => {
    // A 30ms promise with ms=0 must still resolve — no timeout armed.
    await expect(withTimeout(delay(30, 'slow'), 0, 'no-timeout')).resolves.toBe(
      'slow'
    );
  });
});
