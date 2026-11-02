// The library query runs on a per-window interruptible connection ("lane"),
// latest-wins: a new request or a cancel must actually STOP a running scan
// (a substring path search over millions of rows takes seconds), not let it
// finish in the background and hold the connection. Exercised against real
// SQLite with a deliberately slow statement.
jest.mock('../main/queryLog', () => ({ logQuery: jest.fn() }));

// eslint-disable-next-line import/first
import {
  QUERY_CANCELLED,
  cancelMediaQuery,
  closeAllMediaQueryLanes,
  runMediaQuery,
  setMediaQueryDbPath,
} from '../main/media-query-lanes';
// eslint-disable-next-line import/first
import { Database } from '../main/database';

// Several seconds of pure CPU in SQLite; no tables needed.
const SLOW_SQL = `WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 200000000)
  SELECT count(*) AS n FROM c`;
const FAST_SQL = 'SELECT 42 AS n';

function fakeSender(id: number) {
  const listeners: Record<string, () => void> = {};
  return {
    id,
    once: (event: string, cb: () => void) => {
      listeners[event] = cb;
    },
    destroy: () => listeners.destroyed?.(),
  } as any;
}

describe('media query lanes', () => {
  let shared: Database;

  beforeAll(async () => {
    shared = new Database(':memory:');
    await shared.ready;
  });

  afterAll(async () => {
    await shared.close();
  });

  beforeEach(() => {
    // Each lane opens its own connection; ':memory:' is fine because the
    // statements under test need no tables.
    setMediaQueryDbPath(`:memory:`);
  });

  afterEach(async () => {
    await closeAllMediaQueryLanes();
    setMediaQueryDbPath('');
  });

  it('a newer request interrupts the running one (latest wins)', async () => {
    const sender = fakeSender(1);
    const started = Date.now();
    const slow = runMediaQuery(sender, shared, SLOW_SQL, []);
    // Let the slow statement get going.
    await new Promise((r) => setTimeout(r, 100));
    const fast = runMediaQuery(sender, shared, FAST_SQL, []);

    await expect(fast).resolves.toEqual([{ n: 42 }]);
    await expect(slow).resolves.toBe(QUERY_CANCELLED);
    // Interrupted, not run to completion.
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it('cancel stops the running query', async () => {
    const sender = fakeSender(2);
    const started = Date.now();
    const slow = runMediaQuery(sender, shared, SLOW_SQL, []);
    await new Promise((r) => setTimeout(r, 100));
    await cancelMediaQuery(sender);
    await expect(slow).resolves.toBe(QUERY_CANCELLED);
    expect(Date.now() - started).toBeLessThan(2000);

    // The lane is reusable straight after.
    await expect(runMediaQuery(sender, shared, FAST_SQL, [])).resolves.toEqual(
      [{ n: 42 }]
    );
  });

  it("one window's cancel never touches another window's search", async () => {
    const a = fakeSender(3);
    const b = fakeSender(4);
    const slowA = runMediaQuery(a, shared, SLOW_SQL, []);
    await new Promise((r) => setTimeout(r, 100));
    // B's lane is its own connection: not queued behind A's scan.
    const started = Date.now();
    await expect(runMediaQuery(b, shared, FAST_SQL, [])).resolves.toEqual([
      { n: 42 },
    ]);
    expect(Date.now() - started).toBeLessThan(500);
    await cancelMediaQuery(b); // nothing running on B: a no-op
    await cancelMediaQuery(a);
    await expect(slowA).resolves.toBe(QUERY_CANCELLED);
  });

  it('a burst of requests runs only the last one', async () => {
    const sender = fakeSender(5);
    const first = runMediaQuery(sender, shared, SLOW_SQL, []);
    await new Promise((r) => setTimeout(r, 50));
    const second = runMediaQuery(sender, shared, SLOW_SQL, []);
    const third = runMediaQuery(sender, shared, 'SELECT 3 AS n', []);
    await expect(third).resolves.toEqual([{ n: 3 }]);
    await expect(first).resolves.toBe(QUERY_CANCELLED);
    await expect(second).resolves.toBe(QUERY_CANCELLED);
  });

  it('real SQL errors still reject', async () => {
    const sender = fakeSender(6);
    await expect(
      runMediaQuery(sender, shared, 'SELECT * FROM no_such_table', [])
    ).rejects.toThrow(/no such table/);
  });

  it('falls back to the shared connection before a library is open', async () => {
    setMediaQueryDbPath('');
    const sender = fakeSender(7);
    await expect(runMediaQuery(sender, shared, FAST_SQL, [])).resolves.toEqual(
      [{ n: 42 }]
    );
  });
});
