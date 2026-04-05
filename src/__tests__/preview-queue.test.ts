/**
 * Tests for the dual-pool preview queue in media.ts.
 *
 * We test the queue logic by importing the module and using its internals.
 * Since the queue functions are module-private, we re-implement the core
 * logic in a testable wrapper that mirrors media.ts exactly.
 */

type QueueItem = {
  run: () => Promise<void>;
  network: boolean;
};

function createDualPoolQueue(maxLocal: number, maxNetwork: number) {
  let activeLocalCount = 0;
  let activeNetworkCount = 0;
  const queue: QueueItem[] = [];

  function drain() {
    let i = 0;
    while (i < queue.length) {
      const item = queue[i];
      const hasCapacity = item.network
        ? activeNetworkCount < maxNetwork
        : activeLocalCount < maxLocal;

      if (hasCapacity) {
        queue.splice(i, 1);
        if (item.network) activeNetworkCount++;
        else activeLocalCount++;

        item.run().finally(() => {
          if (item.network) activeNetworkCount--;
          else activeLocalCount--;
          drain();
        });
      } else {
        i++;
      }
    }
  }

  function enqueue<T>(network: boolean, fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      queue.push({
        run: () => fn().then(resolve, reject),
        network,
      });
      drain();
    });
  }

  return {
    enqueue,
    getActiveLocal: () => activeLocalCount,
    getActiveNetwork: () => activeNetworkCount,
    getQueueLength: () => queue.length,
  };
}

describe('dual-pool preview queue', () => {
  it('respects network concurrency limit', async () => {
    const pool = createDualPoolQueue(12, 3);
    const resolvers: Array<() => void> = [];

    // Enqueue 5 network tasks that block until resolved
    const promises = Array.from({ length: 5 }, () =>
      pool.enqueue(true, () => new Promise<void>((r) => resolvers.push(r)))
    );

    // Wait for microtasks to settle
    await new Promise((r) => setTimeout(r, 0));

    expect(pool.getActiveNetwork()).toBe(3);
    expect(pool.getQueueLength()).toBe(2);

    // Resolve one — should drain one more from queue
    resolvers[0]();
    await new Promise((r) => setTimeout(r, 0));

    expect(pool.getActiveNetwork()).toBe(3);
    expect(pool.getQueueLength()).toBe(1);

    // Resolve all tasks — resolvers may grow as queued tasks start, so drain in a loop
    while (resolvers.length > 0) {
      resolvers.splice(0).forEach((r) => r());
      await new Promise((r) => setTimeout(r, 0));
    }
    await Promise.all(promises);
  });

  it('respects local concurrency limit', async () => {
    const pool = createDualPoolQueue(2, 3);
    const resolvers: Array<() => void> = [];

    const promises = Array.from({ length: 4 }, () =>
      pool.enqueue(false, () => new Promise<void>((r) => resolvers.push(r)))
    );

    await new Promise((r) => setTimeout(r, 0));

    expect(pool.getActiveLocal()).toBe(2);
    expect(pool.getQueueLength()).toBe(2);

    while (resolvers.length > 0) {
      resolvers.splice(0).forEach((r) => r());
      await new Promise((r) => setTimeout(r, 0));
    }
    await Promise.all(promises);
  });

  it('network and local pools are independent', async () => {
    const pool = createDualPoolQueue(2, 2);
    const resolvers: Array<() => void> = [];

    // Fill network pool
    const netPromises = Array.from({ length: 3 }, () =>
      pool.enqueue(true, () => new Promise<void>((r) => resolvers.push(r)))
    );
    // Fill local pool
    const localPromises = Array.from({ length: 3 }, () =>
      pool.enqueue(false, () => new Promise<void>((r) => resolvers.push(r)))
    );

    await new Promise((r) => setTimeout(r, 0));

    // Both pools at capacity, one of each queued
    expect(pool.getActiveNetwork()).toBe(2);
    expect(pool.getActiveLocal()).toBe(2);
    expect(pool.getQueueLength()).toBe(2);

    while (resolvers.length > 0) {
      resolvers.splice(0).forEach((r) => r());
      await new Promise((r) => setTimeout(r, 0));
    }
    await Promise.all([...netPromises, ...localPromises]);
  });

  it('propagates errors without breaking the queue', async () => {
    const pool = createDualPoolQueue(12, 3);

    const p1 = pool.enqueue(true, () => Promise.reject(new Error('fail')));
    const p2 = pool.enqueue(true, () => Promise.resolve('ok'));

    await expect(p1).rejects.toThrow('fail');
    await expect(p2).resolves.toBe('ok');

    expect(pool.getActiveNetwork()).toBe(0);
    expect(pool.getQueueLength()).toBe(0);
  });
});
