# Dual-Pool Thumbnail Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent SMB connection drops by limiting concurrent ffmpeg thumbnail generation for network-hosted files, while preserving full speed for local files.

**Architecture:** A new `network-drive-cache.ts` module detects whether a file path lives on a Windows network drive (UNC or mapped drive letter). The existing single-pool preview queue in `media.ts` is replaced with a dual-pool queue that enforces separate concurrency limits for network (3) and local (12) paths.

**Tech Stack:** Node.js `child_process.execSync` for `wmic` detection, TypeScript, Jest for testing.

**Spec:** `docs/superpowers/specs/2026-04-04-dual-pool-thumbnail-concurrency-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/main/network-drive-cache.ts` | Create | Detect and cache which drive letters are network drives. Export `isNetworkPath()`. |
| `src/main/media.ts` | Modify (lines 1-38, 341-342) | Replace single pool with dual pool. Pass `filePath` to `enqueuePreview`. |
| `src/__tests__/network-drive-cache.test.ts` | Create | Unit tests for `isNetworkPath`. |
| `src/__tests__/preview-queue.test.ts` | Create | Unit tests for dual-pool queue behavior. |

---

### Task 1: Network Drive Detection Module

**Files:**
- Create: `src/__tests__/network-drive-cache.test.ts`
- Create: `src/main/network-drive-cache.ts`

- [ ] **Step 1: Write failing tests for `isNetworkPath`**

Create `src/__tests__/network-drive-cache.test.ts`:

```typescript
import { isNetworkPath, _resetCacheForTesting } from '../main/network-drive-cache';
import child_process from 'child_process';

jest.mock('child_process');
const mockExecSync = child_process.execSync as jest.MockedFunction<typeof child_process.execSync>;

// Mock os.platform
jest.mock('os', () => ({ platform: () => 'win32' }));

beforeEach(() => {
  _resetCacheForTesting();
  mockExecSync.mockReset();
});

const WMIC_OUTPUT = `DeviceID  DriveType  \r\nC:        3          \r\nD:        3          \r\nZ:        4          \r\n`;

describe('isNetworkPath', () => {
  it('detects UNC paths as network', () => {
    expect(isNetworkPath('\\\\server\\share\\file.jpg')).toBe(true);
  });

  it('detects forward-slash UNC paths as network', () => {
    expect(isNetworkPath('//server/share/file.jpg')).toBe(true);
  });

  it('detects mapped network drive via wmic', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('Z:\\photos\\cat.jpg')).toBe(true);
  });

  it('detects local drive via wmic', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('C:\\Users\\me\\photo.jpg')).toBe(false);
  });

  it('caches wmic results across calls', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    isNetworkPath('C:\\file1.jpg');
    isNetworkPath('C:\\file2.jpg');
    isNetworkPath('Z:\\file3.jpg');
    expect(mockExecSync).toHaveBeenCalledTimes(1);
  });

  it('re-queries wmic for unknown drive letters', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    isNetworkPath('C:\\file.jpg'); // populates cache

    const UPDATED_OUTPUT = `DeviceID  DriveType  \r\nC:        3          \r\nD:        3          \r\nX:        4          \r\nZ:        4          \r\n`;
    mockExecSync.mockReturnValue(Buffer.from(UPDATED_OUTPUT));
    expect(isNetworkPath('X:\\new-nas\\file.jpg')).toBe(true);
    expect(mockExecSync).toHaveBeenCalledTimes(2);
  });

  it('treats all paths as network when wmic fails', () => {
    mockExecSync.mockImplementation(() => { throw new Error('wmic not found'); });
    expect(isNetworkPath('C:\\local\\file.jpg')).toBe(true);
  });

  it('handles lowercase drive letters', () => {
    mockExecSync.mockReturnValue(Buffer.from(WMIC_OUTPUT));
    expect(isNetworkPath('z:\\photos\\cat.jpg')).toBe(true);
    expect(isNetworkPath('c:\\Users\\me\\photo.jpg')).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/network-drive-cache.test.ts --no-cache`
Expected: FAIL — module `../main/network-drive-cache` does not exist.

- [ ] **Step 3: Implement `network-drive-cache.ts`**

Create `src/main/network-drive-cache.ts`:

```typescript
import { execSync } from 'child_process';
import { platform } from 'os';

// Map of uppercase drive letter (e.g. "C") to boolean (true = network)
let driveTypeCache: Map<string, boolean> | null = null;
let wmicFailed = false;

function queryDriveTypes(): Map<string, boolean> {
  const map = new Map<string, boolean>();
  try {
    const output = execSync('wmic logicaldisk get DeviceID,DriveType', {
      encoding: 'utf-8',
      timeout: 5000,
      windowsHide: true,
    });
    // Parse lines like "C:        3"
    // DriveType 4 = Network
    for (const line of output.split(/\r?\n/)) {
      const match = line.match(/^([A-Za-z]):\s+(\d+)/);
      if (match) {
        const letter = match[1].toUpperCase();
        const driveType = parseInt(match[2], 10);
        map.set(letter, driveType === 4);
      }
    }
    wmicFailed = false;
  } catch {
    wmicFailed = true;
  }
  return map;
}

export function isNetworkPath(filePath: string): boolean {
  // UNC paths are always network
  if (filePath.startsWith('\\\\') || filePath.startsWith('//')) {
    return true;
  }

  // Non-Windows: no network drive detection
  if (platform() !== 'win32') {
    return false;
  }

  // Extract drive letter
  const driveLetter = filePath.match(/^([A-Za-z]):/)?.[1]?.toUpperCase();
  if (!driveLetter) {
    return false;
  }

  // Populate cache on first call
  if (driveTypeCache === null) {
    driveTypeCache = queryDriveTypes();
  }

  // If wmic failed, be conservative — treat everything as network
  if (wmicFailed) {
    return true;
  }

  // If drive letter not in cache, re-query (drive mounted after startup)
  if (!driveTypeCache.has(driveLetter)) {
    driveTypeCache = queryDriveTypes();
    if (wmicFailed) {
      return true;
    }
  }

  return driveTypeCache.get(driveLetter) ?? false;
}

/** Reset internal cache — for testing only */
export function _resetCacheForTesting(): void {
  driveTypeCache = null;
  wmicFailed = false;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest src/__tests__/network-drive-cache.test.ts --no-cache`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/network-drive-cache.ts src/__tests__/network-drive-cache.test.ts
git commit -m "feat: add network drive detection for Windows mapped drives"
```

---

### Task 2: Dual-Pool Preview Queue

**Files:**
- Create: `src/__tests__/preview-queue.test.ts`
- Modify: `src/main/media.ts:1-38` (queue system) and `src/main/media.ts:341-342` (call site)

- [ ] **Step 1: Write failing tests for dual-pool queue**

Create `src/__tests__/preview-queue.test.ts`:

```typescript
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

    // Resolve remaining
    resolvers.forEach((r) => r());
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

    resolvers.forEach((r) => r());
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

    resolvers.forEach((r) => r());
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
```

- [ ] **Step 2: Run tests to verify they pass (these test the queue algorithm in isolation)**

Run: `npx jest src/__tests__/preview-queue.test.ts --no-cache`
Expected: All 4 tests PASS. (These tests validate the algorithm we'll copy into `media.ts`.)

- [ ] **Step 3: Update the queue system in `media.ts`**

In `src/main/media.ts`, replace lines 1-38 with:

```typescript
import { Database } from './database';
import path from 'path';
import crypto from 'crypto';
import * as clipboard from './clipboard';
import type Store from 'electron-store';
import { asyncCreateThumbnail } from './image-processing';
import { getFileType } from '../file-types';
import { IpcMainInvokeEvent, shell } from 'electron';
import fs from 'fs';
import { isNetworkPath } from './network-drive-cache';

const MAX_CONCURRENT_LOCAL = 12;
const MAX_CONCURRENT_NETWORK = 3;
let activeLocalCount = 0;
let activeNetworkCount = 0;
const previewQueue: Array<{
  run: () => Promise<void>;
  network: boolean;
}> = [];

function drainPreviewQueue() {
  let i = 0;
  while (i < previewQueue.length) {
    const item = previewQueue[i];
    const hasCapacity = item.network
      ? activeNetworkCount < MAX_CONCURRENT_NETWORK
      : activeLocalCount < MAX_CONCURRENT_LOCAL;

    if (hasCapacity) {
      previewQueue.splice(i, 1);
      if (item.network) activeNetworkCount++;
      else activeLocalCount++;

      item.run().finally(() => {
        if (item.network) activeNetworkCount--;
        else activeLocalCount--;
        drainPreviewQueue();
      });
    } else {
      i++;
    }
  }
}

function enqueuePreview<T>(filePath: string, fn: () => Promise<T>): Promise<T> {
  const network = isNetworkPath(filePath);
  return new Promise<T>((resolve, reject) => {
    previewQueue.push({
      run: () => fn().then(resolve, reject),
      network,
    });
    drainPreviewQueue();
  });
}
```

- [ ] **Step 4: Update the `fetchMediaPreview` call site**

In `src/main/media.ts`, change the `fetchMediaPreview` function (around line 341-342) from:

```typescript
    return enqueuePreview(async () => {
      const filePath = args[0];
```

to:

```typescript
    const filePath = args[0];
    return enqueuePreview(filePath, async () => {
```

The full function should read:

```typescript
const fetchMediaPreview =
  (db: Database, store: Store) =>
  (_: IpcMainInvokeEvent, args: FetchMediaPreviewInput) => {
    const filePath = args[0];
    return enqueuePreview(filePath, async () => {
      const cache = args[1] || 'thumbnail_path_600';
      const timeStamp = args[2] || 0;
      const userHomeDirectory = require('os').homedir();
      const defaultBasePath = path.join(
        path.join(userHomeDirectory, '.lowkey')
      );
      const dbPath = store.get('dbPath', defaultBasePath) as string;
      const regenerateMediaCache = store.get(
        'regenerateMediaCache',
        false
      ) as boolean;
      const basePath = path.dirname(dbPath);
      const thumbnailFullPath = getMediaCachePath(
        filePath,
        basePath,
        cache,
        timeStamp
      );

      const thumbnailExists = await checkIfMediaCacheExists(thumbnailFullPath);
      if (!thumbnailExists || regenerateMediaCache) {
        await asyncCreateThumbnail(filePath, basePath, cache, timeStamp);
      }
      return thumbnailFullPath;
    });
  };
```

- [ ] **Step 5: Run all tests**

Run: `npx jest --no-cache`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/main/media.ts src/__tests__/preview-queue.test.ts
git commit -m "feat: dual-pool thumbnail queue — separate limits for network vs local"
```
