# Dual-Pool Thumbnail Concurrency

## Problem

When the list view loads, visible items trigger thumbnail generation via ffmpeg. The current system allows up to 24 concurrent ffmpeg processes (`MAX_CONCURRENT_PREVIEWS = 24` in `src/main/media.ts`). Each process opens the source media file to extract a frame.

When source files live on an SMB network share (common for NAS-backed media libraries), 24 simultaneous reads overwhelm the SMB connection, causing it to drop. Once dropped, no files on that share are accessible until the connection recovers.

Libraries can be all-local, all-network, or mixed within the same list view.

## Solution

Replace the single concurrency pool with two independent pools — one for network paths, one for local paths — sharing one queue. Network paths get a conservative limit; local paths keep a higher limit.

### Network Drive Detection

**New file: `src/main/network-drive-cache.ts`**

Exports a single function:

```typescript
function isNetworkPath(filePath: string): boolean
```

Detection logic:
1. **UNC paths** (`\\server\...` or `//server/...`): always network, no lookup needed.
2. **Drive letter paths** (`X:\...`): look up the drive letter in a cached map of drive types.

The cache is populated by running:

```
wmic logicaldisk get DeviceID,DriveType
```

via `child_process.execSync`. `DriveType` value `4` means network drive. The output is parsed into a `Map<string, boolean>` mapping uppercase drive letters to whether they are network drives.

**Cache lifecycle:**
- Populated lazily on first call to `isNetworkPath`.
- If a drive letter is not found in the cache (drive mounted after startup), re-run the detection command and update the cache. This re-detection happens at most once per unknown drive letter.
- On non-Windows platforms, `isNetworkPath` always returns `false` (network drive detection is Windows-only for now, as stated in requirements).

**Fallback:** If `wmic` fails for any reason, treat all paths as network (conservative — protects SMB at the cost of slower local thumbnails until next successful detection).

### Queue Changes

**File: `src/main/media.ts`**

Replace:

```typescript
const MAX_CONCURRENT_PREVIEWS = 24;
let activePreviewCount = 0;
```

With:

```typescript
const MAX_CONCURRENT_LOCAL = 12;
const MAX_CONCURRENT_NETWORK = 3;
let activeLocalCount = 0;
let activeNetworkCount = 0;
```

Queue items gain a `network` boolean tag, set at enqueue time by calling `isNetworkPath(filePath)`:

```typescript
const previewQueue: Array<{
  run: () => Promise<void>;
  network: boolean;
}> = [];
```

**`enqueuePreview`** gains the file path so it can classify:

```typescript
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

**`drainPreviewQueue`** iterates the queue and picks items that have capacity in their respective pool:

```typescript
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
```

This ensures network items don't block behind local items that have capacity, and vice versa.

### Call Site Update

In `fetchMediaPreview`, pass the file path to `enqueuePreview`:

```typescript
// Before:
return enqueuePreview(async () => { ... });

// After:
return enqueuePreview(filePath, async () => { ... });
```

## What Does NOT Change

- **`src/main/image-processing.ts`** — worker pool unchanged.
- **`src/main/image-processing-worker.js`** — ffmpeg execution unchanged.
- **Renderer side** — no changes. IPC interface unchanged.
- **React Query caching and visibility delay** — unchanged.
- **`regenerateThumbnail` handler** — does not go through the preview queue currently, and single manual regenerations don't cause the SMB flood. No change needed.

## Files Changed

| File | Change |
|------|--------|
| `src/main/network-drive-cache.ts` | **New.** `isNetworkPath()` with cached `wmic` detection. |
| `src/main/media.ts` | Replace single pool with dual pool. Update `enqueuePreview` signature. |

## Testing

- **Unit test `isNetworkPath`:** UNC paths return true. Local drive letters return false. Unknown drive letters trigger re-detection.
- **Unit test queue:** Verify network items respect `MAX_CONCURRENT_NETWORK` while local items respect `MAX_CONCURRENT_LOCAL` independently.
- **Manual test:** Open a library with files on a mapped network drive. Confirm thumbnails generate without dropping the SMB connection. Confirm local files still generate at full speed.
