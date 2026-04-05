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
    for (const line of output.toString().split(/\r?\n/)) {
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
