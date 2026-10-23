import { execSync } from 'child_process';
import { platform } from 'os';

// Map of uppercase drive letter (e.g. "C") to boolean (true = network)
let driveTypeCache: Map<string, boolean> | null = null;
let queryFailed = false;
let lastQueryAt = 0;

// An unknown letter re-queries (a drive mounted after startup), but no more
// often than this: every probe is a synchronous spawn on the main process.
const REQUERY_INTERVAL_MS = 30_000;

function parseWmic(output: string): Map<string, boolean> {
  // Parse lines like "C:        3"
  // DriveType 4 = Network
  const map = new Map<string, boolean>();
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z]):\s+(\d+)/);
    if (match) {
      map.set(match[1].toUpperCase(), parseInt(match[2], 10) === 4);
    }
  }
  return map;
}

// `net use` lists only network mappings ("OK  Z:  \\server\share  ..."), so
// every letter it doesn't name is local. Unlike wmic (removed from Windows
// 11 24H2+) it ships everywhere and answers in ~50 ms.
function parseNetUse(output: string): Map<string, boolean> {
  const map = new Map<string, boolean>();
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/\s([A-Za-z]):\s+\\\\/);
    if (match) map.set(match[1].toUpperCase(), true);
  }
  return map;
}

function run(cmd: string): string {
  return execSync(cmd, {
    encoding: 'utf-8',
    timeout: 5000,
    windowsHide: true,
  }).toString();
}

function queryDriveTypes(): Map<string, boolean> {
  lastQueryAt = Date.now();
  try {
    const map = parseWmic(run('wmic logicaldisk get DeviceID,DriveType'));
    queryFailed = false;
    return map;
  } catch {
    // wmic missing: fall through to net use
  }
  try {
    const map = parseNetUse(run('net use'));
    // net use is authoritative for every letter: absent means local.
    for (const letter of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {
      if (!map.has(letter)) map.set(letter, false);
    }
    queryFailed = false;
    return map;
  } catch {
    queryFailed = true;
    return new Map();
  }
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

  // If re-query is due for an unknown letter (drive mounted after startup)
  if (
    !driveTypeCache.has(driveLetter) &&
    Date.now() - lastQueryAt >= REQUERY_INTERVAL_MS
  ) {
    driveTypeCache = queryDriveTypes();
  }

  // If every probe failed, be conservative — treat everything as network
  if (queryFailed) {
    return true;
  }

  return driveTypeCache.get(driveLetter) ?? false;
}

/** Reset internal cache — for testing only */
export function _resetCacheForTesting(): void {
  driveTypeCache = null;
  queryFailed = false;
  lastQueryAt = 0;
}
