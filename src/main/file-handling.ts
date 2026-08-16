const path = require('path');

export function isValidFilePath(filePath: string) {
  try {
    // Normalize the file path to resolve any redundant navigation elements
    const normalizedPath = path.normalize(filePath);

    // Check if the normalized path is an absolute path
    if (!path.isAbsolute(normalizedPath)) {
      // You might not need this check if you accept relative paths
      throw new Error('The path is not absolute.');
    }

    // Check if the normalized path is the same as the input file path
    if (normalizedPath !== filePath) {
      throw new Error('The path contains redundant navigation elements.');
    }

    // Check if the path contains any invalid characters
    // eslint-disable-next-line no-control-regex
    if (/^[\u0000-\u001F\u007F<>:"|?*]+$/.test(filePath)) {
      throw new Error('The path contains invalid characters.');
    }

    return true;
  } catch (error: any) {
    console.error(error.message);
    return false;
  }
}

// What a second instance was asked to open. Its argv arrives with Chromium
// switches mixed in, so scan from the END (the user's file follows the exe and
// any dev-mode app path) for the last switch-free argument that names a real
// file or directory. `startIndex` skips the exe (packaged: 1) or the exe plus
// the app path (dev: 2), mirroring how the boot path reads process.argv.
export function filePathFromArgv(argv: string[], startIndex: number): string {
  const fs = require('fs');
  for (let i = argv.length - 1; i >= startIndex; i--) {
    const arg = argv[i];
    if (typeof arg !== 'string' || arg === '' || arg.startsWith('-')) continue;
    if (!isValidFilePath(arg)) continue;
    try {
      if (fs.existsSync(arg)) return arg;
    } catch {
      // treat unreadable candidates as non-matches
    }
  }
  return '';
}

export async function deleteFile(filePath: string) {
  try {
    const fs = require('fs').promises;
    await fs.unlink(filePath);
    return filePath;
  } catch (error: any) {
    console.error(error.message);
    return false;
  }
}
