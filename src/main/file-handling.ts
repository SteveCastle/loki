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
