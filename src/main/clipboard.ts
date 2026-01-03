import { exec, execSync } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';

/**
 * Native cross-platform file clipboard operations.
 * Replaces electron-clipboard-ex with platform-specific implementations.
 */

const platform = process.platform;

/**
 * Copy file paths to the system clipboard so they can be pasted in file managers.
 * @param filePaths - Array of absolute file paths to copy
 */
export function writeFilePaths(filePaths: string[]): void {
  if (!filePaths || filePaths.length === 0) {
    return;
  }

  switch (platform) {
    case 'win32':
      writeFilePathsWindows(filePaths);
      break;
    case 'darwin':
      writeFilePathsMac(filePaths);
      break;
    case 'linux':
      writeFilePathsLinux(filePaths);
      break;
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

/**
 * Windows implementation using PowerShell and .NET clipboard APIs.
 * Sets files to clipboard in a format that Windows Explorer can paste.
 */
function writeFilePathsWindows(filePaths: string[]): void {
  // Normalize paths to Windows format
  const normalizedPaths = filePaths.map((p) => path.resolve(p));

  // Build PowerShell script that uses System.Windows.Forms.Clipboard
  const pathsArray = normalizedPaths
    .map((p) => `"${p.replace(/"/g, '`"')}"`)
    .join(',');

  const psScript = `
Add-Type -AssemblyName System.Windows.Forms
$files = New-Object System.Collections.Specialized.StringCollection
$paths = @(${pathsArray})
foreach ($p in $paths) {
    $files.Add($p) | Out-Null
}
[System.Windows.Forms.Clipboard]::SetFileDropList($files)
`;

  try {
    execSync(
      `powershell -NoProfile -NonInteractive -Command "${psScript.replace(/"/g, '\\"').replace(/\n/g, ' ')}"`,
      { windowsHide: true }
    );
  } catch (error) {
    console.error('Failed to copy files to clipboard on Windows:', error);
    throw error;
  }
}

/**
 * macOS implementation using AppleScript and NSPasteboard.
 * Uses osascript to set files on the clipboard for Finder paste.
 */
function writeFilePathsMac(filePaths: string[]): void {
  // Convert to POSIX file references for AppleScript
  const posixFiles = filePaths
    .map((p) => `POSIX file "${path.resolve(p).replace(/"/g, '\\"')}"`)
    .join(', ');

  const appleScript = `
    set theFiles to {${posixFiles}}
    tell application "Finder"
      set the clipboard to theFiles
    end tell
  `;

  try {
    execSync(`osascript -e '${appleScript.replace(/'/g, "'\\''")}'`);
  } catch (error) {
    console.error('Failed to copy files to clipboard on macOS:', error);
    throw error;
  }
}

/**
 * Linux implementation using xclip with proper MIME types.
 * Sets both text/uri-list and gnome-copied-files for broad compatibility.
 */
function writeFilePathsLinux(filePaths: string[]): void {
  // Convert to file:// URIs
  const fileUris = filePaths.map((p) => `file://${path.resolve(p)}`);

  // Create gnome-copied-files format (copy\nuri1\nuri2...)
  const gnomeFormat = 'copy\n' + fileUris.join('\n');

  // Create text/uri-list format
  const uriListFormat = fileUris.join('\n');

  // Try xclip first (most common), then xsel as fallback
  const tmpFile = path.join(os.tmpdir(), `clipboard-${Date.now()}.txt`);

  try {
    // Check if xclip is available
    try {
      execSync('which xclip', { stdio: 'ignore' });

      // Write gnome-copied-files format for file managers like Nautilus, Dolphin
      fs.writeFileSync(tmpFile, gnomeFormat);
      execSync(
        `xclip -selection clipboard -t x-special/gnome-copied-files -i "${tmpFile}"`,
        { stdio: 'ignore' }
      );

      // Also set text/uri-list for other applications
      fs.writeFileSync(tmpFile, uriListFormat);
      execSync(`xclip -selection clipboard -t text/uri-list -i "${tmpFile}"`, {
        stdio: 'ignore',
      });

      return;
    } catch {
      // xclip not available, try xsel
    }

    // Try xsel as fallback
    try {
      execSync('which xsel', { stdio: 'ignore' });

      // xsel doesn't support custom MIME types as well, but we can try text/uri-list
      fs.writeFileSync(tmpFile, uriListFormat);
      execSync(`xsel --clipboard --input < "${tmpFile}"`, {
        stdio: 'ignore',
        shell: '/bin/bash',
      });

      console.warn(
        'Using xsel - file paste may not work in all file managers. Install xclip for better compatibility.'
      );
      return;
    } catch {
      // xsel not available either
    }

    // Try wl-copy for Wayland
    try {
      execSync('which wl-copy', { stdio: 'ignore' });

      // wl-copy supports custom MIME types
      fs.writeFileSync(tmpFile, gnomeFormat);
      execSync(
        `wl-copy --type x-special/gnome-copied-files < "${tmpFile}"`,
        { stdio: 'ignore', shell: '/bin/bash' }
      );

      return;
    } catch {
      // wl-copy not available
    }

    throw new Error(
      'No clipboard tool found. Please install xclip (recommended), xsel, or wl-copy (Wayland).'
    );
  } finally {
    // Clean up temp file
    try {
      if (fs.existsSync(tmpFile)) {
        fs.unlinkSync(tmpFile);
      }
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Read file paths from the system clipboard.
 * @returns Array of file paths, or empty array if clipboard doesn't contain files
 */
export function readFilePaths(): string[] {
  switch (platform) {
    case 'win32':
      return readFilePathsWindows();
    case 'darwin':
      return readFilePathsMac();
    case 'linux':
      return readFilePathsLinux();
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

function readFilePathsWindows(): string[] {
  const psScript = `
Add-Type -AssemblyName System.Windows.Forms
$files = [System.Windows.Forms.Clipboard]::GetFileDropList()
$files | ForEach-Object { Write-Output $_ }
`;

  try {
    const result = execSync(
      `powershell -NoProfile -NonInteractive -Command "${psScript.replace(/"/g, '\\"').replace(/\n/g, ' ')}"`,
      { windowsHide: true, encoding: 'utf8' }
    );
    return result
      .trim()
      .split('\n')
      .map((p) => p.trim())
      .filter((p) => p.length > 0);
  } catch {
    return [];
  }
}

function readFilePathsMac(): string[] {
  const appleScript = `
    tell application "Finder"
      set theClipboard to the clipboard as alias list
      set output to ""
      repeat with anAlias in theClipboard
        set output to output & POSIX path of anAlias & linefeed
      end repeat
      return output
    end tell
  `;

  try {
    const result = execSync(`osascript -e '${appleScript.replace(/'/g, "'\\''")}'`, {
      encoding: 'utf8',
    });
    return result
      .trim()
      .split('\n')
      .map((p) => p.trim())
      .filter((p) => p.length > 0);
  } catch {
    return [];
  }
}

function readFilePathsLinux(): string[] {
  try {
    // Try xclip first
    try {
      execSync('which xclip', { stdio: 'ignore' });
      const result = execSync(
        'xclip -selection clipboard -t text/uri-list -o 2>/dev/null',
        { encoding: 'utf8', shell: '/bin/bash' }
      );
      return parseUriList(result);
    } catch {
      // xclip failed or not available
    }

    // Try wl-paste for Wayland
    try {
      execSync('which wl-paste', { stdio: 'ignore' });
      const result = execSync('wl-paste --type text/uri-list 2>/dev/null', {
        encoding: 'utf8',
        shell: '/bin/bash',
      });
      return parseUriList(result);
    } catch {
      // wl-paste failed or not available
    }

    return [];
  } catch {
    return [];
  }
}

function parseUriList(uriList: string): string[] {
  return uriList
    .trim()
    .split('\n')
    .map((uri) => uri.trim())
    .filter((uri) => uri.startsWith('file://'))
    .map((uri) => decodeURIComponent(uri.replace('file://', '')));
}
