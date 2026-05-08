#!/usr/bin/env node
/**
 * Pre-build script to ensure platform-specific binaries are available.
 * Downloads FFmpeg + UnRAR for the current platform if any are missing.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

const platform = os.platform();
const binDir = path.join(__dirname, '..', '..', 'src', 'main', 'resources', 'bin');

console.log('Checking platform-specific binaries...');

function checkAndDownload(platformDir, platformName, binaryExtension = '') {
  const platformBinDir = path.join(binDir, platformDir);
  const required = ['ffmpeg', 'ffprobe', 'unrar'];
  const missing = required.filter(
    (name) => !fs.existsSync(path.join(platformBinDir, `${name}${binaryExtension}`))
  );

  if (missing.length > 0) {
    console.log(`${platformName} binaries missing: ${missing.join(', ')}. Downloading...`);
    const downloadScript = path.join(platformBinDir, 'download-binaries.sh');

    if (!fs.existsSync(downloadScript)) {
      console.error(`Error: download-binaries.sh not found in ${platformBinDir}!`);
      process.exit(1);
    }

    try {
      // Run via the script's own directory + relative name so we don't
      // hand bash a Windows path with backslashes (it would interpret
      // the backslashes as escape sequences and report the file as
      // missing). cwd handles the path translation cleanly on every
      // platform, including Git Bash on Windows.
      execSync('bash ./download-binaries.sh', {
        stdio: 'inherit',
        cwd: platformBinDir,
      });
      console.log(`✓ ${platformName} binaries downloaded successfully`);
    } catch (error) {
      console.error(`Failed to download ${platformName} binaries:`, error.message);
      console.error(`You can manually run: ${downloadScript}`);
      process.exit(1);
    }
  } else {
    console.log(`✓ ${platformName} binaries already present (ffmpeg, ffprobe, unrar)`);
  }
}

// Download binaries for the current platform
if (platform === 'linux') {
  checkAndDownload('linux', 'Linux');
} else if (platform === 'darwin') {
  checkAndDownload('darwin', 'macOS');
} else if (platform === 'win32') {
  checkAndDownload('win32', 'Windows', '.exe');
}

console.log('Platform-specific binaries check complete.\n');
