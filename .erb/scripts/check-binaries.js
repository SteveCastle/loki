#!/usr/bin/env node
/**
 * Pre-build script to ensure platform-specific binaries are available
 * Downloads FFmpeg binaries for all platforms if they don't exist
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

const platform = os.platform();
const binDir = path.join(__dirname, '..', '..', 'src', 'main', 'resources', 'bin');

console.log('Checking platform-specific binaries...');

function downloadBinaries(platformDir, platformName, binaryExtension = '') {
  const platformBinDir = path.join(binDir, platformDir);
  const ffmpegPath = path.join(platformBinDir, `ffmpeg${binaryExtension}`);
  const ffprobePath = path.join(platformBinDir, `ffprobe${binaryExtension}`);
  
  // Check if binaries exist
  const ffmpegExists = fs.existsSync(ffmpegPath);
  const ffprobeExists = fs.existsSync(ffprobePath);
  
  if (!ffmpegExists || !ffprobeExists) {
    console.log(`${platformName} FFmpeg binaries not found. Downloading...`);
    const downloadScript = path.join(platformBinDir, 'download-binaries.sh');
    
    if (!fs.existsSync(downloadScript)) {
      console.error(`Error: download-binaries.sh not found in ${platformBinDir}!`);
      process.exit(1);
    }
    
    try {
      execSync(`bash "${downloadScript}"`, { stdio: 'inherit' });
      console.log(`✓ ${platformName} binaries downloaded successfully`);
    } catch (error) {
      console.error(`Failed to download ${platformName} binaries:`, error.message);
      console.error(`You can manually run: ${downloadScript}`);
      process.exit(1);
    }
  } else {
    console.log(`✓ ${platformName} FFmpeg binaries already present`);
  }
}

// Download binaries for the current platform
if (platform === 'linux') {
  downloadBinaries('linux', 'Linux');
} else if (platform === 'darwin') {
  downloadBinaries('darwin', 'macOS');
} else if (platform === 'win32') {
  downloadBinaries('win32', 'Windows', '.exe');
}

console.log('Platform-specific binaries check complete.\n');
