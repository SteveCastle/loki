#!/usr/bin/env node
/**
 * Pre-build script to ensure platform-specific binaries are available
 * Downloads Linux FFmpeg binaries if they don't exist
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

const platform = os.platform();
const binDir = path.join(__dirname, '..', '..', 'src', 'main', 'resources', 'bin');

console.log('Checking platform-specific binaries...');

if (platform === 'linux' || process.env.ELECTRON_BUILDER_PLATFORM === 'linux') {
  const linuxBinDir = path.join(binDir, 'linux');
  const ffmpegPath = path.join(linuxBinDir, 'ffmpeg');
  const ffprobePath = path.join(linuxBinDir, 'ffprobe');
  
  // Check if binaries exist
  const ffmpegExists = fs.existsSync(ffmpegPath);
  const ffprobeExists = fs.existsSync(ffprobePath);
  
  if (!ffmpegExists || !ffprobeExists) {
    console.log('Linux FFmpeg binaries not found. Downloading...');
    const downloadScript = path.join(linuxBinDir, 'download-binaries.sh');
    
    if (!fs.existsSync(downloadScript)) {
      console.error('Error: download-binaries.sh not found!');
      process.exit(1);
    }
    
    try {
      execSync(`bash "${downloadScript}"`, { stdio: 'inherit' });
      console.log('✓ Linux binaries downloaded successfully');
    } catch (error) {
      console.error('Failed to download Linux binaries:', error.message);
      console.error('You can manually run: src/main/resources/bin/linux/download-binaries.sh');
      process.exit(1);
    }
  } else {
    console.log('✓ Linux FFmpeg binaries already present');
  }
}

console.log('Platform-specific binaries check complete.\n');
