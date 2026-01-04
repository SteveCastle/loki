import { execSync } from 'child_process';
import { platform } from 'os';
import path from 'path';
import fs from 'fs';

const isWindows = platform() === 'win32';

function killElectronProcesses() {
  try {
    if (isWindows) {
      // Kill electron processes on Windows
      console.log('Killing any running Electron processes...');
      try {
        execSync('taskkill /F /IM electron.exe /T 2>nul', { stdio: 'ignore' });
      } catch (e) {
        // Ignore if no processes found
      }
      try {
        execSync('taskkill /F /IM "Lowkey Media Viewer.exe" /T 2>nul', { stdio: 'ignore' });
      } catch (e) {
        // Ignore if no processes found
      }
    } else {
      // Kill electron processes on Unix-like systems
      console.log('Killing any running Electron processes...');
      try {
        // Use pgrep to find electron processes and kill them individually
        // This is safer than pkill as it doesn't risk killing the build process
        const electronPids = execSync('pgrep -f "electron" || true', { encoding: 'utf8' }).trim();
        if (electronPids) {
          const pids = electronPids.split('\n').filter(pid => pid.trim());
          const currentPid = process.pid;
          const parentPid = process.ppid;
          
          pids.forEach(pid => {
            const numPid = parseInt(pid, 10);
            // Don't kill the current process, parent process, or process group leaders
            if (numPid && numPid !== currentPid && numPid !== parentPid && numPid !== 1) {
              try {
                // Check if it's actually an electron process (not this script)
                const cmdline = execSync(`ps -p ${numPid} -o args= 2>/dev/null || true`, { encoding: 'utf8' }).trim();
                // Only kill if it's an actual electron binary or Electron app
                if (cmdline && (cmdline.includes('/electron ') || cmdline.includes('/electron$') || cmdline.includes('Lowkey Media Viewer'))) {
                  execSync(`kill ${numPid} 2>/dev/null || true`, { stdio: 'ignore' });
                }
              } catch (e) {
                // Ignore errors - process might have already exited
              }
            }
          });
        }
      } catch (e) {
        // Ignore errors
      }
    }
  } catch (error) {
    console.warn('Could not kill Electron processes:', error.message);
  }
}

function waitForFileRelease(filePath, maxAttempts = 10, delayMs = 500) {
  if (!fs.existsSync(filePath)) {
    return true; // File doesn't exist, so it's "released"
  }

  for (let i = 0; i < maxAttempts; i++) {
    try {
      // Try to open the file in write mode to check if it's locked
      const fd = fs.openSync(filePath, 'r+');
      fs.closeSync(fd);
      return true; // File is not locked
    } catch (error) {
      if (error.code === 'EBUSY' || error.code === 'EPERM') {
        // File is still locked, wait and retry
        if (i < maxAttempts - 1) {
          console.log(`File ${filePath} is locked, waiting ${delayMs}ms... (attempt ${i + 1}/${maxAttempts})`);
          const start = Date.now();
          while (Date.now() - start < delayMs) {
            // Busy wait
          }
        }
      } else {
        // Different error, might be okay
        return true;
      }
    }
  }
  return false; // File is still locked after all attempts
}

function cleanupBuildDirectory() {
  const buildPath = path.join(process.cwd(), 'release', 'build');
  const asarPath = path.join(buildPath, 'win-unpacked', 'resources', 'app.asar');

  if (fs.existsSync(asarPath)) {
    console.log(`Checking if ${asarPath} is locked...`);
    const released = waitForFileRelease(asarPath);
    if (!released) {
      console.warn(`Warning: ${asarPath} is still locked. The build may fail.`);
      console.warn('Please close any applications using files in the build directory.');
    }
  }
}

// Main execution
console.log('Pre-build cleanup: Ensuring build directory is ready...');
killElectronProcesses();

// Wait a bit for processes to fully terminate
const waitTime = isWindows ? 1000 : 500;
console.log(`Waiting ${waitTime}ms for processes to terminate...`);
const start = Date.now();
while (Date.now() - start < waitTime) {
  // Busy wait
}

cleanupBuildDirectory();
console.log('Pre-build cleanup complete.');


