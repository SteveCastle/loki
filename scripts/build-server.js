#!/usr/bin/env node
/**
 * Build script for the Loki media server.
 * Builds the UI, copies it into media-server/loki-static, then compiles
 * the Go binary with the SPA embedded.
 *
 * Works on macOS, Windows, and Linux.
 *
 * Usage:  node scripts/build-server.js
 */
const { execSync } = require('child_process');
const path = require('path');
const process = require('process');

const ROOT = path.resolve(__dirname, '..');
const SERVER_DIR = path.join(ROOT, 'media-server');

function run(cmd, opts = {}) {
  console.log(`> ${cmd}`);
  execSync(cmd, { stdio: 'inherit', cwd: ROOT, ...opts });
}

// 1. Stop any running media-server process (best-effort, ignore errors)
console.log('\n--- Stopping running server (if any) ---');
try {
  if (process.platform === 'win32') {
    execSync('taskkill /F /IM media-server.exe', { stdio: 'ignore' });
  } else {
    execSync('pkill -f media-server || true', { stdio: 'ignore' });
  }
} catch {
  // Not running — that's fine
}

// 2. Build the renderer (webpack)
console.log('\n--- Building UI ---');
run('npm run build:web');

// 3. Build the Go binary
console.log('\n--- Building Go server ---');
const ext = process.platform === 'win32' ? '.exe' : '';
// On Windows the server already presents itself via a system tray icon, so
// the default console subsystem just produces a redundant black CLI window
// when the user launches the .exe. `-H=windowsgui` switches the PE subsystem
// to GUI so Windows won't allocate a console for the process. No effect on
// macOS or Linux — those linkers ignore the flag.
const ldflags =
  process.platform === 'win32' ? ' -ldflags="-H=windowsgui"' : '';
run(`go build${ldflags} -o media-server${ext} .`, { cwd: SERVER_DIR });

console.log(`\n✓ Server built: media-server/media-server${ext}`);
