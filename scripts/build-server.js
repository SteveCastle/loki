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
const fs = require('fs');
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
const ldflags = process.platform === 'win32' ? '' : '';
run(`go build${ldflags} -o media-server${ext} .`, { cwd: SERVER_DIR });

console.log(`\n✓ Server built: media-server/media-server${ext}`);

// 4. Build the bundled worker binaries (media-server/bin/). The server spawns
// these as subprocesses and evolves their CLI flags in lockstep with its own
// task code — an out-of-date embed.exe fails every embed/faces job with
// "flag provided but not defined". They ship next to the server, so rebuild
// them whenever the server is rebuilt. Skip with SKIP_BUNDLED=1.
if (process.env.SKIP_BUNDLED !== '1') {
  console.log('\n--- Building bundled worker binaries (bin/) ---');
  // The workers REQUIRE cgo (ONNX runtime C API). Since Go 1.20, a missing C
  // compiler makes CGO_ENABLED silently default to 0 and the build then
  // SUCCEEDS with runtime stubs — every embed/autotag/faces job fails with
  // "built without cgo; ONNX inference unavailable". Forcing CGO_ENABLED=1
  // turns "no compiler on PATH" into a loud build failure instead.
  const cgoEnv = { ...process.env, CGO_ENABLED: '1' };
  try {
    run(`go build -o bin${path.sep}embed${ext} ./cmd/embed`, { cwd: SERVER_DIR, env: cgoEnv });
    run(`go build -o bin${path.sep}onnxtag${ext} ./cmd/onnxtag`, { cwd: SERVER_DIR, env: cgoEnv });
  } catch (err) {
    console.error(
      '\n✗ Worker binaries failed to build. They need cgo, which needs a C ' +
        'compiler on PATH (Windows: mingw-w64 gcc, e.g. C:\\ProgramData\\mingw64\\mingw64\\bin). ' +
        'Fix the toolchain or rerun with SKIP_BUNDLED=1 to keep the existing binaries.'
    );
    throw err;
  }
  // Belt-and-braces: verify neither binary compiled in the no-cgo stub.
  for (const name of ['embed', 'onnxtag']) {
    const bin = fs.readFileSync(path.join(SERVER_DIR, 'bin', `${name}${ext}`));
    if (bin.includes('built without cgo')) {
      throw new Error(
        `bin/${name}${ext} was built WITHOUT cgo (ONNX disabled at runtime). ` +
          'Ensure a C compiler is on PATH and rebuild.'
      );
    }
  }
  console.log(`\n✓ Bundled binaries built (cgo verified): media-server/bin/{embed,onnxtag}${ext}`);
}

// 5. Build the lokictl CLI (ships next to the server binary).
// Skip with SKIP_CLI=1 for a server-only rebuild.
if (process.env.SKIP_CLI !== '1') {
  console.log('\n--- Building lokictl CLI ---');
  run(`go build -o lokictl${ext} ./cmd/lokictl`, { cwd: SERVER_DIR });
  console.log(`\n✓ CLI built: media-server/lokictl${ext}`);
}
