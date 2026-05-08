import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { dependencies } from '../../release/app/package.json';
import { devDependencies } from '../../package.json';
import webpackPaths from '../configs/webpack.paths';

if (
  Object.keys(dependencies || {}).length > 0 &&
  fs.existsSync(webpackPaths.appNodeModulesPath)
) {
  // Fast path: re-run each native dep's own install hook. The hooks in this
  // project (notably sqlite3's `prebuild-install -r napi || node-gyp rebuild`)
  // try to download a prebuilt NAPI binary first and only fall back to
  // compiling from C++ source if the download fails. NAPI binaries are
  // forward-compatible across Node/Electron versions, so when the prebuilt
  // is found we don't need an Electron-specific ABI rebuild — the binary
  // works in Electron directly. This lets contributors without MSVC C++
  // build tools install the project successfully.
  let usedPrebuilt = true;
  for (const dep of Object.keys(dependencies)) {
    const depDir = path.join(webpackPaths.appNodeModulesPath, dep);
    if (!fs.existsSync(path.join(depDir, 'binding.gyp'))) continue;
    try {
      execSync('npm rebuild', { cwd: depDir, stdio: 'inherit' });
    } catch (err) {
      console.warn(
        `[postinstall] prebuilt install for ${dep} failed; falling back to electron-rebuild (requires C++ build tools).`
      );
      usedPrebuilt = false;
      break;
    }
  }

  if (!usedPrebuilt) {
    // Compile from source against Electron's ABI. Requires Visual Studio
    // C++ Build Tools on Windows, Xcode CLT on macOS, gcc/g++ on Linux.
    const electronVersion = devDependencies?.electron || '39.2.7';
    const cleanVersion = electronVersion.replace(/^[\^~]/, '');

    const electronRebuildCmd =
      `../../node_modules/.bin/electron-rebuild --force --types prod,dev,optional --module-dir . --version ${cleanVersion}`;
    const cmd =
      process.platform === 'win32'
        ? electronRebuildCmd.replace(/\//g, '\\')
        : electronRebuildCmd;
    execSync(cmd, {
      cwd: webpackPaths.appPath,
      stdio: 'inherit',
    });
  }
}
