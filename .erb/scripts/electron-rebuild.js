import { execSync } from 'child_process';
import fs from 'fs';
import { dependencies } from '../../release/app/package.json';
import { devDependencies } from '../../package.json';
import webpackPaths from '../configs/webpack.paths';

if (
  Object.keys(dependencies || {}).length > 0 &&
  fs.existsSync(webpackPaths.appNodeModulesPath)
) {
  // Get electron version from root package.json
  const electronVersion = devDependencies?.electron || '39.2.7';
  // Remove the ^ or ~ prefix if present
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
