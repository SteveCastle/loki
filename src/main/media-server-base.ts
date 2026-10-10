import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

// Base URL of the local Lowkey Media Server. The server's port is
// configurable (config.json "port" / LOWKEY_PORT env), so discover it the
// same way lokictl does: LOWKEY_PORT env > the server's own config.json >
// the server's compiled-in default (10111, "L0K1"). Resolved once at preload
// time — the renderer reads window.electron.mediaServerBase synchronously.
const DEFAULT_MEDIA_SERVER_PORT = 10111;

export function mediaServerConfigPath(): string {
  // Mirrors the Go server's platform.GetDataDir() per OS
  // (AppName "lowkey-media-viewer" / AppDisplayName "Lowkey Media Viewer").
  switch (process.platform) {
    case 'win32':
      return process.env.APPDATA
        ? path.join(process.env.APPDATA, 'Lowkey Media Viewer', 'config.json')
        : path.join(os.homedir(), '.lowkey-media-viewer', 'config.json');
    case 'darwin':
      return path.join(
        os.homedir(),
        'Library',
        'Application Support',
        'Lowkey Media Viewer',
        'config.json'
      );
    default:
      return path.join(
        process.env.XDG_DATA_HOME ||
          path.join(os.homedir(), '.local', 'share'),
        'lowkey-media-viewer',
        'config.json'
      );
  }
}

export function detectMediaServerBase(): string {
  let port = 0;
  const envPort = parseInt(process.env.LOWKEY_PORT || '', 10);
  if (envPort > 0 && envPort <= 65535) {
    port = envPort;
  } else {
    try {
      const cfg = JSON.parse(fs.readFileSync(mediaServerConfigPath(), 'utf8'));
      if (
        typeof cfg.port === 'number' &&
        cfg.port > 0 &&
        cfg.port <= 65535
      ) {
        port = cfg.port;
      }
    } catch {
      // no server config readable — fall through to the default
    }
  }
  return `http://localhost:${port || DEFAULT_MEDIA_SERVER_PORT}`;
}
