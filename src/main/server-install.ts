// One-click install of the Lowkey Media Server from the desktop app. The
// release pipeline already ships a friendly per-OS artifact; this downloads
// the one matching this machine from GitHub, installs it the way a user
// would, starts it, and waits until it answers on its HTTP port.
//
//   Windows  per-user Inno installer, run silently (no UAC, Start Menu entry,
//            start-with-Windows task, uninstaller).
//   macOS    the signed menu-bar .app out of the DMG, copied to /Applications
//            (~/Applications when that isn't writable).
//   Linux    the tar.gz payload unpacked under ~/.local/share.
//
// The viewer always starts the server with --background, so it never pops a
// browser tab; onboarding continues in the viewer's own setup window
// (server-login.ts, 'server-setup'). With the "Start media server with the
// viewer" setting on (the default) the viewer also starts an installed
// server whenever it launches (autoStartServer), and offers an upgrade when
// the running server is older than the viewer.
import { app, ipcMain, net, BrowserWindow } from 'electron';
import { execFile, spawn } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import type Store from 'electron-store';
import { detectMediaServerBase } from './media-server-base';
import { logEvent } from './errorLog';

const RELEASES_API = 'https://api.github.com/repos/stevecastle/loki/releases';
const PROMPTED_KEY = 'serverInstallPrompted';
const INSTALL_PATH_KEY = 'serverInstallPath';
const AUTO_START_KEY = 'autoStartServer';
const MAC_APP_NAME = 'Lowkey Media Server.app';

export type ServerInstallPhase =
  | 'idle'
  | 'resolving'
  | 'downloading'
  | 'installing'
  | 'starting'
  | 'done'
  | 'error';

export interface ServerInstallProgress {
  phase: ServerInstallPhase;
  received?: number;
  total?: number;
  version?: string;
  error?: string;
}

export interface ServerInstallStatus {
  /** A release artifact exists for this OS/arch. */
  supported: boolean;
  running: boolean;
  /** Path of an installed server (exe, or .app on macOS), if one was found. */
  installedPath: string | null;
  /** The first-run prompt was answered (or skipped because a server exists). */
  prompted: boolean;
  /** Version the running server reports ('dev' for local builds). */
  serverVersion: string | null;
  appVersion: string;
  /** The running server is an older release than this viewer. */
  updateAvailable: boolean;
  /** Start the installed server whenever the viewer launches. */
  autoStart: boolean;
  progress: ServerInstallProgress;
}

/** Release asset for a platform/arch, or null when none is published. */
export function serverAssetName(
  platform: NodeJS.Platform,
  arch: string
): string | null {
  switch (platform) {
    case 'win32':
      // x64 only; Windows on ARM runs it under emulation.
      return 'lowkey-media-server-setup-windows-amd64.exe';
    case 'darwin':
      return `lowkey-media-server-darwin-${
        arch === 'arm64' ? 'arm64' : 'amd64'
      }.dmg`;
    case 'linux':
      return arch === 'x64' ? 'lowkey-media-server-linux-amd64.tar.gz' : null;
    default:
      return null;
  }
}

function linuxInstallRoot(): string {
  return path.join(
    process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'),
    'lowkey-media-server'
  );
}

/** Where each OS's install lands, most likely first. */
function candidatePaths(): string[] {
  switch (process.platform) {
    case 'win32':
      return [
        path.join(
          process.env.LOCALAPPDATA ||
            path.join(os.homedir(), 'AppData', 'Local'),
          'Programs',
          'Lowkey Media Server',
          'lowkeymediaserver.exe'
        ),
      ];
    case 'darwin':
      return [
        path.join('/Applications', MAC_APP_NAME),
        path.join(os.homedir(), 'Applications', MAC_APP_NAME),
      ];
    case 'linux':
      return [path.join(linuxInstallRoot(), 'linux-amd64', 'lowkeymediaserver')];
    default:
      return [];
  }
}

function findInstalled(store: Store): string | null {
  const remembered = store.get(INSTALL_PATH_KEY, '') as string;
  for (const p of [remembered, ...candidatePaths()]) {
    if (p && fs.existsSync(p)) return p;
  }
  return null;
}

interface ServerProbe {
  running: boolean;
  version: string | null;
}

/** Any HTTP answer from the configured port means a server is up. */
async function probeServer(timeoutMs = 1500): Promise<ServerProbe> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await net.fetch(`${detectMediaServerBase()}/health`, {
      signal: ctrl.signal,
    });
    let version: string | null = null;
    try {
      const data = (await res.json()) as { version?: unknown };
      if (typeof data.version === 'string') version = data.version;
    } catch {
      // older servers answer without a version
    }
    return { running: true, version };
  } catch {
    return { running: false, version: null };
  } finally {
    clearTimeout(timer);
  }
}

async function isServerRunning(timeoutMs = 1500): Promise<boolean> {
  return (await probeServer(timeoutMs)).running;
}

/**
 * a < b for dotted release versions ("2.35.0" < "2.36.0"). Anything that
 * isn't a release version (a 'dev' build, a server too old to report one)
 * never counts as outdated — only a known-older release offers an upgrade.
 */
export function isOlderVersion(a: string | null, b: string): boolean {
  const parse = (v: string | null) => {
    const s = (v || '').replace(/^v/, '');
    return /^\d+(\.\d+)*$/.test(s) ? s.split('.').map(Number) : null;
  };
  const pa = parse(a);
  const pb = parse(b);
  if (!pa || !pb) return false;
  for (let i = 0; i < Math.max(pa.length, pb.length); i += 1) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d < 0;
  }
  return false;
}

/** Polls until the server is up (or, with up=false, until it's gone). */
async function waitForServer(timeoutMs: number, up = true): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if ((await isServerRunning()) === up) return true;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
  digest?: string | null;
}

async function fetchRelease(
  which: string
): Promise<{ tag: string; assets: ReleaseAsset[] } | null> {
  const res = await net.fetch(`${RELEASES_API}/${which}`, {
    headers: {
      'User-Agent': 'Lowkey-Media-Viewer',
      Accept: 'application/vnd.github.v3+json',
    },
  });
  if (!res.ok) return null;
  const data = (await res.json()) as {
    tag_name?: string;
    assets?: ReleaseAsset[];
  };
  return { tag: data.tag_name || '', assets: data.assets || [] };
}

/** Prefer the server from this app's own release; fall back to the latest. */
async function resolveAsset(
  name: string
): Promise<{ version: string; asset: ReleaseAsset }> {
  for (const which of [`tags/v${app.getVersion()}`, 'latest']) {
    const rel = await fetchRelease(which);
    const asset = rel?.assets.find((a) => a.name === name);
    if (rel && asset) return { version: rel.tag.replace(/^v/, ''), asset };
  }
  throw new Error(`No release publishes ${name}`);
}

async function download(
  asset: ReleaseAsset,
  dest: string,
  onProgress: (received: number, total: number) => void
): Promise<void> {
  const res = await net.fetch(asset.browser_download_url, {
    headers: { 'User-Agent': 'Lowkey-Media-Viewer' },
  });
  if (!res.ok || !res.body) {
    throw new Error(`Download failed: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get('content-length')) || asset.size || 0;
  const hash = crypto.createHash('sha256');
  const out = fs.createWriteStream(dest);
  const reader = res.body.getReader();
  let received = 0;
  let lastReport = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      hash.update(value);
      received += value.byteLength;
      if (!out.write(value)) {
        await new Promise<void>((r) => out.once('drain', () => r()));
      }
      const now = Date.now();
      if (now - lastReport > 150) {
        lastReport = now;
        onProgress(received, total);
      }
    }
  } finally {
    await new Promise<void>((r) => out.end(() => r()));
  }
  onProgress(received, total);
  // GitHub publishes a sha256 for every release asset; check it when present.
  const expected = asset.digest?.startsWith('sha256:')
    ? asset.digest.slice('sha256:'.length).toLowerCase()
    : null;
  if (expected && hash.digest('hex') !== expected) {
    throw new Error('Downloaded server failed its checksum');
  }
}

function run(cmd: string, args: string[], timeoutMs = 10 * 60_000) {
  return new Promise<void>((resolve, reject) => {
    execFile(cmd, args, { timeout: timeoutMs }, (err, _stdout, stderr) => {
      if (err) {
        reject(new Error(`${path.basename(cmd)} failed: ${stderr || err.message}`));
      } else {
        resolve();
      }
    });
  });
}

function canWrite(dir: string): boolean {
  try {
    fs.accessSync(dir, fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

/** Installs the downloaded artifact; returns the installed exe / .app path. */
async function installArtifact(file: string, workDir: string): Promise<string> {
  switch (process.platform) {
    case 'win32': {
      // No start-with-Windows login item: the viewer starts the server
      // itself (AUTO_START_KEY). The installer stops a running server
      // before replacing its files (setup.iss PrepareToInstall).
      await run(file, [
        '/VERYSILENT',
        '/SUPPRESSMSGBOXES',
        '/NORESTART',
        '/SP-',
        '/MERGETASKS=!autostart',
      ]);
      const exe = candidatePaths()[0];
      if (!fs.existsSync(exe)) throw new Error('Installer did not produce the server');
      return exe;
    }
    case 'darwin': {
      const mnt = path.join(workDir, 'mnt');
      fs.mkdirSync(mnt, { recursive: true });
      await run('hdiutil', ['attach', '-nobrowse', '-readonly', '-mountpoint', mnt, file]);
      try {
        const destDir = canWrite('/Applications')
          ? '/Applications'
          : path.join(os.homedir(), 'Applications');
        fs.mkdirSync(destDir, { recursive: true });
        const dest = path.join(destDir, MAC_APP_NAME);
        fs.rmSync(dest, { recursive: true, force: true });
        await run('ditto', [path.join(mnt, MAC_APP_NAME), dest]);
        return dest;
      } finally {
        await run('hdiutil', ['detach', mnt, '-quiet']).catch(() => undefined);
      }
    }
    default: {
      const root = linuxInstallRoot();
      fs.mkdirSync(root, { recursive: true });
      await run('tar', ['-xzf', file, '-C', root]);
      const exe = candidatePaths()[0];
      if (!fs.existsSync(exe)) throw new Error('Archive did not contain the server');
      return exe;
    }
  }
}

/**
 * Stops a running server ahead of an upgrade on macOS/Linux, where nothing
 * else would (the Windows installer does it itself). Best effort: the
 * caller waits for the port to go quiet either way.
 */
async function stopServer(): Promise<void> {
  if (process.platform === 'darwin') {
    await run('osascript', ['-e', 'quit app "Lowkey Media Server"'], 15_000).catch(
      () => undefined
    );
  } else if (process.platform === 'linux') {
    await run('pkill', ['-TERM', '-x', 'lowkeymediaserver'], 15_000).catch(
      () => undefined
    );
  }
}

// --background: started by the viewer, so the server must not open a
// browser tab (media-server/launch.go).
function launch(installed: string): void {
  if (process.platform === 'darwin') {
    // -g: don't bring the menu-bar app to the foreground.
    spawn('open', ['-g', installed, '--args', '--background'], {
      detached: true,
      stdio: 'ignore',
    }).unref();
    return;
  }
  // Detached so the server outlives the viewer; cwd = its own dir so the
  // exe-relative bin/ resolver behaves exactly as when started by hand.
  spawn(installed, ['--background'], {
    cwd: path.dirname(installed),
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  }).unref();
}

export const autoStartEnabled = (store: Store): boolean =>
  store.get(AUTO_START_KEY, true) as boolean;

/**
 * Starts the installed server in the background when the viewer launches,
 * unless the user turned that off or one is already answering. Kept off the
 * launch → first-media path: main.ts calls it a few seconds after the first
 * window paints. Quiet on failure — the palette's server section shows the
 * state and a Start button.
 */
export async function autoStartServer(store: Store): Promise<void> {
  if (!autoStartEnabled(store)) return;
  const installed = findInstalled(store);
  if (!installed) return;
  try {
    if (await isServerRunning()) return;
    launch(installed);
    logEvent({
      level: 'info',
      scope: 'server-install',
      message: 'auto-started the media server',
      data: { installed },
    });
  } catch (err) {
    logEvent({ scope: 'server-install', message: 'auto-start failed', error: err });
  }
}

export function registerServerInstallHandlers(store: Store): void {
  let progress: ServerInstallProgress = { phase: 'idle' };
  let inFlight: Promise<{ ok: boolean; error?: string }> | null = null;

  const report = (next: ServerInstallProgress) => {
    progress = next;
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) win.webContents.send('server-install-progress', next);
    }
  };

  const startAndWait = async (installed: string) => {
    report({ ...progress, phase: 'starting' });
    launch(installed);
    if (!(await waitForServer(60_000))) {
      throw new Error('The server was installed but did not start');
    }
  };

  const guarded = (job: () => Promise<void>) => {
    if (inFlight) return inFlight;
    inFlight = job()
      .then(() => {
        report({ ...progress, phase: 'done' });
        return { ok: true };
      })
      .catch((err: unknown) => {
        const error = err instanceof Error ? err.message : String(err);
        logEvent({ scope: 'server-install', message: error, error: err });
        report({ ...progress, phase: 'error', error });
        return { ok: false, error };
      })
      .finally(() => {
        inFlight = null;
      });
    return inFlight;
  };

  ipcMain.handle('server-install-status', async (): Promise<ServerInstallStatus> => {
    const { running, version: serverVersion } = await probeServer();
    const installedPath = findInstalled(store);
    let prompted = store.get(PROMPTED_KEY, false) as boolean;
    // A machine that already has a server never needs the first-run offer.
    if (!prompted && (running || installedPath)) {
      store.set(PROMPTED_KEY, true);
      prompted = true;
    }
    return {
      supported: serverAssetName(process.platform, process.arch) !== null,
      running,
      installedPath,
      prompted,
      serverVersion,
      appVersion: app.getVersion(),
      updateAvailable:
        running &&
        !!installedPath &&
        isOlderVersion(serverVersion, app.getVersion()),
      autoStart: autoStartEnabled(store),
      progress,
    };
  });

  ipcMain.handle('server-autostart-set', (_event, args: unknown) => {
    const on = Array.isArray(args) ? !!args[0] : !!args;
    store.set(AUTO_START_KEY, on);
    return on;
  });

  ipcMain.handle('server-install-dismiss-prompt', () => {
    store.set(PROMPTED_KEY, true);
  });

  ipcMain.handle('server-install', () =>
    guarded(async () => {
      store.set(PROMPTED_KEY, true);
      const name = serverAssetName(process.platform, process.arch);
      if (!name) throw new Error('No server build for this platform');
      report({ phase: 'resolving' });
      const { version, asset } = await resolveAsset(name);
      const workDir = fs.mkdtempSync(path.join(app.getPath('temp'), 'lowkey-server-'));
      try {
        const file = path.join(workDir, asset.name);
        await download(asset, file, (received, total) =>
          report({ phase: 'downloading', version, received, total })
        );
        if (process.platform !== 'win32') fs.chmodSync(file, 0o755);
        report({ phase: 'installing', version });
        // An upgrade: the old server has to let go of its files and its
        // port, or the wait below would see the old one still answering.
        if (await isServerRunning()) {
          await stopServer();
          if (
            process.platform !== 'win32' &&
            !(await waitForServer(20_000, false))
          ) {
            throw new Error('Could not stop the running server to upgrade it');
          }
        }
        const installed = await installArtifact(file, workDir);
        if (process.platform === 'win32') await waitForServer(20_000, false);
        store.set(INSTALL_PATH_KEY, installed);
        await startAndWait(installed);
      } finally {
        fs.rm(workDir, { recursive: true, force: true }, () => undefined);
      }
    })
  );

  ipcMain.handle('server-start', () =>
    guarded(async () => {
      const installed = findInstalled(store);
      if (!installed) throw new Error('The server is not installed');
      progress = { phase: 'idle' };
      await startAndWait(installed);
    })
  );
}
