import path from 'path';
import { BrowserWindow, shell } from 'electron';
import { resolveHtmlPath } from './util';

// The Scene editor (src/renderer/components/scene/) is the viewer's own
// renderer bundle booted in scene mode: index.html?scene=<JSON> renders the
// editor instead of the viewer. It needs the same preload as the viewer (the
// gsm:// media scheme, the stored media-server token), so it is an ordinary
// app window, just not a viewer one. One window per source file: launching
// the same file again focuses it.

const sceneWindows = new Map<string, BrowserWindow>();

export interface SceneWindowOptions {
  preload: string;
  icon: string;
  /** Called with each new window (main.ts registers it as a non-primary window). */
  onCreate?: (win: BrowserWindow) => void;
}

export function openSceneWindow(mediaPath: string, time: number, opts: SceneWindowOptions) {
  if (!mediaPath) return;
  const key = path.normalize(mediaPath).toLowerCase();
  const open = sceneWindows.get(key);
  if (open && !open.isDestroyed()) {
    if (open.isMinimized()) open.restore();
    open.focus();
    return;
  }

  const win = new BrowserWindow({
    show: false,
    width: 1480,
    height: 940,
    minWidth: 960,
    minHeight: 620,
    backgroundColor: '#0b0b0b',
    title: 'Scene editor',
    icon: opts.icon,
    autoHideMenuBar: true,
    webPreferences: {
      webSecurity: true,
      nodeIntegration: true,
      preload: opts.preload,
    },
  });
  sceneWindows.set(key, win);
  opts.onCreate?.(win);

  const param = JSON.stringify({ path: mediaPath, time: Number.isFinite(time) ? time : 0 });
  win.loadURL(resolveHtmlPath('index.html', `?scene=${encodeURIComponent(param)}`));
  win.once('ready-to-show', () => win.show());
  win.on('closed', () => {
    if (sceneWindows.get(key) === win) sceneWindows.delete(key);
  });
  win.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });
}
