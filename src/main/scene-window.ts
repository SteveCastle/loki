import path from 'path';
import { BrowserWindow, dialog, ipcMain, shell, webContents } from 'electron';
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

const isSceneWindow = (w: BrowserWindow) => [...sceneWindows.values()].includes(w);

// Viewer windows: everything but scene editors and Lowkey Studio.
const viewerWindows = () =>
  BrowserWindow.getAllWindows().filter(
    (w) => !w.isDestroyed() && !isSceneWindow(w) && !w.webContents.getURL().startsWith('studio:')
  );

const focus = (w: BrowserWindow) => {
  if (w.isMinimized()) w.restore();
  w.show();
  w.focus();
};

// Pick requests in flight: request id -> the editor (webContents id) that asked.
const pickRequesters = new Map<string, number>();

/**
 * Relays reference picking between a Scene editor and the viewer (see
 * components/scene/pick-bridge.ts): an editor's request opens the viewer's
 * picking banner and brings the viewer forward; the viewer's answer goes back
 * to that editor, which comes forward again. Also the editor's multi-file
 * dialog for references outside the library.
 */
export function registerScenePickRelay() {
  ipcMain.on('scene-pick-request', (event, args) => {
    const msg = Array.isArray(args) ? args[0] : null;
    if (!msg || (msg.type !== 'open' && msg.type !== 'close')) return;
    if (msg.type === 'open' && msg.req?.id) pickRequesters.set(msg.req.id, event.sender.id);
    if (msg.type === 'close') pickRequesters.delete(msg.id);
    const viewers = viewerWindows();
    viewers.forEach((w) => w.webContents.send('scene-pick-request', msg));
    if (msg.type === 'open' && viewers[0]) focus(viewers[0]);
  });
  ipcMain.on('scene-pick-result', (_event, args) => {
    const msg = Array.isArray(args) ? args[0] : null;
    if (!msg?.id) return;
    const requester = pickRequesters.get(msg.id);
    pickRequesters.delete(msg.id);
    // Any other viewer window showing the same request drops its banner.
    viewerWindows().forEach((w) => w.webContents.send('scene-pick-request', { type: 'close', id: msg.id }));
    const wc = requester !== undefined ? webContents.fromId(requester) : undefined;
    if (!wc || wc.isDestroyed()) return;
    wc.send('scene-pick-result', msg);
    const win = BrowserWindow.fromWebContents(wc);
    if (win) focus(win);
  });
  ipcMain.handle('select-reference-files', async (event, args) => {
    const kinds: string[] = Array.isArray(args?.[0]) ? args[0] : ['image'];
    const ext: Record<string, string[]> = {
      image: ['jpg', 'jpeg', 'png', 'webp', 'avif', 'bmp', 'gif', 'heic', 'heif', 'jfif'],
      video: ['mp4', 'mov', 'mkv', 'webm', 'm4v', 'avi'],
      audio: ['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'opus'],
    };
    const win = BrowserWindow.fromWebContents(event.sender);
    const opts = {
      title: 'Add references',
      properties: ['openFile', 'multiSelections'] as Array<'openFile' | 'multiSelections'>,
      filters: [{ name: kinds.join(', '), extensions: kinds.flatMap((k) => ext[k] || []) }],
    };
    const result = win ? await dialog.showOpenDialog(win, opts) : await dialog.showOpenDialog(opts);
    return result.canceled ? [] : result.filePaths;
  });
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
  const wcId = win.webContents.id;
  win.on('closed', () => {
    if (sceneWindows.get(key) === win) sceneWindows.delete(key);
    // A closed editor withdraws its pick requests.
    [...pickRequesters].forEach(([id, requester]) => {
      if (requester !== wcId) return;
      pickRequesters.delete(id);
      viewerWindows().forEach((w) => w.webContents.send('scene-pick-request', { type: 'close', id }));
    });
  });
  win.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });
}
