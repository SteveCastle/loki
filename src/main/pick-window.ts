import { BrowserWindow, ipcMain, shell, webContents } from 'electron';
import { resolveHtmlPath } from './util';

// Reference picker windows (see src/renderer/components/scene/pick-bridge.ts).
// A Scene editor or the Transform modal asks for references; the picker is the
// viewer's own bundle booted with index.html?pick=<JSON request> — the whole
// viewer over the last session, plus a picking banner — in a window of its own,
// so the window that asked never navigates away. The answer goes back to the
// requester, which comes forward again, and the picker closes. One picker at a
// time: a new request replaces the open one (whose requester gets a cancel).

export interface PickWindowOptions {
  preload: string;
  icon: string;
  /** Called with each new window (main.ts registers it as a non-primary window). */
  onCreate?: (win: BrowserWindow) => void;
}

interface OpenPick {
  id: string;
  /** webContents id of the window that asked. */
  requester: number;
  win: BrowserWindow;
  /** Stops watching the requester. */
  unwatch: () => void;
}

let open: OpenPick | null = null;

const focus = (w: BrowserWindow) => {
  if (w.isMinimized()) w.restore();
  w.show();
  w.focus();
};

/** Ends the open pick: tells the requester (paths, or null = cancelled) and closes the picker. */
function finish(id: string, paths: string[] | null, notify = true) {
  if (!open || open.id !== id) return;
  const { requester, win, unwatch } = open;
  open = null;
  unwatch();
  const wc = webContents.fromId(requester);
  if (notify && wc && !wc.isDestroyed()) {
    wc.send('scene-pick-result', { id, paths });
    const rw = BrowserWindow.fromWebContents(wc);
    if (rw) focus(rw);
  }
  if (!win.isDestroyed()) win.close();
}

export function registerPickWindows(opts: PickWindowOptions) {
  ipcMain.on('open-pick-window', (event, args) => {
    const req = Array.isArray(args) ? args[0] : null;
    if (typeof req?.id !== 'string' || !Array.isArray(req.kinds)) return;
    if (open) finish(open.id, null);

    const win = new BrowserWindow({
      show: false,
      width: 1280,
      height: 860,
      minWidth: 720,
      minHeight: 480,
      backgroundColor: '#0b0b0b',
      title: req.label ? `Choose references · ${req.label}` : 'Choose references',
      icon: opts.icon,
      autoHideMenuBar: true,
      webPreferences: {
        webSecurity: true,
        nodeIntegration: true,
        nodeIntegrationInWorker: true,
        preload: opts.preload,
      },
    });
    // The requester going away withdraws its pick.
    const requester = event.sender;
    const onGone = () => finish(req.id, null, false);
    requester.once('destroyed', onGone);
    const pick: OpenPick = {
      id: req.id,
      requester: requester.id,
      win,
      unwatch: () => {
        if (!requester.isDestroyed()) requester.removeListener('destroyed', onGone);
      },
    };
    open = pick;
    opts.onCreate?.(win);

    const param = JSON.stringify({ id: req.id, kinds: req.kinds, label: req.label || '' });
    win.loadURL(resolveHtmlPath('index.html', `?pick=${encodeURIComponent(param)}`));
    win.once('ready-to-show', () => focus(win));
    // The page sets its own title from the viewer; keep ours.
    win.on('page-title-updated', (e) => e.preventDefault());
    // Closed without an answer: the requester's pick is cancelled.
    win.on('closed', () => finish(pick.id, null));
    win.webContents.setWindowOpenHandler((edata) => {
      shell.openExternal(edata.url);
      return { action: 'deny' };
    });
  });

  // The picker's answer.
  ipcMain.on('scene-pick-result', (_event, args) => {
    const msg = Array.isArray(args) ? args[0] : null;
    if (typeof msg?.id !== 'string') return;
    finish(msg.id, Array.isArray(msg.paths) ? msg.paths : null);
  });

  // The requester withdrawing its pick (e.g. the Transform modal closed).
  ipcMain.on('scene-pick-request', (_event, args) => {
    const msg = Array.isArray(args) ? args[0] : null;
    if (msg?.type === 'close' && typeof msg.id === 'string') finish(msg.id, null, false);
  });
}
