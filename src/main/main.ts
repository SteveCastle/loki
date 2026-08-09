import path from 'path';
import fs from 'fs';
import os from 'os';
import {
  app,
  BrowserWindow,
  shell,
  ipcMain,
  protocol,
  dialog,
  net,
  IpcMainInvokeEvent,
} from 'electron';
import invariant from 'tiny-invariant';
import Store from 'electron-store';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';
import {
  registerSessionStoreHandlers,
  setupSessionStoreLifecycle,
} from './sessionStore';
import { cleanupArchives } from './archives';
import { registerSubtitleHandlers } from './subtitles';
import { logEvent, installGlobalErrorHandlers } from './errorLog';
import { withTimeout } from './async-timeout';
import { isValidFilePath } from './file-handling';
import { registerStudioProtocol, openStudioWindow } from './studio-window';
import {
  mark,
  getBootId,
  appVersion,
  startLoopLagSampler,
  finishLaunchTrace,
  traceMediaRead,
  describeLaunch,
} from './startup-trace';

import type { Database } from './database';

// Prevent hard crashes from unhandled errors in the main process, and persist
// them to <userData>/app-log.jsonl so field hangs/crashes can be diagnosed.
installGlobalErrorHandlers();

// Register custom protocol schemes as privileged (must be done before app ready)
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'gsm',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      bypassCSP: true,
    },
  },
  {
    // Serves the bundled Lowkey Studio app (see studio-window.ts). `secure`
    // matters: the studio needs a secure context for WebGPU + ES modules.
    scheme: 'studio',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      bypassCSP: true,
    },
  },
]);

// Expose HTMLMediaElement.audioTracks / videoTracks to renderer code.
// Without this flag Chromium leaves these collections empty even on
// MP4 files that contain multiple audio streams.
app.commandLine.appendSwitch('enable-blink-features', 'AudioVideoTracks');

// Heavy modules (database implementation, media, taxonomy, metadata, load-files)
// are dynamically imported when needed to speed up cold start.

// app.commandLine.appendSwitch('remote-debugging-port', '8315');

let db: Database | null = null;
let macPath = '';

// Launch-path timing lives in startup-trace.ts; see the header there for what
// is measured and why. Start the event-loop sampler immediately: the stalls
// worth catching happen during module loading and DB init, before any window
// exists.
const markLaunch = (name: string, data?: Record<string, unknown>) =>
  mark(name, data);
startLoopLagSampler();

// Imported on use, not at module load. electron-updater drags in ~580KB of
// dependencies (js-yaml, semver, fs-extra, builder-util-runtime, sax,
// electron-log) — about a third of the main bundle — and every byte of it was
// being parsed before the app could create a window, for a check that
// deliberately doesn't run until 1.5s after first paint. Nothing on the path to
// showing the user their file needs any of it.
async function checkForUpdatesInBackground() {
  try {
    const [{ autoUpdater }, { default: log }] = await Promise.all([
      import('electron-updater'),
      import('electron-log'),
    ]);
    log.transports.file.level = 'info';
    autoUpdater.logger = log;
    // checkForUpdatesAndNotify rejects when the release has no update manifest
    // (e.g. a 404 on latest.yml for a build published without one). Swallow it:
    // a failed update check is non-fatal and shouldn't pollute the error log.
    await autoUpdater.checkForUpdatesAndNotify();
  } catch (err) {
    logEvent({
      level: 'warn',
      scope: 'main:autoUpdater',
      message: 'update check failed',
      error: err,
    });
  }
}

let mainWindow: BrowserWindow | null = null;

// Persist renderer-side errors and load failures to the same app-log.jsonl as
// main-process errors. The renderer forwards window.onerror, unhandled
// rejections, and failed IPC invokes here (see preload + platform.ts).
ipcMain.on('log-event', (_event, entry) => {
  try {
    logEvent({
      level: entry?.level ?? 'error',
      scope: `renderer:${entry?.scope ?? 'unknown'}`,
      message: String(entry?.message ?? ''),
      data: entry?.data ?? null,
      error: entry?.error ?? null,
    });
  } catch {
    // never let logging throw
  }
});

// Make Main Process Args available to renderer process.
ipcMain.handle('get-main-args', () => {
  return process.argv;
});

ipcMain.handle('get-mac-path', () => {
  return macPath;
});

// Boot args in ONE synchronous call. The preload used to assemble
// window.appArgs from three sequential `invoke`s, which (a) cost three IPC
// round trips on the critical path and (b) raced the renderer bundle: appArgs
// is read at module-eval time, so a late reply meant `initialFile` came up
// empty and the file the user double-clicked never opened. sendSync runs
// before the page's own scripts, so the value is always there in time.
ipcMain.on('get-boot-args', (event) => {
  event.returnValue = {
    argv: process.argv,
    macPath,
    appUserData: app.getPath('userData'),
    // Lets the renderer's startup marks join the main process's timeline.
    bootId: getBootId(),
    appVersion: appVersion(),
  };
});

// The renderer reports when the user can actually see their media (or that it
// gave up waiting), which is the only true end of the launch. `args` is the
// array the preload's sendMessage wraps around its payload.
ipcMain.on('startup-first-media', (_event, args) => {
  const reason = Array.isArray(args) ? args[0]?.reason : undefined;
  finishLaunchTrace(typeof reason === 'string' ? reason : 'first-media');
});

ipcMain.handle('capture-region', async (_event, [rect]) => {
  if (!mainWindow) return null;
  // rect is in renderer CSS pixels; capturePage expects the same space.
  const r = {
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    width: Math.round(rect.width),
    height: Math.round(rect.height),
  };
  if (r.width <= 0 || r.height <= 0) return null;
  const image = await mainWindow.webContents.capturePage(r);
  return image.toPNG(); // Buffer; serialized to the renderer as a Uint8Array
});

// Window Controls
ipcMain.on('shutdown', async () => {
  // Shutdown the app.
  app.quit();
});

ipcMain.on('minimize', async () => {
  if (os.platform() === 'darwin') {
    if (mainWindow?.isFullScreen()) {
      mainWindow.once('leave-full-screen', function () {
        mainWindow?.minimize();
      });
      mainWindow?.setFullScreen(false);
    } else {
      mainWindow?.minimize();
    }
  } else {
    mainWindow?.minimize();
  }
});

ipcMain.on('open-external', async (event, args) => {
  const url = args[0];
  shell.openExternal(url);
});

ipcMain.on('show-item-in-folder', async (event, args) => {
  const filePath = args[0];
  if (filePath) shell.showItemInFolder(filePath);
});

// Launch Lowkey Studio in its own window with the given media auto-imported.
ipcMain.on('open-studio', async (_event, args) => {
  openStudioWindow(Array.isArray(args) ? args.filter((p) => typeof p === 'string') : []);
});

ipcMain.on('toggle-fullscreen', async () => {
  // Shutdown the app.
  mainWindow?.setFullScreen(!mainWindow?.isFullScreen());
});

ipcMain.on('set-always-on-top', async (event, args) => {
  const alwaysOnTop = args[0];
  const wasFullScreen = mainWindow?.isFullScreen();
  const wasFocused = mainWindow?.isFocused();

  console.log(
    `Setting always-on-top to: ${alwaysOnTop}, fullscreen: ${wasFullScreen}`
  );

  // Always apply the setting, even in fullscreen mode
  mainWindow?.setAlwaysOnTop(alwaysOnTop);

  // If the window was focused before and we're enabling always-on-top, ensure it stays focused
  if (wasFocused && alwaysOnTop) {
    mainWindow?.focus();
  }
});

// Electron Store Provider (for settings/config)
const store = new Store();

// Session Store Provider (for ephemeral session data like library, cursor, etc.)
registerSessionStoreHandlers();
registerSubtitleHandlers();
setupSessionStoreLifecycle();
// These are SYNCHRONOUS from the renderer (sendSync), and the batched one runs
// before the state machine exists — getInitialContext is built from it. A throw
// here (electron-store re-parses config.json on access, so a file truncated by
// a crash or a force-quit mid-write throws "Unexpected end of JSON input")
// leaves returnValue unset, which silently hands the renderer `undefined` for
// every setting INCLUDING dbPath — and the app then boots a different database
// with no visible error. Log loudly rather than fail quietly.
const storeReadFailed = (scope: string, err: unknown) => {
  logEvent({
    scope: `ipc:${scope}`,
    message:
      'reading config.json failed — settings will fall back to defaults ' +
      '(is the file truncated?)',
    error: err,
  });
};

ipcMain.on('electron-store-get', async (event, key, defaultValue) => {
  try {
    event.returnValue = store.get(key, defaultValue);
  } catch (err) {
    storeReadFailed('electron-store-get', err);
    event.returnValue = defaultValue;
  }
});
ipcMain.on('electron-store-set', async (event, key, val) => {
  try {
    store.set(key, val);
  } catch (err) {
    storeReadFailed('electron-store-set', err);
  }
});

// Batched synchronous get to reduce startup IPC roundtrips
ipcMain.on('electron-store-get-many', async (event, keyDefaultPairs) => {
  const pairs: [string, any][] = Array.isArray(keyDefaultPairs)
    ? keyDefaultPairs
    : [];
  const result: { [key: string]: any } = {};
  try {
    for (const [k, def] of pairs) {
      result[k] = store.get(k, def);
    }
  } catch (err) {
    // Fall back to the defaults the renderer asked for rather than an empty
    // object: `undefined` settings are worse than the documented defaults.
    for (const [k, def] of pairs) {
      if (!(k in result)) result[k] = def;
    }
    storeReadFailed('electron-store-get-many', err);
  }
  event.returnValue = result;
});

ipcMain.handle('get-user-data-path', async () => {
  return app.getPath('userData');
});

// Check for updates from GitHub releases
ipcMain.handle('check-for-updates', async () => {
  const currentVersion = app.getVersion();

  try {
    const response = await net.fetch(
      'https://api.github.com/repos/stevecastle/loki/releases/latest',
      {
        headers: {
          'User-Agent': 'Lowkey-Media-Viewer',
          Accept: 'application/vnd.github.v3+json',
        },
      }
    );

    if (!response.ok) {
      return {
        currentVersion,
        latestVersion: null,
        updateAvailable: false,
        error: `GitHub API error: ${response.status}`,
      };
    }

    const data = (await response.json()) as { tag_name?: string };
    const latestTag = data.tag_name || '';
    // Remove 'v' prefix if present for comparison
    const latestVersion = latestTag.replace(/^v/, '');

    // Compare versions (simple semver comparison)
    const current = currentVersion.split('.').map(Number);
    const latest = latestVersion.split('.').map(Number);

    let updateAvailable = false;
    for (let i = 0; i < Math.max(current.length, latest.length); i++) {
      const c = current[i] || 0;
      const l = latest[i] || 0;
      if (l > c) {
        updateAvailable = true;
        break;
      } else if (c > l) {
        // Current version is ahead (dev build)
        break;
      }
    }

    return {
      currentVersion,
      latestVersion,
      updateAvailable,
      error: null,
    };
  } catch (err) {
    return {
      currentVersion,
      latestVersion: null,
      updateAvailable: false,
      error: err instanceof Error ? err.message : 'Unknown error',
    };
  }
});

// Initialize a new DB
ipcMain.handle('load-db', async (event, args) => {
  const dbPath = args[0];
  console.log('LOADING DB:', dbPath);
  //create path if it doesn't exist

  const dir = path.dirname(dbPath);
  await fs.promises.mkdir(dir, { recursive: true });
  markLaunch('load-db-invoked');
  // Lazy import database implementation to reduce cold-start cost
  const dbModule = await import('./database');
  const { retryAsync, isDatabaseLockedError } = await import('./db-retry');
  markLaunch('db-module-loaded');

  // The Go media-server can hold a write lock on the same dream.sqlite while
  // the app starts. busy_timeout (set in Database) waits each attempt out;
  // this retry rides out a lock that outlasts a single busy_timeout window so
  // a transient lock never leaves the renderer stuck on a blank loading screen.
  //
  // Overall timeout: a lock SQLite never resolves (or a busy_timeout the driver
  // ignores for certain lock types) would otherwise leave this promise pending
  // forever — the renderer's loadingDB state has an onError escape but no way to
  // react to a hang. The timeout converts that hang into a rejection so the
  // renderer falls back to manual DB selection instead of spinning forever.
  // 60s comfortably covers 5 retries (each up to a 5s busy_timeout + backoff).
  const LOAD_DB_TIMEOUT_MS = 60_000;
  logEvent({ level: 'info', scope: 'ipc:load-db', message: 'load-db start', data: { dbPath } });
  try {
    db = await withTimeout(
      retryAsync(
        async () => {
          const candidate = new dbModule.Database(dbPath);
          try {
            await candidate.ready; // open + busy_timeout + WAL before migrating
            await dbModule.initDB(candidate);
            return candidate;
          } catch (e) {
            // Drop the half-open handle before retrying so we don't leak the
            // connection or hold our own lock across attempts.
            await candidate.close().catch(() => undefined);
            throw e;
          }
        },
        {
          retries: 5,
          isRetryable: isDatabaseLockedError,
          baseDelayMs: 300,
          onRetry: (err, attempt, delayMs) => {
            console.warn(
              `[load-db] database locked, retry ${attempt} in ${delayMs}ms:`,
              (err as Error)?.message
            );
            logEvent({
              level: 'warn',
              scope: 'ipc:load-db',
              message: `database locked, retry ${attempt} in ${delayMs}ms`,
              error: err,
            });
          },
        }
      ),
      LOAD_DB_TIMEOUT_MS,
      'load-db'
    );
  } catch (err) {
    // Surface to the renderer (onError -> manualSetup) and record why.
    logEvent({ scope: 'ipc:load-db', message: 'load-db failed', data: { dbPath }, error: err });
    throw err;
  }
  logEvent({ level: 'info', scope: 'ipc:load-db', message: 'load-db ready', data: { dbPath } });
  ipcMain.removeHandler('load-media-by-tags');
  ipcMain.removeHandler('refresh-library');
  ipcMain.removeHandler('load-media-by-description-search');
  ipcMain.removeHandler('load-tags-by-media-path');
  ipcMain.removeHandler('copy-file-into-clipboard');
  ipcMain.removeHandler('load-taxonomy');
  ipcMain.removeHandler('load-categories');
  ipcMain.removeHandler('load-category-tags');
  ipcMain.removeHandler('load-all-tags');
  ipcMain.removeHandler('get-tag');
  ipcMain.removeHandler('get-tag-count');
  ipcMain.removeHandler('get-category-count');
  ipcMain.removeHandler('create-tag');
  ipcMain.removeHandler('create-category');
  ipcMain.removeHandler('create-assignment');
  ipcMain.removeHandler('delete-assignment');
  ipcMain.removeHandler('update-assignment-weight');
  ipcMain.removeHandler('update-tag-weight');
  ipcMain.removeHandler('fetch-tag-preview');
  ipcMain.removeHandler('update-timestamp');
  ipcMain.removeHandler('remove-timestamp');
  ipcMain.removeHandler('fetch-media-preview');
  ipcMain.removeHandler('add-media');
  ipcMain.removeHandler('record-battle');
  ipcMain.removeHandler('update-description');
  ipcMain.removeHandler('select-new-path');
  ipcMain.removeHandler('rename-category');
  ipcMain.removeHandler('delete-category');
  ipcMain.removeHandler('rename-tag');
  ipcMain.removeHandler('move-tag');
  ipcMain.removeHandler('order-tags');
  ipcMain.removeHandler('delete-tag');
  ipcMain.removeHandler('create-job');
  ipcMain.removeHandler('load-duplicates-by-path');
  ipcMain.removeHandler('merge-duplicates-by-path');
  ipcMain.removeHandler('list-thumbnails');
  ipcMain.removeHandler('regenerate-thumbnail');
  ipcMain.removeHandler('delete-file');
  ipcMain.removeHandler('forget-media');
  ipcMain.removeHandler('merge-item-metadata');
  ipcMain.removeHandler('move-media');
  ipcMain.removeHandler('load-files');
  ipcMain.removeHandler('load-file-metadata');
  ipcMain.removeHandler('load-gif-metadata');
  ipcMain.removeHandler('import-files');
  ipcMain.removeHandler('update-tag-description');
  ipcMain.removeHandler('update-category-description');
  ipcMain.removeHandler('update-category-tag-view-mode');
  ipcMain.removeHandler('apply-elo-ordering');
  ipcMain.removeHandler('consolidate-tag-files');
  ipcMain.removeHandler('consolidate-category-files');
  ipcMain.removeHandler('load-media-by-query');

  // Dynamically import heavy modules in parallel and register handlers
  const [mediaModule, taxonomyModule, metadataModule, loadFilesModule] =
    await Promise.all([
      import('./media'),
      import('./taxonomy'),
      import('./metadata'),
      import('./load-files'),
    ]);
  markLaunch('handler-modules-loaded');

  // Register Media Events
  ipcMain.handle('load-files', loadFilesModule.loadFiles(db));
  ipcMain.handle('refresh-library', loadFilesModule.refreshLibrary(db));
  ipcMain.handle('load-media-by-tags', mediaModule.loadMediaByTags(db));
  ipcMain.handle(
    'load-media-by-description-search',
    mediaModule.loadMediaByDescriptionSearch(db)
  );
  ipcMain.handle('load-media-by-query', mediaModule.loadMediaByQuery(db));
  ipcMain.handle('record-battle', mediaModule.recordBattle(db));
  ipcMain.handle('update-description', mediaModule.updateDescription(db));
  ipcMain.handle(
    'copy-file-into-clipboard',
    mediaModule.copyFileIntoClipboard()
  );
  ipcMain.handle('delete-file', mediaModule.deleteMedia(db));
  ipcMain.handle('forget-media', mediaModule.forgetMedia(db));
  ipcMain.handle('merge-item-metadata', mediaModule.mergeItemMetadata(db));
  ipcMain.handle('move-media', mediaModule.moveMedia(db));
  ipcMain.handle('import-files', mediaModule.importFiles(db));
  ipcMain.handle(
    'load-duplicates-by-path',
    mediaModule.loadDuplicatesByPath(db)
  );
  ipcMain.handle(
    'merge-duplicates-by-path',
    mediaModule.mergeDuplicatesByPath(db)
  );

  // Register Metadata/Taxonomy Events
  ipcMain.handle(
    'load-tags-by-media-path',
    taxonomyModule.loadTagsByMediaPath(db)
  );
  ipcMain.handle('load-categories', taxonomyModule.loadCategories(db));
  ipcMain.handle('load-category-tags', taxonomyModule.loadCategoryTags(db));
  ipcMain.handle('load-all-tags', taxonomyModule.loadAllTags(db));
  ipcMain.handle('get-tag', taxonomyModule.getTag(db));
  ipcMain.handle('get-tag-count', taxonomyModule.getTagCount(db));
  ipcMain.handle('get-category-count', taxonomyModule.getCategoryCount(db));
  ipcMain.handle('create-tag', taxonomyModule.createTag(db));
  ipcMain.handle('create-category', taxonomyModule.createCategory(db));
  ipcMain.handle(
    'create-assignment',
    taxonomyModule.createAssignment(db, store)
  );
  ipcMain.handle('delete-assignment', taxonomyModule.deleteAssignment(db));
  ipcMain.handle(
    'update-assignment-weight',
    taxonomyModule.updateAssignmentWeight(db)
  );
  ipcMain.handle('update-tag-weight', taxonomyModule.updateTagWeight(db));
  ipcMain.handle('fetch-tag-preview', taxonomyModule.fetchTagPreview(db));
  ipcMain.handle('update-timestamp', taxonomyModule.updateTimestamp(db));
  ipcMain.handle('remove-timestamp', taxonomyModule.removeTimestamp(db));
  ipcMain.handle(
    'fetch-media-preview',
    mediaModule.fetchMediaPreview(db, store)
  );
  ipcMain.handle('list-thumbnails', mediaModule.listThumbnails(store));
  ipcMain.handle(
    'regenerate-thumbnail',
    mediaModule.regenerateThumbnail(store)
  );
  ipcMain.handle('load-file-metadata', metadataModule.loadFileMetaData(db));
  ipcMain.handle('load-gif-metadata', metadataModule.loadGifMetadata());

  ipcMain.handle(
    'select-new-path',
    taxonomyModule.selectNewPath(mainWindow)
  );
  ipcMain.handle('rename-category', taxonomyModule.renameCategory(db));
  ipcMain.handle('delete-category', taxonomyModule.deleteCategory(db));
  ipcMain.handle('rename-tag', taxonomyModule.renameTag(db));
  ipcMain.handle('move-tag', taxonomyModule.moveTag(db));
  ipcMain.handle('order-tags', taxonomyModule.orderTags(db));
  ipcMain.handle('delete-tag', taxonomyModule.deleteTag(db));
  ipcMain.handle(
    'update-tag-description',
    taxonomyModule.updateTagDescription(db)
  );
  ipcMain.handle(
    'update-category-description',
    taxonomyModule.updateCategoryDescription(db)
  );
  ipcMain.handle(
    'update-category-tag-view-mode',
    taxonomyModule.updateCategoryTagViewMode(db)
  );
  ipcMain.handle('apply-elo-ordering', taxonomyModule.applyEloOrdering(db));
  ipcMain.handle(
    'consolidate-tag-files',
    taxonomyModule.consolidateTagFiles(db)
  );
  ipcMain.handle(
    'consolidate-category-files',
    taxonomyModule.consolidateCategoryFiles(db)
  );
  if (!mainWindow) return;
  // Job creation removed - now handled by external job runner service
});

type SelectDBInput = [string | undefined];
ipcMain.handle(
  'select-db',
  async (_: IpcMainInvokeEvent, args: SelectDBInput) => {
    invariant(mainWindow, 'mainWindow is not defined');
    const defaultPath = args[0];
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile', 'promptToCreate', 'dontAddToRecent'],
      defaultPath,
      filters: [{ name: 'Lowkey Media Database', extensions: ['sqlite'] }],
    });

    if (!result.canceled) {
      console.log('SELECTED FILE PATH:', result);
      return result.filePaths[0];
    } else {
      return null;
    }
  }
);

// Handle file selection event from renderer process
type SelectFileInput = [string | undefined];
ipcMain.handle(
  'select-file',
  async (_: IpcMainInvokeEvent, args: SelectFileInput) => {
    invariant(mainWindow, 'mainWindow is not defined');
    const defaultPath = args[0];
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      defaultPath,
      filters: [
        {
          name: 'Media',
          extensions: [
            'jpg',
            'jpeg',
            'png',
            'gif',
            'bmp',
            'svg',
            'jfif',
            'pjpeg',
            'pjp',
            'webp',
            'avif',
            'mp4',
            'mov',
            'mkv',
            'webm',
            'cbz',
            'zip',
          ],
        },

        {
          name: 'Images',
          extensions: [
            'jpg',
            'jpeg',
            'png',
            'gif',
            'bmp',
            'svg',
            'jfif',
            'pjpeg',
            'pjp',
            'webp',
            'avif',
          ],
        },
        { name: 'Movies', extensions: ['mp4', 'mkv', 'webm', 'mov'] },
        { name: 'Archives', extensions: ['cbz', 'zip'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (!result.canceled) {
      return result.filePaths[0];
    } else {
      return null;
    }
  }
);

// Handle directory selection event from renderer process
type SelectDirectoryInput = [string | undefined];
ipcMain.handle(
  'select-directory',
  async (_: IpcMainInvokeEvent, args: SelectDirectoryInput) => {
    invariant(mainWindow, 'mainWindow is not defined');
    const defaultPath = args[0];
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
      defaultPath,
    });

    if (!result.canceled) {
      return result.filePaths[0];
    } else {
      return null;
    }
  }
);

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

const isDebug =
  process.env.NODE_ENV === 'development' || process.env.DEBUG_PROD === 'true';

if (isDebug) {
  require('electron-debug')();
  app.commandLine.appendSwitch('inspect');
}

const createWindow = async () => {
  if (isDebug) {
    // await installExtensions();
  }

  const RESOURCES_PATH = app.isPackaged
    ? path.join(process.resourcesPath, 'assets')
    : path.join(__dirname, '../../assets');

  const getAssetPath = (...paths: string[]): string => {
    return path.join(RESOURCES_PATH, ...paths);
  };

  markLaunch('creating-window');
  mainWindow = new BrowserWindow({
    show: false,
    width: 1024,
    height: 728,
    fullscreen: true,
    frame: false,
    titleBarStyle: 'hidden',
    // A 256px icon, NOT assets/icon.png. That file is 1849x1850 (it exists at
    // that size because electron-builder needs a large source to generate .icns
    // / .ico at package time), and BrowserWindow decodes and rescales whatever
    // it is handed SYNCHRONOUSLY: measured at ~180ms of blocked main process on
    // every single launch — a quarter of the time to first media, and blocking
    // exactly when the preload's media read and the renderer's first IPC need
    // the main process. At 256px the same work is a couple of milliseconds and
    // the window/taskbar icon looks identical.
    icon: getAssetPath('icon-window.png'),
    webPreferences: {
      webSecurity: true,
      nodeIntegration: true,
      nodeIntegrationInWorker: true,
      preload: app.isPackaged
        ? path.join(__dirname, 'preload.js')
        : path.join(__dirname, '../../.erb/dll/preload.js'),
    },
  });

  markLaunch('window-created', { packaged: app.isPackaged });
  // What this launch was asked to open — a file (and how big, on which root),
  // a directory, or nothing. Without it the timings can't be compared.
  describeLaunch(isValidFilePath(process.argv[1]) ? process.argv[1] : macPath);
  mainWindow.loadURL(resolveHtmlPath(`index.html`));
  markLaunch('load-url-called');

  mainWindow.webContents.on('did-finish-load', () => markLaunch('did-finish-load'));

  mainWindow.on('ready-to-show', () => {
    markLaunch('ready-to-show');
    if (!mainWindow) {
      throw new Error('"mainWindow" is not defined');
    }
    if (process.env.START_MINIMIZED) {
      mainWindow.minimize();
    } else {
      mainWindow.show();
    }
    // Defer auto updates until after first paint
    setTimeout(() => {
      checkForUpdatesInBackground();
    }, 1500);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle fullscreen state changes to keep always-on-top in sync
  mainWindow.on('leave-full-screen', () => {
    // Re-apply always-on-top setting when exiting fullscreen
    // Use a small delay to ensure the window has fully transitioned out of fullscreen
    setTimeout(() => {
      const alwaysOnTop = store.get('alwaysOnTop', false) as boolean;
      console.log('Exiting fullscreen, alwaysOnTop setting:', alwaysOnTop);

      // Always re-apply the setting to ensure sync, whether true or false
      mainWindow?.setAlwaysOnTop(alwaysOnTop);
      console.log(
        `Applied always-on-top: ${alwaysOnTop} after exiting fullscreen`
      );

      // Ensure window stays focused if always-on-top was enabled
      if (alwaysOnTop) {
        setTimeout(() => {
          mainWindow?.focus();
        }, 50);
      }
    }, 200);
  });

  mainWindow.on('enter-full-screen', () => {
    // When entering fullscreen, the always-on-top state might be overridden
    // but we don't need to do anything special here as fullscreen takes precedence
  });

  const menuBuilder = new MenuBuilder(mainWindow);
  menuBuilder.buildMenu();
  markLaunch('menu-built');

  // Open urls in the user's browser
  mainWindow.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
  });

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    console.error(
      `Renderer process gone: reason=${details.reason}, exitCode=${details.exitCode}`
    );
    if (details.reason === 'crashed' || details.reason === 'oom') {
      mainWindow?.reload();
    }
  });

  // Auto updater initialized after first paint (see ready-to-show)
};

/**
 * Add event listeners...
 */
app.on('open-file', (event, path) => {
  event.preventDefault();
  console.log('OPEN FILE:', path);
  macPath = path;
});

const gsmMimeTypes: Record<string, string> = {
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.avi': 'video/x-msvideo',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.ogg': 'audio/ogg',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.jfif': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml',
};

app.on('ready', async () => {
  registerStudioProtocol();
  protocol.handle('gsm', async (request) => {
    try {
      // Parse URL to file path
      const parsed = new URL(request.url);
      let filePath = decodeURIComponent(parsed.pathname);
      if (process.platform === 'win32' && parsed.host) {
        filePath = `${parsed.host.toUpperCase()}:${filePath}`;
      } else if (process.platform === 'win32' && filePath.startsWith('/')) {
        filePath = filePath.slice(1);
      } else if (process.platform !== 'win32' && parsed.host) {
        filePath = `/${parsed.host}${filePath}`;
      }
      filePath = path.normalize(filePath);

      // Times the opening reads only (see traceMediaRead) — this is the actual
      // delivery of the bytes the user is waiting to see.
      const trace = traceMediaRead(filePath, !!request.headers.get('Range'));

      // Get file info
      let stats: fs.Stats;
      try {
        stats = await fs.promises.stat(filePath);
        trace.stat(true, stats.size);
      } catch {
        trace.stat(false);
        return new Response('Not Found', { status: 404 });
      }

      const fileSize = stats.size;
      const ext = path.extname(filePath).toLowerCase();
      const contentType = gsmMimeTypes[ext] || 'application/octet-stream';
      const rangeHeader = request.headers.get('Range');

      // Handle range requests (for video seeking)
      if (rangeHeader) {
        const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
        if (match) {
          const start = parseInt(match[1], 10);
          const end = match[2] ? parseInt(match[2], 10) : fileSize - 1;

          if (start >= fileSize) {
            trace.responding(416);
            trace.done('error', 0);
            return new Response('Range Not Satisfiable', {
              status: 416,
              headers: { 'Content-Range': `bytes */${fileSize}` },
            });
          }

          const clampedEnd = Math.min(end, fileSize - 1);
          const chunkSize = clampedEnd - start + 1;

          const stream = fs.createReadStream(filePath, { start, end: clampedEnd });
          let sent = 0;
          trace.responding(206, { chunkSize });
          return new Response(
            new ReadableStream({
              start(controller) {
                stream.on('data', (chunk: Buffer) => {
                  sent += chunk.length;
                  controller.enqueue(new Uint8Array(chunk));
                });
                stream.on('end', () => {
                  trace.done('end', sent);
                  controller.close();
                });
                stream.on('error', (e) => {
                  trace.done('error', sent);
                  controller.error(e);
                });
              },
              cancel() {
                trace.done('cancel', sent);
                stream.destroy();
              },
            }),
            {
              status: 206,
              headers: {
                'Content-Type': contentType,
                'Content-Length': chunkSize.toString(),
                'Content-Range': `bytes ${start}-${clampedEnd}/${fileSize}`,
                'Accept-Ranges': 'bytes',
              },
            }
          );
        }
      }

      // Full file request - stream with ETag for cache revalidation
      const etag = `"${stats.mtimeMs.toString(36)}-${fileSize.toString(36)}"`;
      const ifNoneMatch = request.headers.get('If-None-Match');
      if (ifNoneMatch === etag) {
        // Revalidation hit: Chromium already holds the bytes. Worth seeing —
        // it's how the preload's warm-up pays off on the real <img>.
        trace.responding(304);
        trace.done('end', 0);
        return new Response(null, { status: 304 });
      }

      const stream = fs.createReadStream(filePath);
      let sent = 0;
      trace.responding(200);
      return new Response(
        new ReadableStream({
          start(controller) {
            stream.on('data', (chunk: Buffer) => {
              sent += chunk.length;
              controller.enqueue(new Uint8Array(chunk));
            });
            stream.on('end', () => {
              trace.done('end', sent);
              controller.close();
            });
            stream.on('error', (e) => {
              trace.done('error', sent);
              controller.error(e);
            });
          },
          cancel() {
            trace.done('cancel', sent);
            stream.destroy();
          },
        }),
        {
          status: 200,
          headers: {
            'Content-Type': contentType,
            'Content-Length': fileSize.toString(),
            'Accept-Ranges': 'bytes',
            'ETag': etag,
            'Cache-Control': 'no-cache',
          },
        }
      );
    } catch (error) {
      console.error('GSM protocol error:', error);
      return new Response('Internal Error', { status: 500 });
    }
  });
});

app.on('before-quit', async () => {
  if (db) {
    try {
      await db.close();
    } catch (err) {
      console.error('Error closing database on quit:', err);
    }
    db = null;
  }
  try {
    await cleanupArchives();
  } catch (err) {
    console.error('Error cleaning up archive cache on quit:', err);
  }
});

app.on('window-all-closed', () => {
  // Respect the OSX convention of having the application in memory even
  // after all windows have been closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app
  .whenReady()
  .then(() => {
    markLaunch('app-ready');
    createWindow();
    app.on('activate', () => {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow();
    });
  })
  .catch(console.log);
