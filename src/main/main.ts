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
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import invariant from 'tiny-invariant';
import Store from 'electron-store';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';
import {
  registerSessionStoreHandlers,
  setupSessionStoreLifecycle,
} from './sessionStore';

import type { Database } from './database';

// Register custom protocol scheme as privileged (must be done before app ready)
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
]);

// Heavy modules (database implementation, media, taxonomy, metadata, load-files)
// are dynamically imported when needed to speed up cold start.

// app.commandLine.appendSwitch('remote-debugging-port', '8315');

let db: Database | null = null;
let macPath = '';

class AppUpdater {
  constructor() {
    log.transports.file.level = 'info';
    autoUpdater.logger = log;
    autoUpdater.checkForUpdatesAndNotify();
  }
}

let mainWindow: BrowserWindow | null = null;

// Make Main Process Args available to renderer process.
ipcMain.handle('get-main-args', () => {
  return process.argv;
});

ipcMain.handle('get-mac-path', () => {
  return macPath;
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
setupSessionStoreLifecycle();
ipcMain.on('electron-store-get', async (event, key, defaultValue) => {
  event.returnValue = store.get(key, defaultValue);
});
ipcMain.on('electron-store-set', async (event, key, val) => {
  store.set(key, val);
});

// Batched synchronous get to reduce startup IPC roundtrips
ipcMain.on('electron-store-get-many', async (event, keyDefaultPairs) => {
  try {
    const pairs: [string, any][] = Array.isArray(keyDefaultPairs)
      ? keyDefaultPairs
      : [];
    const result: { [key: string]: any } = {};
    for (const [k, def] of pairs) {
      result[k] = store.get(k, def);
    }
    event.returnValue = result;
  } catch (err) {
    console.error('electron-store-get-many error', err);
    event.returnValue = {};
  }
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
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  // Lazy import database implementation to reduce cold-start cost
  const dbModule = await import('./database');
  db = new dbModule.Database(dbPath);
  await dbModule.initDB(db);
  ipcMain.removeHandler('load-media-by-tags');
  ipcMain.removeHandler('load-media-by-description-search');
  ipcMain.removeHandler('load-tags-by-media-path');
  ipcMain.removeHandler('copy-file-into-clipboard');
  ipcMain.removeHandler('load-taxonomy');
  ipcMain.removeHandler('get-tag-count');
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
  ipcMain.removeHandler('update-elo');
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
  ipcMain.removeHandler('load-files');
  ipcMain.removeHandler('load-file-metadata');
  ipcMain.removeHandler('load-gif-metadata');

  // Dynamically import heavy modules in parallel and register handlers
  const [mediaModule, taxonomyModule, metadataModule, loadFilesModule] =
    await Promise.all([
      import('./media'),
      import('./taxonomy'),
      import('./metadata'),
      import('./load-files'),
    ]);

  // Register Media Events
  ipcMain.handle('load-files', loadFilesModule.loadFiles(db));
  ipcMain.handle('refresh-library', loadFilesModule.refreshLibrary(db));
  ipcMain.handle('load-media-by-tags', mediaModule.loadMediaByTags(db));
  ipcMain.handle(
    'load-media-by-description-search',
    mediaModule.loadMediaByDescriptionSearch(db)
  );
  ipcMain.handle('update-elo', mediaModule.updateElo(db));
  ipcMain.handle('update-description', mediaModule.updateDescription(db));
  ipcMain.handle(
    'copy-file-into-clipboard',
    mediaModule.copyFileIntoClipboard()
  );
  ipcMain.handle('delete-file', mediaModule.deleteMedia(db));
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
  ipcMain.handle('load-taxonomy', taxonomyModule.loadTaxonomy(db));
  ipcMain.handle('get-tag-count', taxonomyModule.getTagCount(db));
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
    taxonomyModule.selectNewPath(db, mainWindow)
  );
  ipcMain.handle('rename-category', taxonomyModule.renameCategory(db));
  ipcMain.handle('delete-category', taxonomyModule.deleteCategory(db));
  ipcMain.handle('rename-tag', taxonomyModule.renameTag(db));
  ipcMain.handle('move-tag', taxonomyModule.moveTag(db));
  ipcMain.handle('order-tags', taxonomyModule.orderTags(db));
  ipcMain.handle('delete-tag', taxonomyModule.deleteTag(db));
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
            'mp4',
            'mov',
            'mkv',
            'webm',
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
          ],
        },
        { name: 'Movies', extensions: ['mp4', 'mkv', 'webm', 'mov'] },
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

  mainWindow = new BrowserWindow({
    show: false,
    width: 1024,
    height: 728,
    fullscreen: true,
    frame: false,
    titleBarStyle: 'hidden',
    icon: getAssetPath('icon.png'),
    webPreferences: {
      webSecurity: true,
      nodeIntegration: true,
      nodeIntegrationInWorker: true,
      preload: app.isPackaged
        ? path.join(__dirname, 'preload.js')
        : path.join(__dirname, '../../.erb/dll/preload.js'),
    },
  });

  mainWindow.loadURL(resolveHtmlPath(`index.html`));

  mainWindow.on('ready-to-show', () => {
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
      // eslint-disable-next-line
      new AppUpdater();
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

  // Open urls in the user's browser
  mainWindow.webContents.setWindowOpenHandler((edata) => {
    shell.openExternal(edata.url);
    return { action: 'deny' };
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

app.on('ready', async () => {
  mainWindow?.webContents.openDevTools();
  // Custom protocol handler with full range request support for video seeking
  protocol.handle('gsm', async (request) => {
    try {
      const parsed = new URL(request.url);
      let filePath = decodeURIComponent(parsed.pathname);

      // On Windows, the drive letter becomes the URL host (e.g., gsm://c/Users/... )
      // Reconstruct the path: host + pathname = c + /Users/... = C:/Users/...
      if (process.platform === 'win32' && parsed.host) {
        filePath = `${parsed.host.toUpperCase()}:${filePath}`;
      } else if (process.platform === 'win32' && filePath.startsWith('/')) {
        // Fallback: remove leading slash if path has drive letter (e.g., /C:/...)
        filePath = filePath.slice(1);
      } else if (process.platform !== 'win32' && parsed.host) {
        // On macOS/Linux, if we have a host, the path was split incorrectly
        // (e.g., gsm://Users/runes/file.jpg -> host="Users", pathname="/runes/file.jpg")
        // Reconstruct: /${host}${pathname} = /Users/runes/file.jpg
        filePath = `/${parsed.host}${filePath}`;
      }

      // Normalize path
      filePath = path.normalize(filePath);

      // Check if file exists
      if (!fs.existsSync(filePath)) {
        return new Response('Not Found', { status: 404 });
      }

      // Get file stats
      const stats = fs.statSync(filePath);
      const fileSize = stats.size;

      // Determine MIME type
      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes: Record<string, string> = {
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
        '.bmp': 'image/bmp',
        '.svg': 'image/svg+xml',
      };
      const contentType = mimeTypes[ext] || 'application/octet-stream';

      // Check for Range header (needed for video seeking)
      const rangeHeader = request.headers.get('Range');

      if (rangeHeader) {
        // Parse range header: "bytes=start-end" or "bytes=start-"
        const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
        if (match) {
          const start = parseInt(match[1], 10);
          const end = match[2] ? parseInt(match[2], 10) : fileSize - 1;
          const chunkSize = end - start + 1;

          // Create stream for the requested range
          const stream = fs.createReadStream(filePath, { start, end });
          const webStream = new ReadableStream({
            start(controller) {
              stream.on('data', (chunk: Buffer) => {
                controller.enqueue(new Uint8Array(chunk));
              });
              stream.on('end', () => controller.close());
              stream.on('error', (err: Error) => controller.error(err));
            },
            cancel() {
              stream.destroy();
            },
          });

          return new Response(webStream, {
            status: 206, // Partial Content
            headers: {
              'Content-Type': contentType,
              'Content-Length': chunkSize.toString(),
              'Content-Range': `bytes ${start}-${end}/${fileSize}`,
              'Accept-Ranges': 'bytes',
            },
          });
        }
      }

      // No range request - return full file
      const stream = fs.createReadStream(filePath);
      const webStream = new ReadableStream({
        start(controller) {
          stream.on('data', (chunk: Buffer) => {
            controller.enqueue(new Uint8Array(chunk));
          });
          stream.on('end', () => controller.close());
          stream.on('error', (err: Error) => controller.error(err));
        },
        cancel() {
          stream.destroy();
        },
      });

      return new Response(webStream, {
        status: 200,
        headers: {
          'Content-Type': contentType,
          'Content-Length': fileSize.toString(),
          'Accept-Ranges': 'bytes',
        },
      });
    } catch (error) {
      console.error('Protocol handler error:', error);
      return new Response('Internal Error', { status: 500 });
    }
  });
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
    createWindow();
    app.on('activate', () => {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow();
    });
  })
  .catch(console.log);
