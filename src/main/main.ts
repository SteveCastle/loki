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
  IpcMainInvokeEvent,
} from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import invariant from 'tiny-invariant';
import Store from 'electron-store';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';

import { Database, initDB } from './database';

import {
  loadMediaByTags,
  loadMediaByDescriptionSearch,
  copyFileIntoClipboard,
  updateElo,
  deleteMedia,
  fetchMediaPreview,
  updateDescription,
} from './media';
import {
  loadTaxonomy,
  getTagCount,
  createCategory,
  createTag,
  createAssignment,
  deleteAssignment,
  updateAssignmentWeight,
  updateTagWeight,
  fetchTagPreview,
  renameCategory,
  deleteCategory,
  renameTag,
  moveTag,
  orderTags,
  deleteTag,
  loadTagsByMediaPath,
  selectNewPath,
  updateTimestamp,
  removeTimestamp,
} from './taxonomy';

import { loadFileMetaData } from './metadata';
import { createJob } from './jobs';
import { loadFiles } from './load-files';

// app.commandLine.appendSwitch('remote-debugging-port', '8315');
app.commandLine.appendSwitch('inspect');

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

// Electron Store Provider
const store = new Store();
ipcMain.on('electron-store-get', async (event, key, defaultValue) => {
  event.returnValue = store.get(key, defaultValue);
});
ipcMain.on('electron-store-set', async (event, key, val) => {
  store.set(key, val);
});

ipcMain.handle('get-user-data-path', async () => {
  return app.getPath('userData');
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
  db = new Database(dbPath);
  await initDB(db);
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
  ipcMain.removeHandler('delete-file');
  ipcMain.removeHandler('load-files');
  ipcMain.removeHandler('load-file-metadata');

  // Register Media Events
  ipcMain.handle('load-files', loadFiles(db));
  ipcMain.handle('load-media-by-tags', loadMediaByTags(db));
  ipcMain.handle(
    'load-media-by-description-search',
    loadMediaByDescriptionSearch(db)
  );
  ipcMain.handle('update-elo', updateElo(db));
  ipcMain.handle('update-description', updateDescription(db));
  ipcMain.handle('copy-file-into-clipboard', copyFileIntoClipboard());
  ipcMain.handle('delete-file', deleteMedia(db));

  // Register Metaata Events
  ipcMain.handle('load-tags-by-media-path', loadTagsByMediaPath(db));

  // Register Taxonomy Eventsmet
  ipcMain.handle('load-taxonomy', loadTaxonomy(db));
  ipcMain.handle('get-tag-count', getTagCount(db));
  ipcMain.handle('create-tag', createTag(db));
  ipcMain.handle('create-category', createCategory(db));
  ipcMain.handle('create-assignment', createAssignment(db, store));
  ipcMain.handle('delete-assignment', deleteAssignment(db));
  ipcMain.handle('update-assignment-weight', updateAssignmentWeight(db));
  ipcMain.handle('update-tag-weight', updateTagWeight(db));
  ipcMain.handle('fetch-tag-preview', fetchTagPreview(db));
  ipcMain.handle('update-timestamp', updateTimestamp(db));
  ipcMain.handle('remove-timestamp', removeTimestamp(db));
  ipcMain.handle('fetch-media-preview', fetchMediaPreview(db, store));
  ipcMain.handle('load-file-metadata', loadFileMetaData(db));

  ipcMain.handle('select-new-path', selectNewPath(db, mainWindow));
  ipcMain.handle('rename-category', renameCategory(db));
  ipcMain.handle('delete-category', deleteCategory(db));
  ipcMain.handle('rename-tag', renameTag(db));
  ipcMain.handle('move-tag', moveTag(db));
  ipcMain.handle('order-tags', orderTags(db));
  ipcMain.handle('delete-tag', deleteTag(db));
  if (!mainWindow) return;
  ipcMain.handle('create-job', createJob(db, mainWindow.webContents));
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

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

const isDebug =
  process.env.NODE_ENV === 'development' || process.env.DEBUG_PROD === 'true';

if (isDebug) {
  require('electron-debug')();
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

  // Remove this if your app does not use auto updates
  // eslint-disable-next-line
  new AppUpdater();
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
  protocol.registerFileProtocol('gsm', (request, callback) => {
    const url = request.url.replace('gsm:', '');
    try {
      const decodedUrl = decodeURIComponent(url);
      return callback(decodedUrl);
    } catch (error) {
      console.error(error);
      return callback('404');
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
