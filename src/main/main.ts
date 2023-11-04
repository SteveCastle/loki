import path from 'path';
import fs from 'fs';
import { app, BrowserWindow, shell, ipcMain, protocol, dialog } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import invariant from 'tiny-invariant';
import Store from 'electron-store';
import MenuBuilder from './menu';
import { resolveHtmlPath } from './util';

import { Database, initDB } from './database';

import {
  loadMediaByTags,
  addMedia,
  copyFileIntoClipboard,
  fetchMediaPreview,
} from './media';
import {
  loadTaxonomy,
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
  deleteTag,
  loadTagsByMediaPath,
  selectNewPath,
} from './taxonomy';

import { loadFileMetaData } from './metadata';
import { createJob } from './jobs';

let db: Database | null = null;

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

// Window Controls
ipcMain.on('shutdown', async () => {
  // Shutdown the app.
  app.quit();
});

ipcMain.on('minimize', async () => {
  // Shutdown the app.
  mainWindow?.minimize();
});

ipcMain.on('open-external', async (event, args) => {
  const url = args[0];
  shell.openExternal(url);
});

ipcMain.on('toggle-fullscreen', async () => {
  // Shutdown the app.
  mainWindow?.setFullScreen(!mainWindow?.isFullScreen());
});

// Electron Store Provider
const store = new Store();
ipcMain.on('electron-store-get', async (event, key, defaultValue) => {
  event.returnValue = store.get(key, defaultValue);
});
ipcMain.on('electron-store-set', async (event, key, val) => {
  store.set(key, val);
});

ipcMain.handle('get-user-data-path', async (event) => {
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
  ipcMain.removeHandler('load-tags-by-media-path');
  ipcMain.removeHandler('copy-file-into-clipboard');
  ipcMain.removeHandler('load-taxonomy');
  ipcMain.removeHandler('create-tag');
  ipcMain.removeHandler('create-category');
  ipcMain.removeHandler('create-assignment');
  ipcMain.removeHandler('delete-assignment');
  ipcMain.removeHandler('update-assignment-weight');
  ipcMain.removeHandler('update-tag-weight');
  ipcMain.removeHandler('fetch-tag-preview');
  ipcMain.removeHandler('fetch-media-preview');
  ipcMain.removeHandler('add-media');

  ipcMain.removeHandler('select-new-path');
  ipcMain.removeHandler('rename-category');
  ipcMain.removeHandler('delete-category');
  ipcMain.removeHandler('rename-tag');
  ipcMain.removeHandler('move-tag');
  ipcMain.removeHandler('delete-tag');
  ipcMain.removeHandler('create-job');

  // Register Media Events
  ipcMain.handle('add-media', addMedia(db));
  ipcMain.handle('load-media-by-tags', loadMediaByTags(db));
  ipcMain.handle('copy-file-into-clipboard', copyFileIntoClipboard());

  // Register Metaata Events
  ipcMain.handle('load-tags-by-media-path', loadTagsByMediaPath(db));

  // Register Taxonomy Events
  ipcMain.handle('load-taxonomy', loadTaxonomy(db));
  ipcMain.handle('create-tag', createTag(db));
  ipcMain.handle('create-category', createCategory(db));
  ipcMain.handle('create-assignment', createAssignment(db, store));
  ipcMain.handle('delete-assignment', deleteAssignment(db));
  ipcMain.handle('update-assignment-weight', updateAssignmentWeight(db));
  ipcMain.handle('update-tag-weight', updateTagWeight(db));
  ipcMain.handle('fetch-tag-preview', fetchTagPreview(db));
  ipcMain.handle('fetch-media-preview', fetchMediaPreview(store));

  ipcMain.handle('select-new-path', selectNewPath(db, mainWindow));
  ipcMain.handle('rename-category', renameCategory(db));
  ipcMain.handle('delete-category', deleteCategory(db));
  ipcMain.handle('rename-tag', renameTag(db));
  ipcMain.handle('move-tag', moveTag(db));
  ipcMain.handle('delete-tag', deleteTag(db));
  if (!mainWindow) return;
  ipcMain.handle('create-job', createJob(db, mainWindow.webContents));
});

ipcMain.handle('select-db', async () => {
  invariant(mainWindow, 'mainWindow is not defined');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'promptToCreate', 'dontAddToRecent'],
    filters: [{ name: 'Lowkey Media Database', extensions: ['sqlite'] }],
  });

  if (!result.canceled) {
    console.log('SELECTED FILE PATH:', result);
    return result.filePaths[0];
  } else {
    return null;
  }
});

// Handle file selection event from renderer process
ipcMain.handle('select-file', async () => {
  invariant(mainWindow, 'mainWindow is not defined');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
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
      { name: 'Movies', extensions: ['mp4', 'mkv', 'webm'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });

  if (!result.canceled) {
    return result.filePaths[0];
  } else {
    return null;
  }
});

ipcMain.handle('load-file-metadata', loadFileMetaData);

if (process.env.NODE_ENV === 'production') {
  const sourceMapSupport = require('source-map-support');
  sourceMapSupport.install();
}

const isDebug =
  process.env.NODE_ENV === 'development' || process.env.DEBUG_PROD === 'true';

if (isDebug) {
  require('electron-debug')();
}

const installExtensions = async () => {
  const installer = require('electron-devtools-installer');
  const forceDownload = !!process.env.UPGRADE_EXTENSIONS;
  const extensions = ['REACT_DEVELOPER_TOOLS'];

  return installer
    .default(
      extensions.map((name) => installer[name]),
      forceDownload
    )
    .catch(console.log);
};

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
