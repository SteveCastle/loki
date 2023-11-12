import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import * as url from 'url';
import * as path from 'path';
import { isValidFilePath } from './file-handling';
import loadFiles from './load-files';
import { loadTranscript } from './transcript';
import { FilterModeOption } from 'settings';

export type Channels =
  | 'shutdown'
  | 'select-file'
  | 'select-new-path'
  | 'select-db'
  | 'load-db'
  | 'open-external'
  | 'toggle-fullscreen'
  | 'add-media'
  | 'copy-file-into-clipboard'
  | 'load-taxonomy'
  | 'load-file-metadata'
  | 'load-tags-by-media-path'
  | 'create-tag'
  | 'create-job'
  | 'get-jobs'
  | 'complete-job'
  | 'create-category'
  | 'rename-category'
  | 'delete-category'
  | 'rename-tag'
  | 'move-tag'
  | 'create-assignment'
  | 'fetch-tag-preview'
  | 'fetch-media-preview'
  | 'update-tag-weight'
  | 'delete-assignment'
  | 'delete-tag'
  | 'update-assignment-weight'
  | 'generate-transcript'
  | 'minimize';

const loadMediaFromDB = async (
  tags: string[],
  mode: FilterModeOption = 'EXCLUSIVE'
) => {
  const files = await ipcRenderer.invoke('load-media-by-tags', [tags, mode]);
  return files;
};

const fetchTagPreview = async (tag: string) => {
  const results = await ipcRenderer.invoke('fetch-tag-preview', [tag]);
  if (!results) return null;
  return results;
};

const fetchMediaPreview = async (
  tag: string,
  cache: string,
  timeStamp: number
) => {
  const results = await ipcRenderer.invoke('fetch-media-preview', [
    tag,
    cache,
    timeStamp,
  ]);
  if (!results) return null;
  return results;
};

contextBridge.exposeInMainWorld('electron', {
  loadFiles,
  loadMediaFromDB,
  fetchTagPreview,
  fetchMediaPreview,
  loadTranscript,
  userHome: path.join(process.env.HOME || '', '.lowkey', 'dream.sqlite'),
  store: {
    get(key: string, defaultValue: any) {
      return ipcRenderer.sendSync('electron-store-get', key, defaultValue);
    },
    set(property: string, val: any) {
      ipcRenderer.send('electron-store-set', property, val);
    },
  },
  url: {
    format: url.format,
  },
  ipcRenderer: {
    sendMessage(channel: Channels, args: unknown[]) {
      ipcRenderer.send(channel, args);
    },
    invoke(channel: Channels, args: unknown[]) {
      return ipcRenderer.invoke(channel, args);
    },
    on(channel: Channels, func: (...args: unknown[]) => void) {
      const subscription = (_event: IpcRendererEvent, ...args: unknown[]) =>
        func(...args);
      ipcRenderer.on(channel, subscription);

      return () => ipcRenderer.removeListener(channel, subscription);
    },
    once(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.once(channel, (_event, ...args) => func(...args));
    },
    removeListener(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.removeListener(channel, func);
    },
  },
});

// Get the electron main process args from ipc and expose to mainWorld.
async function loadMainArgs() {
  const mainProcessArgs = await ipcRenderer.invoke('get-main-args');
  const appUserData = await ipcRenderer.invoke('get-user-data-path');
  const macPath = await ipcRenderer.invoke('get-mac-path');
  const filePath = isValidFilePath(mainProcessArgs[1])
    ? mainProcessArgs[1]
    : macPath;
  contextBridge.exposeInMainWorld('appArgs', {
    filePath,
    appUserData,
    dbPath: path.join(appUserData, 'dream.sqlite'),
    allArgs: process.argv,
  });
}
loadMainArgs();
