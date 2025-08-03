import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';
import * as url from 'url';
import * as path from 'path';
import { isValidFilePath } from './file-handling';
import { loadTranscript, modifyTranscript } from './transcript';
import { FilterModeOption } from 'settings';

export type Channels =
  | 'shutdown'
  | 'select-file'
  | 'select-directory'
  | 'load-files'
  | 'select-new-path'
  | 'select-db'
  | 'load-db'
  | 'open-external'
  | 'toggle-fullscreen'
  | 'set-always-on-top'
  | 'add-media'
  | 'update-elo'
  | 'update-description'
  | 'copy-file-into-clipboard'
  | 'load-taxonomy'
  | 'get-tag-count'
  | 'load-file-metadata'
  | 'load-tags-by-media-path'
  | 'create-tag'
  // Job-related IPC handlers removed - now handled by external job runner service
  | 'create-category'
  | 'rename-category'
  | 'delete-category'
  | 'rename-tag'
  | 'move-tag'
  | 'order-tags'
  | 'create-assignment'
  | 'fetch-tag-preview'
  | 'fetch-media-preview'
  | 'update-tag-weight'
  | 'delete-assignment'
  | 'delete-tag'
  | 'update-assignment-weight'
  | 'update-timestamp'
  | 'remove-timestamp'
  | 'generate-transcript'
  | 'modify-transcript'
  | 'delete-file'
  | 'minimize';

const loadMediaFromDB = async (
  tags: string[],
  mode: FilterModeOption = 'EXCLUSIVE'
) => {
  const files = await ipcRenderer.invoke('load-media-by-tags', [tags, mode]);
  return files;
};

const loadMediaByDescriptionSearch = async (
  description: string,
  tags?: string[],
  filteringMode?: string
) => {
  const files = await ipcRenderer.invoke('load-media-by-description-search', [
    description,
    tags,
    filteringMode,
  ]);
  return files;
};

const fetchTagPreview = async (tag: string) => {
  const results = await ipcRenderer.invoke('fetch-tag-preview', [tag]);
  if (!results) return null;
  return results;
};

const fetchTagCount = async (tag: string) => {
  const count = await ipcRenderer.invoke('get-tag-count', [tag]);
  return count;
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
  loadMediaFromDB,
  loadMediaByDescriptionSearch,
  fetchTagPreview,
  fetchTagCount,
  fetchMediaPreview,
  loadTranscript,
  modifyTranscript,
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
