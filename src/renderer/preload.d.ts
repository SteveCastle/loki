import { Channels } from 'main/preload';
import type { VttCue } from 'main/parse-vtt';
import {
  FilterModeOption,
  FilterOption,
  OrderingOption,
  SortByOption,
} from 'settings';

declare global {
  interface Window {
    appArgs: {
      filePath: string;
      allArgs: string[];
      appUserData: string;
      dbPath: string;
    };
    electron: {
      userHome: string;
      store: {
        get(key: string, defaultValue: any): unknown;
        set(property: string, val: unknown): void;
      };
      url: {
        format: typeof import('url').format;
      };
      loadFiles: (
        targetDir: string,
        filters: FilterOption,
        sortBy: SortByOption,
        recursive: boolean
      ) => Promise<{ path: string }[]>;
      loadMediaFromDB: (
        tag: string[],
        mode: FilterModeOption
      ) => Promise<{ path: string }[]>;
      fetchTagPreview: (tag: string) => Promise<string>;
      fetchMediaPreview: (
        path: string,
        cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
        timeStamp?: number
      ) => Promise<string>;
      loadTranscript: (filePath: string) => Promise<VttCue[]>;
      loadTaxonomy: () => Promise<string[]>;
      ipcRenderer: {
        invoke(channel: Channels, args: unknown[]): Promise<unknown>;
        removeListener(
          channel: string,
          func: (...args: unknown[]) => void
        ): void;
        sendMessage(channel: Channels, args: unknown[]): void;
        on(
          channel: string,
          func: (...args: unknown[]) => void
        ): (() => void) | undefined;
        once(channel: string, func: (...args: unknown[]) => void): void;
      };
    };
  }
}

export {};
