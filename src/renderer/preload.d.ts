import { Channels } from 'main/preload';
import type { VttCue } from 'main/parse-vtt';
import { FilterModeOption } from 'settings';

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
      loadMediaFromDB: (
        tag: string[],
        mode: FilterModeOption
      ) => Promise<{ path: string }[]>;
      loadMediaByDescriptionSearch: (
        description: string,
        tags?: string[],
        filteringMode?: string
      ) => Promise<{ path: string }[]>;
      fetchTagPreview: (tag: string) => Promise<string>;
      fetchTagCount: (tag: string) => Promise<number>;
      fetchMediaPreview: (
        path: string,
        cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
        timeStamp?: number
      ) => Promise<string>;
      loadTranscript: (filePath: string) => Promise<VttCue[]>;
      modifyTranscript: (input: {
        mediaPath: string;
        cueIndex: number;
        startTime?: string;
        endTime?: string;
        text?: string;
      }) => Promise<boolean>;
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
