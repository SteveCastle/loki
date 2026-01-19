import { Channels } from 'main/preload';
import type { VttCue } from 'main/parse-vtt';
import { FilterModeOption } from 'settings';

// Update check result type
interface UpdateCheckResult {
  currentVersion: string;
  latestVersion: string | null;
  updateAvailable: boolean;
  error: string | null;
}

// Session store data types
type SessionKey = 'library' | 'cursor' | 'query' | 'previous';

interface SessionLibraryData {
  library: Array<{
    path: string;
    tagLabel?: string;
    mtimeMs: number;
    weight?: number;
    timeStamp?: number;
    elo?: number | null;
    description?: string;
    height?: number | null;
    width?: number | null;
  }>;
  initialFile: string;
}

interface SessionCursorData {
  cursor: number;
  scrollPosition?: number;
}

interface SessionQueryData {
  dbQuery: {
    tags: string[];
  };
  mostRecentTag: string;
  mostRecentCategory: string;
  textFilter: string;
}

// State type for tracking which mode the library was loaded from
type LibraryStateType = 'fs' | 'db' | 'search';

interface SessionPreviousData {
  previousLibrary: SessionLibraryData['library'];
  previousCursor: number;
  // State type tracking for proper restoration
  previousStateType?: LibraryStateType | null;
  previousTextFilter?: string;
  previousDbQuery?: { tags: string[] };
}

interface SessionData {
  library: SessionLibraryData | null;
  cursor: SessionCursorData | null;
  query: SessionQueryData | null;
  previous: SessionPreviousData | null;
  version?: number;
}

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
      // Config store (synchronous, for settings that rarely change)
      store: {
        get(key: string, defaultValue: any): unknown;
        set(property: string, val: unknown): void;
        getMany(pairs: [string, any][]): Record<string, unknown>;
      };
      // Session store (async, for frequently-changing ephemeral data)
      sessionStore: {
        get(key: SessionKey): Promise<SessionData[typeof key]>;
        getAll(): Promise<SessionData>;
        set(key: SessionKey, value: any): Promise<void>;
        setMany(updates: Partial<SessionData>): Promise<void>;
        clear(): Promise<void>;
        clearKeys(keys: SessionKey[]): Promise<void>;
        flush(): Promise<void>;
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
      loadDuplicatesByPath: (
        path: string
      ) => Promise<{ library: { path: string }[]; cursor: number }>;
      mergeDuplicatesByPath: (path: string) => Promise<{
        mergedInto: string;
        deleted: string[];
        copiedTags: number;
      }>;
      getGifMetadata: (filePath: string) => Promise<{
        frameCount: number;
        duration: number;
      } | null>;
      fetchTagPreview: (tag: string) => Promise<string>;
      fetchTagCount: (tag: string) => Promise<number>;
      fetchMediaPreview: (
        path: string,
        cache: 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
        timeStamp?: number
      ) => Promise<string>;
      listThumbnails: (path: string) => Promise<
        {
          cache:
            | 'thumbnail_path_100'
            | 'thumbnail_path_600'
            | 'thumbnail_path_1200';
          path: string;
          exists: boolean;
          size: number;
        }[]
      >;
      regenerateThumbnail: (
        path: string,
        cache:
          | 'thumbnail_path_100'
          | 'thumbnail_path_600'
          | 'thumbnail_path_1200',
        timeStamp?: number
      ) => Promise<string>;
      transcript: {
        loadTranscript: (filePath: string) => Promise<VttCue[] | null>;
        modifyTranscript: (input: {
          mediaPath: string;
          cueIndex: number;
          startTime?: string;
          endTime?: string;
          text?: string;
        }) => Promise<boolean>;
        checkIfWhisperIsInstalled: () => Promise<boolean>;
      };
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
