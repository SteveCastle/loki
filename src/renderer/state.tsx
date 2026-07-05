import React, { createContext } from 'react';
import { useInterpret } from '@xstate/react';
import { uniqueId } from 'lodash';
import path from 'path-browserify';
import { AnyEventObject, assign, createMachine, InterpreterFrom } from 'xstate';
import {
  DetailImageCache,
  FilterModeOption,
  Settings,
  clampVolume,
} from 'settings';
import {
  invoke, send, on, store, appArgs, capabilities, isElectron, mediaServerBase,
  loadMediaByQuery as platformLoadMediaByQuery,
} from './platform';
import type { Query, Predicate } from './query/types';
import { predicateKey } from './query/types';
import {
  addPredicateWithMode,
  removePredicate,
  toggleExclude,
  applyTagClick,
  setPredicateJoin,
  tagsFromQuery,
} from './query/reducer';
import { parseQuery } from './query/parse';
import filter from './filter';
import {
  initSessionStore,
  getSessionValue,
  setSessionValue,
  setSessionValues,
  clearSessionKeys,
  flushSession,
  hasPersistedLibrary as checkHasPersistedLibrary,
  hasPersistedTags as checkHasPersistedTags,
  hasPersistedQuery as checkHasPersistedQuery,
} from './hooks/useSessionStore';
// Job management removed - now handled by external job runner service

export type Item = {
  path: string;
  tagLabel?: string;
  mtimeMs: number;
  weight?: number;
  timeStamp?: number;
  elo?: number | null;
  description?: string;
  height?: number | null;
  width?: number | null;
  score?: number;
};

type Props = {
  children: React.ReactNode;
};

// State type for tracking which mode the library was loaded from
type LibraryStateType = 'fs' | 'db';

type LibraryState = {
  initialFile: string;
  dbPath: string;
  library: Item[];
  libraryLoadId: string;
  initSessionId: string;
  previousLibrary: Item[];
  cursor: number;
  textFilter: string;
  // Track current and previous state types for proper restoration
  currentStateType: LibraryStateType;
  previousStateType: LibraryStateType | null;
  previousTextFilter: string;
  previousDbQuery: { tags: string[] };
  previousQuery: Query;
  // Path the library was loaded for. Captured alongside previousLibrary
  // so back-restoration keeps the UI's path/filter/search/library
  // coherent — without this, library and initialFile drift apart.
  previousInitialFile: string;
  activeCategory: string;
  storedCategories: {
    [key: string]: string;
  };
  storedTags: {
    [key: string]: string[];
  };
  mostRecentTag: string;
  mostRecentCategory: string;
  previousCursor: number;
  settings: Settings;
  hotKeys: {
    [key: string]: string;
  };
  scrollPosition: number;
  previousScrollPosition: number;
  videoPlayer: {
    eventId: string;
    timeStamp: number;
    playing: boolean;
    actualVideoTime: number;
    videoLength: number;
    loopLength: number;
    loopStartTime: number;
    loopCount: number;
    // Detected (or 0 = unknown) frame rate of the loaded video, used to drive
    // single-frame stepping in the controls. See src/renderer/video-frame.ts.
    frameRate: number;
    // Playback speed multiplier (1 = normal), applied to the media element.
    playbackRate: number;
  };
  // Audio tracks discovered on the currently-loaded <video> element. Reset
  // to [] when the path changes; populated on `loadedmetadata`. The list
  // is sourced from HTMLMediaElement.audioTracks (gated by the
  // --enable-blink-features=AudioVideoTracks Chromium flag).
  availableAudioTracks: Array<{
    id: string;
    label: string;
    language: string;
  }>;
  // Index into availableAudioTracks of the user-selected (or default)
  // track. Always 0 on a fresh path; never persisted.
  selectedAudioTrackIndex: number;
  // Sidecar subtitle for the current path. The blob URL is owned by the
  // renderer and revoked when the path changes or the file unloads.
  availableSubtitle: {
    blobUrl: string;
    label: string;
  } | null;
  dbQuery: {
    tags: string[];
  };
  query: Query;
  commandPalette: {
    display: boolean;
    position: { x: number; y: number };
  };
  contextPalette: {
    display: boolean;
    position: { x: number; y: number };
    target:
      | { type: 'library' }
      | { type: 'file'; path: string }
      | { type: 'tag'; tag: string }
      | { type: 'category'; category: string };
  };
  // jobs: removed - now handled by external job runner service
  toasts: {
    id: string;
    type: 'success' | 'error' | 'info';
    title: string;
    message?: string;
    timestamp: number;
  }[];
  // Streaming state to stabilize ordering and selection during directory scans
  streaming: boolean;
  pinnedPath: string | null;
  savedSortByDuringStreaming: Settings['sortBy'] | null;
  userMovedCursorDuringStreaming: boolean;
  // When set to a new unique ID, signals the list to scroll to the current cursor
  scrollToCursorEventId: string | null;
  // libraryLoadId captured at the moment an in-place mutation (e.g. removing
  // a tag from an image) requested its reload. When the list sees that the
  // libraryLoadId transition's "from" value matches this, it preserves scroll
  // instead of resetting. Stale values can't match a future transition's
  // "from" value, so no explicit clear is needed.
  preserveScrollFromLoadId: string | null;
  authToken: string | null;
  // Cache for masonry layout dimensions to maintain stable layout across view switches
  masonryDimensionsCache: Record<string, { width: number; height: number }>;
};

const queryHasVisual = (predicates: Predicate[] = []): boolean =>
  predicates.some(
    (p) => p.type === 'similar' || p.type === 'visual' || p.type === 'clip'
  );

const applySimilaritySort = assign<LibraryState, AnyEventObject>({
  settings: (context) => {
    const hasVisual = queryHasVisual(context.query?.predicates);
    if (hasVisual) {
      return { ...context.settings, sortBy: 'similarity' };
    }
    // Leaving a visual query: if we were on 'similarity', fall back to 'name'.
    if (context.settings.sortBy === 'similarity') {
      return { ...context.settings, sortBy: 'name' };
    }
    return context.settings;
  },
});

const addQueryErrorToast = assign<LibraryState, AnyEventObject>({
  toasts: (context, event) => {
    const err = event.data;
    const message =
      'Query failed: ' +
      (err?.message ?? (typeof err === 'string' ? err : 'unknown error'));
    const newToast = {
      id: uniqueId(),
      type: 'error' as const,
      title: 'Query Error',
      message,
      timestamp: Date.now(),
    };
    return [...context.toasts, newToast];
  },
});

const setLibrary = assign<LibraryState, AnyEventObject>({
  library: (context, event) => {
    const library = event.data.library;
    // Update library and cursor data using session store (async, debounced)
    setSessionValues({
      library: { library, initialFile: context.initialFile },
      cursor: { cursor: event.data.cursor },
    });
    return library;
  },
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
});

// Atomically snapshot the current state into the previous-state slot. Use
// this instead of inlining the six previous* assigns at each save site so a
// future edit cannot accidentally save five of six fields and leave the
// restore inconsistent.
const capturePrevious = assign<LibraryState, AnyEventObject>({
  previousLibrary: (context) => context.library,
  previousCursor: (context) => context.cursor,
  previousStateType: (context) => context.currentStateType,
  previousTextFilter: (context) => context.textFilter,
  previousDbQuery: (context) => ({ ...context.dbQuery }),
  previousQuery: (context) => ({ predicates: [...context.query.predicates] }),
  previousInitialFile: (context) => context.initialFile,
});

// Capture the current view into the previous-state slot ONLY if nothing is
// stored there yet. The query-mutation handlers use this so the first filter
// applied from a view (typically FS) snapshots that view — letting a later
// "removed the last predicate" restore it from memory (loadingFromPreviousLibrary).
// Subsequent edits (slot already full) leave the snapshot intact.
const capturePreviousIfEmpty = assign<LibraryState, AnyEventObject>({
  previousLibrary: (c) =>
    c.previousLibrary.length > 0 ? c.previousLibrary : c.library,
  previousCursor: (c) =>
    c.previousLibrary.length > 0 ? c.previousCursor : c.cursor,
  previousStateType: (c) =>
    c.previousLibrary.length > 0 ? c.previousStateType : c.currentStateType,
  previousTextFilter: (c) =>
    c.previousLibrary.length > 0 ? c.previousTextFilter : c.textFilter,
  previousDbQuery: (c) =>
    c.previousLibrary.length > 0 ? c.previousDbQuery : { ...c.dbQuery },
  previousQuery: (c) =>
    c.previousLibrary.length > 0
      ? c.previousQuery
      : { predicates: [...c.query.predicates] },
  previousInitialFile: (c) =>
    c.previousLibrary.length > 0 ? c.previousInitialFile : c.initialFile,
});

const setLibraryWithPrevious = assign<LibraryState, AnyEventObject>({
  // Only save previous state if not already saved by an action (check if previousLibrary is empty)
  previousLibrary: (context) =>
    context.previousLibrary.length > 0
      ? context.previousLibrary
      : context.library,
  previousCursor: (context) =>
    context.previousLibrary.length > 0
      ? context.previousCursor
      : context.cursor,
  previousStateType: (context) =>
    context.previousLibrary.length > 0
      ? context.previousStateType
      : context.currentStateType,
  previousTextFilter: (context) =>
    context.previousLibrary.length > 0
      ? context.previousTextFilter
      : context.textFilter,
  previousDbQuery: (context) =>
    context.previousLibrary.length > 0
      ? context.previousDbQuery
      : { ...context.dbQuery },
  previousQuery: (context) =>
    context.previousLibrary.length > 0
      ? context.previousQuery
      : { predicates: [...context.query.predicates] },
  previousInitialFile: (context) =>
    context.previousLibrary.length > 0
      ? context.previousInitialFile
      : context.initialFile,
  library: (context, event) => {
    const library = event.data.library;
    // Use existing previous state if already set, otherwise use current state
    const hasPrevious = context.previousLibrary.length > 0;
    const previousLibrary = hasPrevious
      ? context.previousLibrary
      : context.library;
    const previousCursor = hasPrevious
      ? context.previousCursor
      : context.cursor;
    const previousStateType = hasPrevious
      ? context.previousStateType
      : context.currentStateType;
    const previousTextFilter = hasPrevious
      ? context.previousTextFilter
      : context.textFilter;
    const previousDbQuery = hasPrevious
      ? context.previousDbQuery
      : context.dbQuery;
    const previousQuery = hasPrevious
      ? context.previousQuery
      : context.query;
    const previousInitialFile = hasPrevious
      ? context.previousInitialFile
      : context.initialFile;

    // Update all session data using session store (async, debounced, batched)
    setSessionValues({
      library: { library, initialFile: context.initialFile },
      cursor: { cursor: event.data.cursor },
      previous: {
        previousLibrary,
        previousCursor,
        previousStateType,
        previousTextFilter,
        previousDbQuery,
        previousQuery,
        previousInitialFile,
      },
    });

    return library;
  },
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
});

const clearPersistedLibrary = () => {
  // Clear all session data using the new session store
  clearSessionKeys(['library', 'cursor', 'query', 'previous']);
};

const updatePersistedCursor = (context: LibraryState, cursor: number) => {
  // Only update cursor - uses session store with debouncing for high performance
  setSessionValue('cursor', {
    cursor,
    scrollPosition: context.scrollPosition,
  });
};

const updatePersistedState = (context: LibraryState) => {
  // Update query state using session store (async, debounced)
  setSessionValue('query', {
    dbQuery: context.dbQuery,
    query: context.query,
    mostRecentTag: context.mostRecentTag,
    mostRecentCategory: context.mostRecentCategory,
    textFilter: context.textFilter,
  });
};

// Mirror the just-captured previous* fields from context to the session store.
// Why: capturing previous in the assign only updates in-memory context. If the
// app closes (or a beforeunload flush fires) before setLibraryWithPrevious runs
// at the end of the load, the on-disk previous would lag by one transition.
const persistPreviousState = (context: LibraryState) => {
  setSessionValue('previous', {
    previousLibrary: context.previousLibrary,
    previousCursor: context.previousCursor,
    previousStateType: context.previousStateType,
    previousTextFilter: context.previousTextFilter,
    previousDbQuery: context.previousDbQuery,
    previousQuery: context.previousQuery,
    previousInitialFile: context.previousInitialFile,
  });
};

const setPath = assign<LibraryState, AnyEventObject>({
  initialFile: (context, event) => {
    if (!event.data) {
      return context.initialFile;
    }
    clearPersistedLibrary();
    return event.data;
  },
  // Only reset filter setting when the user actually picked a new path.
  // Cancelling the picker should leave filters untouched.
  settings: (context, event) => {
    if (!event.data) return context.settings;
    return {
      ...context.settings,
      filters: 'all',
    };
  },
  // Wipe in-memory previous-state slot when starting in a new workspace,
  // so a back-navigation doesn't restore a library from the prior path.
  previousLibrary: (context, event) => (event.data ? [] : context.previousLibrary),
  previousCursor: (context, event) => (event.data ? 0 : context.previousCursor),
  previousStateType: (context, event) =>
    event.data ? null : context.previousStateType,
  previousTextFilter: (context, event) =>
    event.data ? '' : context.previousTextFilter,
  previousDbQuery: (context, event) =>
    event.data ? { tags: [] } : context.previousDbQuery,
  previousQuery: (context, event) =>
    event.data ? { predicates: [] } : context.previousQuery,
  previousInitialFile: (context, event) =>
    event.data ? '' : context.previousInitialFile,
});

const updateFilePath = assign<LibraryState, AnyEventObject>({
  library: (context, event) => {
    console.log('updateFilePath', event, context);
    const { data } = event;
    if (!data) {
      return context.library;
    }
    const library = [...context.library];
    const item = library.find((item) => item.path === data.path);
    if (item) {
      item.path = data.newPath;
    }
    return library;
  },
});

// createJob removed - jobs now handled by external job runner service

const setDB = assign<LibraryState, AnyEventObject>({
  dbPath: (context, event) => {
    console.log('set dbPath', event);
    if (!event.data) {
      return context.dbPath;
    }
    // Clear session store when changing to a different database
    if (event.data !== context.dbPath) {
      clearSessionKeys(['library', 'cursor', 'query', 'previous']);
    }
    // Only persist real file paths, not the web-mode placeholder
    if (event.data !== 'web') {
      store.set('dbPath', event.data);
    }
    return event.data;
  },
  // Clear initialFile when changing databases so init state doesn't use stale data
  initialFile: (context, event) => {
    if (!event.data || event.data === context.dbPath) {
      return context.initialFile;
    }
    return '';
  },
  // Wipe in-memory previous-state slot when switching databases so a
  // back-navigation can't surface a library from the prior DB.
  previousLibrary: (context, event) =>
    event.data && event.data !== context.dbPath ? [] : context.previousLibrary,
  previousCursor: (context, event) =>
    event.data && event.data !== context.dbPath ? 0 : context.previousCursor,
  previousStateType: (context, event) =>
    event.data && event.data !== context.dbPath
      ? null
      : context.previousStateType,
  previousTextFilter: (context, event) =>
    event.data && event.data !== context.dbPath
      ? ''
      : context.previousTextFilter,
  previousDbQuery: (context, event) =>
    event.data && event.data !== context.dbPath
      ? { tags: [] }
      : context.previousDbQuery,
  previousQuery: (context, event) =>
    event.data && event.data !== context.dbPath
      ? { predicates: [] }
      : context.previousQuery,
  previousInitialFile: (context, event) =>
    event.data && event.data !== context.dbPath
      ? ''
      : context.previousInitialFile,
});

const hasInitialFile = (context: LibraryState) => !!context.initialFile;
const missingDb = (context: LibraryState) => !context.dbPath;

// These guards now use the session store cache (sync read from in-memory cache)
const hasPersistedLibrary = (_context: LibraryState): boolean => {
  return checkHasPersistedLibrary();
};

// Boot/restore routing guard: true when there is ANY persisted filter to
// restore — legacy tags OR a unified query holding non-tag predicates (path /
// category / description / hash). Routing on tags alone misclassified a
// path-only query as "no filter", sending it to the FS view whose entry then
// wiped the persisted predicate.
const hasPersistedFilter = (_context: LibraryState): boolean => {
  return checkHasPersistedTags() || checkHasPersistedQuery();
};

const willHaveTag = (context: LibraryState, event: AnyEventObject) => {
  // Detect if the result of toggling a tag is not an empty tag list.
  // If so return true.
  const tag = event.data.tag;
  const tagList = context.dbQuery.tags;
  const index = tagList.indexOf(tag);
  console.log(tag, tagList);
  const newTagList = [...tagList];
  if (index > -1) {
    newTagList.splice(index, 1);
  } else {
    newTagList.push(tag);
  }
  return newTagList.length !== 0;
};

const isEmpty = (context: LibraryState, event: AnyEventObject) => {
  return event.data.textFilter.length === 0;
};

const notEmpty = (context: LibraryState, event: AnyEventObject) => {
  return event.data.textFilter.length > 0;
};

const willHaveNoTag = (context: LibraryState, event: AnyEventObject) => {
  // Detect if the result of toggling a tag is an empty tag list.
  // If so return true.
  const tag = event.data.tag;
  const tagList = context.dbQuery.tags;
  const index = tagList.indexOf(tag);
  const newTagList = [...tagList];
  if (index > -1) {
    newTagList.splice(index, 1);
  } else {
    newTagList.push(tag);
  }
  return newTagList.length === 0;
};

// Memoize context initialization to prevent unnecessary re-computations
const getInitialContext = (): LibraryState => {
  const batched = store.getMany([
    ['dbPath', null],
    ['activeCategory', ''],
    ['storedCategories', {}],
    ['storedTags', {}],
    ['sortOrder', 'asc'],
    ['sortBy', 'name'],
    ['comicMode', false],
    ['showTagCount', false],
    ['battleMode', false],
    ['libraryLayout', 'bottom'],
    ['applyTagPreview', true],
    ['filteringMode', 'EXCLUSIVE'],
    ['applyTagToAll', false],
    ['scaleMode', 'fit'],
    ['playSound', false],
    ['followTranscript', true],
    ['showTags', 'all'],
    ['hideSuggestedTags', false],
    ['showFileInfo', 'none'],
    ['showControls', false],
    ['gridSize', [4, 4]],
    ['listImageCache', 'thumbnail_path_600'],
    ['detailImageCache', false],
    ['controlMode', 'mouse'],
    ['autoPlay', false],
    ['autoPlayTime', false],
    ['autoPlayVideoLoops', false],
    ['volume', 1.0],
    ['alwaysOnTop', false],
    ['useHLS', false],
    ['subtitlesEnabled', false],
    ['incrementCursor', 'arrowright'],
    ['decrementCursor', 'arrowleft'],
    ['toggleTagPreview', 'shift'],
    ['toggleTagAll', 'control'],
    ['moveToTop', '['],
    ['moveToEnd', ']'],
    ['minimize', 'escape'],
    ['shuffle', 'x'],
    ['copyFilePath', 'c+control'],
    ['copyAllSelectedFiles', 'c+control+shift'],
    ['deleteFile', 'delete'],
    ['applyMostRecentTag', 'a'],
    ['storeCategory1', '1+alt'],
    ['storeCategory2', '2+alt'],
    ['storeCategory3', '3+alt'],
    ['storeCategory4', '4+alt'],
    ['storeCategory5', '5+alt'],
    ['storeCategory6', '6+alt'],
    ['storeCategory7', '7+alt'],
    ['storeCategory8', '8+alt'],
    ['storeCategory9', '9+alt'],
    ['tagCategory1', '1+shift'],
    ['tagCategory2', '2+shift'],
    ['tagCategory3', '3+shift'],
    ['tagCategory4', '4+shift'],
    ['tagCategory5', '5+shift'],
    ['tagCategory6', '6+shift'],
    ['tagCategory7', '7+shift'],
    ['tagCategory8', '8+shift'],
    ['tagCategory9', '9+shift'],
    ['storeTag1', '1+control'],
    ['storeTag2', '2+control'],
    ['storeTag3', '3+control'],
    ['storeTag4', '4+control'],
    ['storeTag5', '5+control'],
    ['storeTag6', '6+control'],
    ['storeTag7', '7+control'],
    ['storeTag8', '8+control'],
    ['storeTag9', '9+control'],
    ['applyTag1', '1'],
    ['applyTag2', '2'],
    ['applyTag3', '3'],
    ['applyTag4', '4'],
    ['applyTag5', '5'],
    ['applyTag6', '6'],
    ['applyTag7', '7'],
    ['applyTag8', '8'],
    ['applyTag9', '9'],
    ['togglePlayPause', ' '],
    ['refreshLibrary', 'r'],
    ['layoutMode', 'grid'],
    ['authToken', null],
  ] as [string, any][]);

  return {
    initialFile: appArgs?.filePath || '',
    dbPath: batched['dbPath'] as string,
    library: [],
    libraryLoadId: '',
    initSessionId: '',
    textFilter: '',
    authToken: batched['authToken'] as string | null,
    activeCategory: batched['activeCategory'] as string,
    storedCategories: batched['storedCategories'] as { [key: string]: string },
    storedTags: batched['storedTags'] as { [key: string]: string[] },
    mostRecentTag: '',
    mostRecentCategory: '',
    cursor: 0,
    previousLibrary: [],
    previousCursor: 0,
    // State type tracking for proper restoration
    currentStateType: 'fs' as LibraryStateType,
    previousStateType: null,
    previousTextFilter: '',
    previousDbQuery: { tags: [] },
    previousQuery: { predicates: [] },
    previousInitialFile: '',
    scrollPosition: 0,
    previousScrollPosition: 0,
    availableAudioTracks: [],
    selectedAudioTrackIndex: 0,
    availableSubtitle: null,
    videoPlayer: {
      eventId: 'initial',
      timeStamp: 0,
      playing: true,
      videoLength: 0,
      actualVideoTime: 0,
      loopLength: 0,
      loopStartTime: 0,
      loopCount: 0,
      frameRate: 0,
      playbackRate: 1,
    },
    settings: {
      order: batched['sortOrder'] as 'asc' | 'desc',
      sortBy: batched['sortBy'] as
        | 'name'
        | 'date'
        | 'weight'
        | 'elo'
        | 'shuffle',
      filters: 'all',
      recursive: false,
      scale: 1,
      comicMode: batched['comicMode'] as boolean,
      showTagCount: batched['showTagCount'] as boolean,
      battleMode: batched['battleMode'] as boolean,
      libraryLayout: batched['libraryLayout'] as 'left' | 'bottom',
      applyTagPreview: batched['applyTagPreview'] as boolean,
      filteringMode: batched['filteringMode'] as FilterModeOption,
      applyTagToAll: batched['applyTagToAll'] as boolean,
      scaleMode: batched['scaleMode'] as 'fit' | 'cover' | number,
      playSound: batched['playSound'] as boolean,
      followTranscript: batched['followTranscript'] as boolean,
      showTags: batched['showTags'] as 'all' | 'list' | 'detail' | 'none',
      hideSuggestedTags: batched['hideSuggestedTags'] as boolean,
      showFileInfo: batched['showFileInfo'] as
        | 'all'
        | 'list'
        | 'detail'
        | 'none',
      showControls: batched['showControls'] as boolean,
      gridSize: batched['gridSize'] as [number, number],
      listImageCache: batched['listImageCache'] as
        | 'thumbnail_path_1200'
        | 'thumbnail_path_600'
        | false,
      detailImageCache: batched['detailImageCache'] as DetailImageCache,
      controlMode: batched['controlMode'] as 'mouse' | 'touchpad',
      autoPlay: batched['autoPlay'] as boolean,
      autoPlayTime: batched['autoPlayTime'] as number | false,
      autoPlayVideoLoops: batched['autoPlayVideoLoops'] as number | false,
      volume: batched['volume'] as number,
      alwaysOnTop: batched['alwaysOnTop'] as boolean,
      layoutMode: batched['layoutMode'] as 'grid' | 'masonry',
      useHLS: batched['useHLS'] as boolean,
      subtitlesEnabled: batched['subtitlesEnabled'] as boolean,
    },
    hotKeys: {
      incrementCursor: batched['incrementCursor'] as string,
      decrementCursor: batched['decrementCursor'] as string,
      toggleTagPreview: batched['toggleTagPreview'] as string,
      toggleTagAll: batched['toggleTagAll'] as string,
      moveToTop: batched['moveToTop'] as string,
      moveToEnd: batched['moveToEnd'] as string,
      minimize: batched['minimize'] as string,
      shuffle: batched['shuffle'] as string,
      copyFile: batched['copyFilePath'] as string,
      copyAllSelectedFiles: batched['copyAllSelectedFiles'] as string,
      deleteFile: batched['deleteFile'] as string,
      applyMostRecentTag: batched['applyMostRecentTag'] as string,
      storeCategory1: batched['storeCategory1'] as string,
      storeCategory2: batched['storeCategory2'] as string,
      storeCategory3: batched['storeCategory3'] as string,
      storeCategory4: batched['storeCategory4'] as string,
      storeCategory5: batched['storeCategory5'] as string,
      storeCategory6: batched['storeCategory6'] as string,
      storeCategory7: batched['storeCategory7'] as string,
      storeCategory8: batched['storeCategory8'] as string,
      storeCategory9: batched['storeCategory9'] as string,
      tagCategory1: batched['tagCategory1'] as string,
      tagCategory2: batched['tagCategory2'] as string,
      tagCategory3: batched['tagCategory3'] as string,
      tagCategory4: batched['tagCategory4'] as string,
      tagCategory5: batched['tagCategory5'] as string,
      tagCategory6: batched['tagCategory6'] as string,
      tagCategory7: batched['tagCategory7'] as string,
      tagCategory8: batched['tagCategory8'] as string,
      tagCategory9: batched['tagCategory9'] as string,
      storeTag1: batched['storeTag1'] as string,
      storeTag2: batched['storeTag2'] as string,
      storeTag3: batched['storeTag3'] as string,
      storeTag4: batched['storeTag4'] as string,
      storeTag5: batched['storeTag5'] as string,
      storeTag6: batched['storeTag6'] as string,
      storeTag7: batched['storeTag7'] as string,
      storeTag8: batched['storeTag8'] as string,
      storeTag9: batched['storeTag9'] as string,
      applyTag1: batched['applyTag1'] as string,
      applyTag2: batched['applyTag2'] as string,
      applyTag3: batched['applyTag3'] as string,
      applyTag4: batched['applyTag4'] as string,
      applyTag5: batched['applyTag5'] as string,
      applyTag6: batched['applyTag6'] as string,
      applyTag7: batched['applyTag7'] as string,
      applyTag8: batched['applyTag8'] as string,
      applyTag9: batched['applyTag9'] as string,
      togglePlayPause: batched['togglePlayPause'] as string,
      refreshLibrary: batched['refreshLibrary'] as string,
    },
    dbQuery: {
      tags: [],
    },
    query: { predicates: [] },
    commandPalette: {
      display: false,
      position: { x: 0, y: 0 },
    },
    contextPalette: {
      display: false,
      position: { x: 0, y: 0 },
      target: { type: 'library' } as LibraryState['contextPalette']['target'],
    },
    // jobs: removed - now handled by external job runner service
    toasts: [],
    streaming: false,
    pinnedPath: null,
    savedSortByDuringStreaming: null,
    userMovedCursorDuringStreaming: false,
    scrollToCursorEventId: null,
    preserveScrollFromLoadId: null,
    masonryDimensionsCache: {},
  };
};

// Shared query-mutation handlers. Defined once so they can be applied to both
// the loaded states (loadedFromFS/loadedFromDB) and the
// loading states (runningQuery/loadingFromDB). Applying them while
// a query is in flight makes chip edits
// feel snappy: the assign updates context.query immediately (so the chip UI
// updates) and targeting runningQuery restarts execution with the latest
// predicates (latest-wins).
const queryMutationOn = {
  ADD_PREDICATE: {
    target: 'runningQuery',
    actions: [
      // Snapshot the pre-query (e.g. FS) view so clearing back to empty can
      // restore it from memory.
      capturePreviousIfEmpty,
      assign<LibraryState, AnyEventObject>((context, event) => {
        // EXCLUSIVE mode replaces the entire query with the selected filter,
        // regardless of predicate type (tag/path/category/description/hash).
        const q = addPredicateWithMode(
          context.query,
          event.data.predicate,
          context.settings.filteringMode
        );
        // The new result set starts at the top. A mounted list view resets
        // itself on the libraryLoadId change, but when the query changes while
        // the list is unmounted (e.g. similar-search from the detail screen)
        // the persisted position would be restored stale on the next mount.
        return { query: q, dbQuery: { tags: tagsFromQuery(q) }, scrollPosition: 0 };
      }),
    ],
  },
  REMOVE_PREDICATE: [
    {
      // Removing the LAST predicate returns to the previous library (e.g. the
      // FS view) from memory — mirrors clearing the last tag / the search.
      target: 'loadingFromPreviousLibrary',
      cond: (context: LibraryState, event: AnyEventObject) =>
        removePredicate(context.query, event.data.key).predicates.length === 0,
    },
    {
      target: 'runningQuery',
      actions: assign<LibraryState, AnyEventObject>((context, event) => {
        const q = removePredicate(context.query, event.data.key);
        return { query: q, dbQuery: { tags: tagsFromQuery(q) }, scrollPosition: 0 };
      }),
    },
  ],
  TOGGLE_EXCLUDE: {
    target: 'runningQuery',
    actions: assign<LibraryState, AnyEventObject>((context, event) => {
      const q = toggleExclude(context.query, event.data.key);
      return { query: q, dbQuery: { tags: tagsFromQuery(q) }, scrollPosition: 0 };
    }),
  },
  SET_PREDICATE_JOIN: {
    target: 'runningQuery',
    actions: assign<LibraryState, AnyEventObject>((context, event) => {
      const q = setPredicateJoin(context.query, event.data.key, event.data.join);
      return { query: q, dbQuery: { tags: tagsFromQuery(q) }, scrollPosition: 0 };
    }),
  },
  SET_QUERY: [
    {
      // Clearing the text to an empty query returns to the previous library.
      target: 'loadingFromPreviousLibrary',
      cond: (_context: LibraryState, event: AnyEventObject) =>
        parseQuery(event.data.text).length === 0,
    },
    {
      target: 'runningQuery',
      actions: [
        capturePreviousIfEmpty,
        assign<LibraryState, AnyEventObject>((_context, event) => {
          const q = { predicates: parseQuery(event.data.text) };
          return { query: q, dbQuery: { tags: tagsFromQuery(q) }, scrollPosition: 0 };
        }),
      ],
    },
  ],
  CLEAR_QUERY: {
    // No actions: loadingFromPreviousLibrary's entry restores query/library
    // from the previous* snapshot (mirrors CLEAR_QUERY_TAG).
    target: 'loadingFromPreviousLibrary',
  },
};

export const libraryMachine = createMachine(
  {
    id: 'library',
    predictableActionArguments: true,
    type: 'parallel',
    context: getInitialContext(),
    on: {
      TOGGLE_PLAY_PAUSE: {
        actions: assign<LibraryState, AnyEventObject>({
          videoPlayer: (context, event) => {
            return {
              ...context.videoPlayer,
              playing: !context.videoPlayer.playing,
            };
          },
        }),
      },
      CACHE_MASONRY_DIMENSIONS: {
        actions: assign<LibraryState, AnyEventObject>({
          masonryDimensionsCache: (context, event) => {
            const { itemKey, width, height } = event;
            // Only update if we don't already have this item cached
            if (context.masonryDimensionsCache[itemKey]) {
              return context.masonryDimensionsCache;
            }
            return {
              ...context.masonryDimensionsCache,
              [itemKey]: { width, height },
            };
          },
        }),
      },
    },
    states: {
      // jobQueue: removed - jobs now handled by external job runner service
      settings: {
        on: {
          SET_AUTH_TOKEN: {
            actions: assign<LibraryState, AnyEventObject>({
              authToken: (context, event) => {
                const token = event.token;
                store.set('authToken', token);
                return token;
              },
            }),
          },
          CHANGE_SETTING: {
            actions: assign<LibraryState, AnyEventObject>({
              settings: (context, event) => {
                console.log('CHANGE_SETTING', context, event);
                const processedData = { ...event.data };

                for (const key in processedData) {
                  // Clamp volume to valid range [0, 1]
                  if (
                    key === 'volume' &&
                    typeof processedData[key] === 'number'
                  ) {
                    processedData[key] = clampVolume(processedData[key]);
                  }

                  store.set(key, processedData[key]);
                  // Handle alwaysOnTop setting specially
                  if (key === 'alwaysOnTop') {
                    send(
                      'set-always-on-top',
                      [processedData[key]]
                    );
                  }
                }
                return {
                  ...context.settings,
                  ...processedData,
                };
              },
              libraryLoadId: (context, event) => {
                const nextSortBy = event?.data?.sortBy as
                  | Settings['sortBy']
                  | undefined;
                if (
                  nextSortBy === 'shuffle' &&
                  context.settings.sortBy !== 'shuffle'
                ) {
                  return uniqueId();
                }
                return context.libraryLoadId;
              },
            }),
          },
          SET_AVAILABLE_AUDIO_TRACKS: {
            actions: assign<LibraryState, AnyEventObject>({
              availableAudioTracks: (_context, event) => event.tracks,
              // Reset selection when the track list changes (new video load).
              selectedAudioTrackIndex: () => 0,
            }),
          },
          SET_AUDIO_TRACK: {
            actions: assign<LibraryState, AnyEventObject>({
              selectedAudioTrackIndex: (_context, event) => event.index,
            }),
          },
          SET_AVAILABLE_SUBTITLE: {
            actions: assign<LibraryState, AnyEventObject>({
              availableSubtitle: (_context, event) => event.subtitle,
            }),
          },
          CHANGE_HOTKEY: {
            actions: assign<LibraryState, AnyEventObject>({
              hotKeys: (context, event) => {
                console.log('CHANGE_HOTKEY', context, event);
                for (const key in event.data) {
                  store.set(key, event.data[key]);
                }
                return {
                  ...context.hotKeys,
                  ...event.data,
                };
              },
            }),
          },
          STORE_CATEGORY: {
            actions: [
              assign<LibraryState, AnyEventObject>({
                storedCategories: (context, event) => {
                  console.log('STORE_CATEGORY', context, event);
                  const { category, position } = event.data;
                  store.set(`storedCategories`, {
                    ...context.storedCategories,
                    [position]: category,
                  });
                  return {
                    ...context.storedCategories,
                    [position]: category,
                  };
                },
              }),
              assign<LibraryState, AnyEventObject>({
                toasts: (context, event) => {
                  const { category, position } = event.data;
                  const newToast = {
                    id: uniqueId(),
                    type: 'success' as const,
                    title: `Slot ${position}`,
                    message: category,
                    timestamp: Date.now(),
                  };
                  return [...context.toasts, newToast];
                },
              }),
            ],
          },
          STORE_TAG: {
            actions: [
              assign<LibraryState, AnyEventObject>({
                storedTags: (context, event) => {
                  console.log('STORE_TAG', context, event);
                  const { tags, position } = event.data;
                  store.set(`storedTags`, {
                    ...context.storedTags,
                    [position]: tags,
                  });
                  return {
                    ...context.storedTags,
                    [position]: tags,
                  };
                },
              }),
              assign<LibraryState, AnyEventObject>({
                toasts: (context, event) => {
                  const { tags, position } = event.data;
                  const newToast = {
                    id: uniqueId(),
                    type:
                      tags.length === 0
                        ? ('error' as const)
                        : ('success' as const),
                    title: `Slot ${position}`,
                    message:
                      tags.length === 0 ? 'No Tags Selected' : tags.join(', '),
                    timestamp: Date.now(),
                  };
                  return [...context.toasts, newToast];
                },
              }),
            ],
          },
          SET_MOST_RECENT_TAG: {
            actions: [
              assign<LibraryState, AnyEventObject>({
                mostRecentTag: (context, event) => {
                  console.log('SET_MOST_RECENT_TAG', context, event);
                  const newTag = event.tag;
                  updatePersistedState({
                    ...context,
                    mostRecentTag: newTag,
                    mostRecentCategory: event.category,
                  });
                  return newTag;
                },
                mostRecentCategory: (context, event) => {
                  console.log('SET_MOST_RECENT_CATEGORY', context, event);
                  return event.category;
                },
              }),
            ],
          },
          ADD_TOAST: {
            actions: assign<LibraryState, AnyEventObject>({
              toasts: (context, event) => {
                const newToast = {
                  id: uniqueId(),
                  type: event.data.type || 'info',
                  title: event.data.title,
                  message: event.data.message,
                  timestamp: Date.now(),
                  durationMs: event.data.durationMs,
                };
                return [...context.toasts, newToast];
              },
            }),
          },
          REMOVE_TOAST: {
            actions: assign<LibraryState, AnyEventObject>({
              toasts: (context, event) => {
                return context.toasts.filter(
                  (toast) => toast.id !== event.data.id
                );
              },
            }),
          },
        },
      },
      commandPalette: {
        on: {
          SHOW_COMMAND_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              commandPalette: (context, event) => {
                return {
                  display: true,
                  position: event.position,
                };
              },
            }),
          },
          HIDE_COMMAND_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              commandPalette: (context, event) => {
                return {
                  display: false,
                  position: event.position,
                };
              },
            }),
          },
        },
      },
      contextPalette: {
        on: {
          SHOW_CONTEXT_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context, event) => {
                return {
                  display: true,
                  position: event.position,
                  target: event.target || { type: 'library' },
                };
              },
              commandPalette: (context) => {
                return {
                  display: false,
                  position: context.commandPalette.position,
                };
              },
            }),
          },
          HIDE_CONTEXT_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context) => {
                return {
                  ...context.contextPalette,
                  display: false,
                };
              },
            }),
          },
          SHOW_COMMAND_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context) => {
                return {
                  ...context.contextPalette,
                  display: false,
                };
              },
            }),
          },
        },
      },
      videoPlayer: {
        on: {
          SET_VIDEO_TIME: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  timeStamp: event.timeStamp,
                  eventId: event.eventId,
                };
              },
            }),
          },
          SET_ACTUAL_VIDEO_TIME: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  actualVideoTime: event.timeStamp,
                };
              },
            }),
          },
          SET_PLAYING_STATE: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  playing: event.playing,
                };
              },
            }),
          },
          LOOP_VIDEO: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  loopLength: event.loopLength,
                  loopStartTime: event.loopStartTime,
                };
              },
            }),
          },
          SET_VIDEO_LENGTH: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                console.log('SET_VIDEO_LENGTH', context, event);
                return {
                  ...context.videoPlayer,
                  videoLength: event.videoLength,
                };
              },
            }),
          },
          SET_VIDEO_FRAME_RATE: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  frameRate: event.frameRate,
                };
              },
            }),
          },
          SET_PLAYBACK_RATE: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context, event) => {
                return {
                  ...context.videoPlayer,
                  playbackRate: event.playbackRate,
                };
              },
            }),
          },
          VIDEO_LOOPED: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context) => {
                return {
                  ...context.videoPlayer,
                  loopCount: context.videoPlayer.loopCount + 1,
                };
              },
            }),
          },
          RESET_LOOP_COUNT: {
            actions: assign<LibraryState, AnyEventObject>({
              videoPlayer: (context) => {
                return {
                  ...context.videoPlayer,
                  loopCount: 0,
                };
              },
            }),
          },
        },
      },
      cursor: {
        on: {
          SET_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context, event) => {
                const cursor = event.idx;
                updatePersistedCursor(context, cursor);
                return cursor;
              },
              scrollToCursorEventId: (context, event) =>
                event.scrollToView ? uniqueId() : context.scrollToCursorEventId,
              videoPlayer: (context) => ({
                ...context.videoPlayer,
                loopCount: 0,
              }),
            }),
          },
          CLEAR_SCROLL_TO_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              scrollToCursorEventId: () => null,
            }),
          },
          RESET_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context, event) => {
                console.log('RESET_CURSOR', context, event);
                const cursor = filter(
                  context.libraryLoadId,
                  context.textFilter,
                  context.library,
                  context.settings.filters,
                  context.settings.sortBy
                ).findIndex(
                  (item: Item) =>
                    item?.path &&
                    event.currentItem?.path &&
                    path.normalize(item?.path).toLowerCase() ===
                      path.normalize(event.currentItem?.path).toLowerCase()
                );
                console.log('index of current item', event.currentItem, cursor);
                const finalCursor = cursor > -1 ? cursor : 0;
                updatePersistedCursor(context, finalCursor);
                return finalCursor;
              },
            }),
          },
          INCREMENT_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>((context, event) => {
              const view = filter(
                context.libraryLoadId,
                context.textFilter,
                context.library,
                context.settings.filters,
                context.settings.sortBy
              );
              const lastIndex = Math.max(0, view.length - 1);
              const nextCursor =
                context.cursor < lastIndex ? context.cursor + 1 : 0;
              updatePersistedCursor(context, nextCursor);
              return {
                cursor: nextCursor,
                pinnedPath: context.streaming
                  ? view[nextCursor]?.path || context.pinnedPath
                  : context.pinnedPath,
                userMovedCursorDuringStreaming: context.streaming
                  ? true
                  : context.userMovedCursorDuringStreaming,
                scrollToCursorEventId: event.scrollToView
                  ? uniqueId()
                  : context.scrollToCursorEventId,
                videoPlayer: {
                  ...context.videoPlayer,
                  loopCount: 0,
                },
              };
            }),
          },
          DECREMENT_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>((context, event) => {
              const view = filter(
                context.libraryLoadId,
                context.textFilter,
                context.library,
                context.settings.filters,
                context.settings.sortBy
              );
              const lastIndex = Math.max(0, view.length - 1);
              const nextCursor =
                context.cursor > 0 ? context.cursor - 1 : lastIndex;
              updatePersistedCursor(context, nextCursor);
              return {
                cursor: nextCursor,
                pinnedPath: context.streaming
                  ? view[nextCursor]?.path || context.pinnedPath
                  : context.pinnedPath,
                userMovedCursorDuringStreaming: context.streaming
                  ? true
                  : context.userMovedCursorDuringStreaming,
                scrollToCursorEventId: event.scrollToView
                  ? uniqueId()
                  : context.scrollToCursorEventId,
                videoPlayer: {
                  ...context.videoPlayer,
                  loopCount: 0,
                },
              };
            }),
          },
        },
      },
      library: {
        initial: 'boot',
        states: {
          boot: {
            always: [
              { target: 'autoSetup', cond: missingDb },
              { target: 'loadingDB' },
            ],
          },
          autoSetup: {
            always: [
              {
                target: 'loadingDB',
                actions: assign<LibraryState, AnyEventObject>({
                  dbPath: () => {
                    const dbPath = appArgs?.dbPath ?? '';
                    // Only persist real file paths, not the web-mode placeholder
                    if (dbPath && dbPath !== 'web') {
                      store.set('dbPath', dbPath);
                    }
                    return dbPath;
                  },
                }),
              },
            ],
          },
          manualSetup: {
            on: {
              SELECT_DB: {
                target: 'selectingDB',
              },
              SELECT_FILE: {
                target: 'selecting',
              },
              SELECT_DIRECTORY: {
                target: 'selectingDirectory',
              },
            },
          },
          selectingDB: {
            invoke: {
              src: (context, event) => {
                console.log('selecting DB', context, event);
                const currentDB = context.dbPath;
                return invoke('select-db', [
                  currentDB,
                ]);
              },
              onDone: {
                target: 'loadingDB',
                actions: ['setDB'],
              },
              onError: {
                target: 'manualSetup',
              },
            },
          },
          loadingDB: {
            invoke: {
              src: (context, event) => {
                console.log('loading DB', context, event);
                return invoke('load-db', [
                  context.dbPath,
                ]);
              },
              onDone: {
                target: 'init',
                actions: [
                  (context) => {
                    // Notify media-server of DB path change so it can switch databases
                    // Skip for the web-mode placeholder — the server already knows its own DB
                    if (context.dbPath === 'web') return;
                    const headers: HeadersInit = {
                      'Content-Type': 'application/json',
                    };
                    if (context.authToken) {
                      headers['Authorization'] = `Bearer ${context.authToken}`;
                    }
                    fetch(`${mediaServerBase}/config`, {
                      method: 'POST',
                      headers,
                      body: JSON.stringify({ dbPath: context.dbPath }),
                    }).catch(() => {
                      // Server may not be running — ignore
                    });
                  },
                ],
              },
              onError: {
                target: 'manualSetup',
              },
            },
          },
          init: {
            always: [
              { target: 'loadingFromFS', cond: hasInitialFile },
              { target: 'loadingFromPersisted', cond: hasPersistedLibrary },
              { target: 'restoringWebSession', cond: () => !isElectron },
              { target: 'selecting' },
            ],
            entry: assign<LibraryState, AnyEventObject>({
              initSessionId: () => uniqueId(),
            }),
          },
          // Web mode: restore persisted query/cursor state from session store before loading
          restoringWebSession: {
            entry: assign<LibraryState, AnyEventObject>({
              dbQuery: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.dbQuery : { tags: [] };
              },
              query: () => {
                const queryData = getSessionValue('query');
                return queryData?.query ?? { predicates: [] };
              },
              mostRecentTag: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.mostRecentTag : '';
              },
              mostRecentCategory: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.mostRecentCategory : '';
              },
              textFilter: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.textFilter : '';
              },
              cursor: () => {
                const cursorData = getSessionValue('cursor');
                return cursorData ? cursorData.cursor : 0;
              },
              // Restore back-navigation slot from session so a reload in
              // web mode keeps the same one-step undo as Electron mode.
              previousLibrary: () => {
                const previousData = getSessionValue('previous');
                return previousData ? previousData.previousLibrary : [];
              },
              previousCursor: () => {
                const previousData = getSessionValue('previous');
                return previousData ? previousData.previousCursor : 0;
              },
              previousStateType: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousStateType ?? null;
              },
              previousTextFilter: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousTextFilter ?? '';
              },
              previousDbQuery: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousDbQuery ?? { tags: [] };
              },
              previousQuery: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousQuery ?? { predicates: [] };
              },
              previousInitialFile: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousInitialFile ?? '';
              },
            }),
            always: [
              { target: 'loadingFromDB', cond: hasPersistedFilter },
              { target: 'selectingDirectory' },
            ],
          },
          selecting: {
            invoke: {
              src: (context, event) => {
                const currentFile = context.initialFile;
                console.log('selecting', context, event);
                return invoke('select-file', [
                  currentFile,
                ]);
              },
              onDone: {
                target: 'loadingFromFS',
                actions: ['setPath'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          selectingDirectory: {
            invoke: {
              src: (context, event) => {
                const currentFile = context.initialFile;
                console.log('selecting directory', context, event);
                return invoke('select-directory', [
                  currentFile,
                ]);
              },
              onDone: {
                target: 'loadingFromFS',
                actions: ['setPath'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          loadingFromFS: {
            entry: assign<LibraryState, AnyEventObject>({
              library: (context) => [{ path: context.initialFile, mtimeMs: 0 }],
              libraryLoadId: () => uniqueId(),
              cursor: 0,
              dbQuery: () => ({ tags: [] }),
              query: () => ({ predicates: [] }),
              streaming: () => true,
              pinnedPath: (context) => context.initialFile,
              savedSortByDuringStreaming: (context) => context.settings.sortBy,
              userMovedCursorDuringStreaming: () => false,
              settings: (context) => ({
                ...context.settings,
                sortBy: 'stream',
              }),
            }),
            invoke: {
              src: (context, event) => {
                console.log('loadingFromFS', context, event);
                const { recursive } = context.settings;
                return invoke('load-files', [
                  context.initialFile,
                  // Ask main to sort with original sort once complete
                  context.savedSortByDuringStreaming || 'name',
                  recursive,
                  { fastest: true },
                ]);
              },
              onDone: {
                target: 'loadedFromFS',
                actions: [
                  // Single atomic assign to prevent intermediate state observations
                  assign<LibraryState, AnyEventObject>((context, event) => {
                    const lib = (event.data?.library || []) as Item[];
                    const newLibraryLoadId = uniqueId();
                    const restoredSortBy = (context.savedSortByDuringStreaming ||
                      'name') as Settings['sortBy'];

                    // Calculate final cursor position using pinned path and restored sort
                    let finalCursor = event.data?.cursor ?? context.cursor;
                    if (Array.isArray(lib) && lib.length > 0) {
                      const tempLibraryLoadId = 'final-' + uniqueId();
                      const sorted = filter(
                        tempLibraryLoadId,
                        context.textFilter,
                        lib,
                        context.settings.filters,
                        restoredSortBy
                      );
                      const preferredPaths = [
                        context.pinnedPath || undefined,
                        context.library[context.cursor]?.path || undefined,
                      ].filter(Boolean) as string[];

                      for (const p of preferredPaths) {
                        const idx = sorted.findIndex(
                          (it: Item) =>
                            it?.path &&
                            path.normalize(it.path).toLowerCase() ===
                              path.normalize(p).toLowerCase()
                        );
                        if (idx !== -1) {
                          finalCursor = idx;
                          break;
                        }
                      }
                      updatePersistedCursor(context, finalCursor);
                    }

                    // Persist library updates
                    setSessionValues({
                      library: { library: lib, initialFile: context.initialFile },
                      cursor: { cursor: finalCursor },
                    });

                    // Return ALL state changes atomically
                    return {
                      library: lib,
                      libraryLoadId: newLibraryLoadId,
                      cursor: finalCursor,
                      settings: {
                        ...context.settings,
                        sortBy: restoredSortBy,
                      },
                      streaming: false,
                      pinnedPath: null,
                      userMovedCursorDuringStreaming: false,
                      savedSortByDuringStreaming: null,
                    };
                  }),
                ],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
            on: {
              LOAD_FILES_BATCH: {
                actions: assign<LibraryState, AnyEventObject>(
                  (context, event) => {
                    const currentView = filter(
                      context.libraryLoadId,
                      context.textFilter,
                      context.library,
                      context.settings.filters,
                      (context.streaming
                        ? 'stream'
                        : context.settings.sortBy) as any
                    );
                    const previousSelectedPath =
                      context.pinnedPath || currentView[context.cursor]?.path;
                    // Use case-insensitive path matching to avoid duplicates on Windows
                    const existing = new Set(
                      context.library.map((item) => item.path.toLowerCase())
                    );
                    const incoming = (event.data?.files || []) as Item[];
                    const newItems = incoming.filter(
                      (f) => f?.path && !existing.has(f.path.toLowerCase())
                    );
                    const updatedLibrary = context.library.concat(newItems);

                    let nextCursor = context.cursor;
                    if (previousSelectedPath) {
                      const tempLibraryLoadId = 'batch-' + uniqueId();
                      const sorted = filter(
                        tempLibraryLoadId,
                        context.textFilter,
                        updatedLibrary,
                        context.settings.filters,
                        (context.streaming
                          ? 'stream'
                          : context.settings.sortBy) as any
                      );
                      const newIndex = sorted.findIndex(
                        (item: Item) =>
                          item?.path &&
                          path.normalize(item.path).toLowerCase() ===
                            path.normalize(previousSelectedPath).toLowerCase()
                      );
                      if (newIndex !== -1) {
                        updatePersistedCursor(context, newIndex);
                        nextCursor = newIndex;
                      }
                    }

                    return {
                      library: updatedLibrary,
                      cursor: nextCursor,
                      libraryLoadId: uniqueId(),
                      streaming: true,
                    };
                  }
                ),
              },
              SET_ACTIVE_CATEGORY: {
                actions: assign<LibraryState, AnyEventObject>({
                  activeCategory: (context, event) => {
                    console.log('SET_ACTIVE_CATEGORY', context, event);
                    store.set(
                      'activeCategory',
                      event.data.category
                    );
                    return event.data.category;
                  },
                }),
              },
              SET_CURSOR: {
                actions: assign<LibraryState, AnyEventObject>({
                  userMovedCursorDuringStreaming: (context) =>
                    context.streaming
                      ? true
                      : context.userMovedCursorDuringStreaming,
                  pinnedPath: (context, event) =>
                    context.streaming
                      ? filter(
                          context.libraryLoadId,
                          context.textFilter,
                          context.library,
                          context.settings.filters,
                          context.settings.sortBy
                        )[event.idx]?.path || context.pinnedPath
                      : context.pinnedPath,
                }),
              },
              LOAD_FILES_DONE: {
                // Do not clear pinnedPath or flip streaming here.
                // Let the invoke onDone handler finalize state so it can
                // compute the correct cursor using the pinned path.
                actions: assign<LibraryState, AnyEventObject>({}),
              },
            },
          },
          loadingFromDB: {
            invoke: {
              src: (context) =>
                platformLoadMediaByQuery(
                  context.query.predicates,
                  context.settings.filteringMode,
                  context.authToken
                ),
              onDone: {
                target: 'loadedFromDB',
                // applySimilaritySort here too (not just on runningQuery) so a
                // loaded/restored visual query also auto-sorts by similarity.
                actions: ['setLibraryWithPrevious', 'applySimilaritySort'],
              },
              onError: {
                target: 'loadedFromFS',
                actions: ['addQueryErrorToast'],
              },
            },
            on: { ...queryMutationOn },
          },
          // Unified query loader. Runs the structured `query` predicate list
          // through a single platform service. New query events (ADD_PREDICATE,
          // REMOVE_PREDICATE, TOGGLE_EXCLUDE, SET_QUERY) target this state.
          runningQuery: {
            invoke: {
              src: (context) =>
                platformLoadMediaByQuery(
                  context.query.predicates,
                  context.settings.filteringMode,
                  context.authToken
                ),
              onDone: {
                target: 'loadedFromDB',
                actions: ['setLibrary', 'applySimilaritySort'],
              },
              onError: {
                target: 'loadedFromFS',
                actions: ['addQueryErrorToast'],
              },
            },
            on: { ...queryMutationOn },
          },
          loadingFromPersisted: {
            entry: assign<LibraryState, AnyEventObject>({
              library: (context) => {
                // Use session store cache (sync read from memory)
                const persistedData = getSessionValue('library');
                return persistedData ? persistedData.library : [];
              },
              initialFile: (context) => {
                const persistedData = getSessionValue('library');
                return persistedData
                  ? persistedData.initialFile
                  : context.initialFile;
              },
              cursor: () => {
                const cursorData = getSessionValue('cursor');
                return cursorData ? cursorData.cursor : 0;
              },
              scrollPosition: () => {
                const cursorData = getSessionValue('cursor');
                return cursorData?.scrollPosition ?? 0;
              },
              previousLibrary: () => {
                const previousData = getSessionValue('previous');
                return previousData ? previousData.previousLibrary : [];
              },
              previousCursor: () => {
                const previousData = getSessionValue('previous');
                return previousData ? previousData.previousCursor : 0;
              },
              // Restore previous state type info for proper back navigation
              previousStateType: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousStateType ?? null;
              },
              previousTextFilter: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousTextFilter ?? '';
              },
              previousDbQuery: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousDbQuery ?? { tags: [] };
              },
              previousQuery: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousQuery ?? { predicates: [] };
              },
              previousInitialFile: () => {
                const previousData = getSessionValue('previous');
                return previousData?.previousInitialFile ?? '';
              },
              dbQuery: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.dbQuery : { tags: [] };
              },
              query: () => {
                const queryData = getSessionValue('query');
                return queryData?.query ?? { predicates: [] };
              },
              mostRecentTag: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.mostRecentTag : '';
              },
              mostRecentCategory: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.mostRecentCategory : '';
              },
              textFilter: () => {
                const queryData = getSessionValue('query');
                return queryData ? queryData.textFilter : '';
              },
              libraryLoadId: () => uniqueId(),
            }),
            always: [
              { target: 'loadedFromDB', cond: hasPersistedFilter },
              { target: 'loadedFromFS' },
            ],
          },
          loadingFromPreviousLibrary: {
            // Restoration runs in three discrete actions so the assign is
            // pure and the side effects are visible: (1) atomic context
            // restore from previous*, (2) mirror the restored snapshot to
            // the session store, (3) `always` then routes to the correct
            // loaded* state based on the restored currentStateType.
            entry: [
              assign<LibraryState, AnyEventObject>({
                library: (context) => context.previousLibrary,
                cursor: (context) => context.previousCursor,
                textFilter: (context) => context.previousTextFilter,
                dbQuery: (context) => ({ ...context.previousDbQuery }),
                query: (context) => ({
                  predicates: [...context.previousQuery.predicates],
                }),
                initialFile: (context) =>
                  context.previousInitialFile || context.initialFile,
                currentStateType: (context) =>
                  context.previousStateType || ('fs' as LibraryStateType),
                libraryLoadId: () => uniqueId(),
                // Clear the previous-state slot now that we've consumed it.
                previousLibrary: () => [],
                previousCursor: () => 0,
                previousStateType: () => null,
                previousTextFilter: () => '',
                previousDbQuery: () => ({ tags: [] }),
                previousQuery: () => ({ predicates: [] }),
                previousInitialFile: () => '',
              }),
              // Mirror the restored snapshot to the session store. context
              // here is post-assign, so library/cursor/textFilter/dbQuery
              // already reflect the restored values.
              (context) => {
                setSessionValues({
                  library: {
                    library: context.library,
                    initialFile: context.initialFile,
                  },
                  cursor: { cursor: context.cursor },
                  previous: {
                    previousLibrary: [],
                    previousCursor: 0,
                    previousStateType: null,
                    previousTextFilter: '',
                    previousDbQuery: { tags: [] },
                    previousQuery: { predicates: [] },
                    previousInitialFile: '',
                  },
                });
                updatePersistedState(context);
              },
            ],
            always: [
              {
                target: 'loadedFromDB',
                cond: (context) => context.currentStateType === 'db',
              },
              {
                // If previous library is empty and we're going back to FS mode,
                // reload from disk instead of showing empty library
                // (but only if we have an initialFile to reload from)
                target: 'loadingFromFS',
                cond: (context) =>
                  context.currentStateType === 'fs' &&
                  context.library.length === 0 &&
                  context.initialFile.length > 0,
              },
              {
                // If no initialFile and empty library, prompt user to select
                target: 'selecting',
                cond: (context) =>
                  context.currentStateType === 'fs' &&
                  context.library.length === 0 &&
                  context.initialFile.length === 0,
              },
              { target: 'loadedFromFS' },
            ],
          },
          loadedFromFS: {
            initial: 'idle',
            // Invariant: in FS mode, no description search and no tag query
            // are active. If we land here with stale filter/query state from a
            // partial restore or some upstream bug, force the context back to
            // a coherent shape so the UI never shows a search/tag pill paired
            // with a filesystem library.
            entry: [
              assign<LibraryState, AnyEventObject>({
                currentStateType: () => 'fs' as LibraryStateType,
                textFilter: () => '',
                dbQuery: () => ({ tags: [] }),
                query: () => ({ predicates: [] }),
              }),
              // Mirror the cleared invariant to session storage so a folder
              // load can never leave session.query holding stale tags. Without
              // this, a later boot would see persisted tags + an FS library
              // and route to loadedFromDB over the wrong library.
              (context) => updatePersistedState(context),
            ],
            on: {
              ...queryMutationOn,
              SELECT_FILE: {
                target: 'selecting',
              },
              SELECT_DIRECTORY: {
                target: 'selectingDirectory',
              },
              SET_ACTIVE_CATEGORY: {
                actions: assign<LibraryState, AnyEventObject>({
                  activeCategory: (context, event) => {
                    console.log('SET_ACTIVE_CATEGORY', context, event);

                    store.set(
                      'activeCategory',
                      event.data.category
                    );
                    return event.data.category;
                  },
                }),
              },
              SET_FILE: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  textFilter: () => '',
                  initialFile: (context, event) => event.path,
                }),
              },
              DELETE_FILE: {
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    library: (context, event) => {
                      console.log('DELETE_FILE', context, event);
                      try {
                        invoke('delete-file', [
                          event.data.path,
                        ]);
                      } catch (e) {
                        console.error(e);
                      }
                      const path = event.data.path;
                      const library = [...context.library];
                      const index = library.findIndex(
                        (item) => item.path === path
                      );
                      if (index > -1) {
                        library.splice(index, 1);
                      }
                      return library;
                    },
                    cursor: (context) => {
                      // If the cursor was on the last item in the library, decrement it.
                      if (context.cursor >= context.library.length - 1) {
                        return context.cursor - 1;
                      }
                      return context.cursor;
                    },
                    libraryLoadId: () => uniqueId(),
                  }),
                  assign<LibraryState, AnyEventObject>({
                    toasts: (context, event) => {
                      const filename =
                        event.data.path.split(/[\\/]/).pop() || event.data.path;
                      const newToast = {
                        id: uniqueId(),
                        type: 'info' as const,
                        title: 'File deleted',
                        message: filename,
                        timestamp: Date.now(),
                      };
                      return [...context.toasts, newToast];
                    },
                  }),
                ],
              },
              CHANGE_SETTING_AND_RELOAD: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  settings: (context, event) => {
                    console.log('CHANGE_SETTING_AND_RELOAD', context, event);
                    const processedData = { ...event.data };

                    for (const key in processedData) {
                      // Clamp volume to valid range [0, 1]
                      if (
                        key === 'volume' &&
                        typeof processedData[key] === 'number'
                      ) {
                        processedData[key] = clampVolume(processedData[key]);
                      }

                      store.set(key, processedData[key]);
                    }
                    return {
                      ...context.settings,
                      ...processedData,
                    };
                  },
                }),
              },
              CHANGE_DB_PATH: {
                target: 'selectingDB',
              },
              SET_QUERY_TAG: [
                {
                  target: 'loadingFromDB',
                  actions: [
                    // Atomic snapshot of current state for back-navigation,
                    // then mutate dbQuery in a separate assign so the capture
                    // and the mutation can't race.
                    capturePrevious,
                    persistPreviousState,
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'SET QUERY TAG TO',
                          context,
                          event.data.tag
                        );
                        return { tags: [event.data.tag] };
                      },
                      // Mirror the tag click into the unified query model.
                      query: (context, event) =>
                        applyTagClick(
                          context.query,
                          event.data.tag,
                          context.settings.filteringMode
                        ),
                    }),
                    (context, event) => {
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: [event.data.tag] },
                        query: applyTagClick(
                          context.query,
                          event.data.tag,
                          context.settings.filteringMode
                        ),
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      clearSessionKeys(['library', 'cursor']);
                    },
                  ],
                },
              ],
              SHUFFLE: {
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: 0,
                  libraryLoadId: () => uniqueId(),
                  settings: (context, event) => {
                    console.log('SHUFFLE', context, event);
                    return {
                      ...context.settings,
                      sortBy: 'shuffle',
                    };
                  },
                }),
              },
              SORTED_SCORE: {
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: 0,
                  libraryLoadId: () => uniqueId(),
                  settings: (context) => ({
                    ...context.settings,
                    sortBy: 'similarity',
                  }),
                }),
              },
            },
            states: {
              idle: {
                on: {
                  LOAD_FILES_BATCH: {
                    actions: assign<LibraryState, AnyEventObject>({
                      library: (context, event) => {
                        // Use the sorted/filtered view to look up the current item,
                        // since cursor is an index into the sorted view, not the raw library
                        const currentView = filter(
                          context.libraryLoadId,
                          context.textFilter,
                          context.library,
                          context.settings.filters,
                          context.settings.sortBy
                        );
                        const previousSelectedPath =
                          context.pinnedPath ||
                          currentView[context.cursor]?.path;
                        // Use case-insensitive path matching to avoid duplicates on Windows
                        const existing = new Set(
                          context.library.map((item) => item.path.toLowerCase())
                        );
                        const incoming = (event.data?.files || []) as Item[];
                        const newItems = incoming.filter(
                          (f) => f?.path && !existing.has(f.path.toLowerCase())
                        );
                        const updatedLibrary = context.library.concat(newItems);

                        if (previousSelectedPath) {
                          const tempLibraryLoadId = 'batch-' + uniqueId();
                          const sorted = filter(
                            tempLibraryLoadId,
                            context.textFilter,
                            updatedLibrary,
                            context.settings.filters,
                            (context.streaming
                              ? 'stream'
                              : context.settings.sortBy) as any
                          );
                          const newIndex = sorted.findIndex(
                            (item: Item) =>
                              item?.path &&
                              path.normalize(item.path).toLowerCase() ===
                                path.normalize(previousSelectedPath).toLowerCase()
                          );
                          if (newIndex !== -1) {
                            updatePersistedCursor(context, newIndex);
                            (event as any).__newCursor = newIndex;
                          }
                        }

                        return updatedLibrary;
                      },
                      cursor: (context, event) => {
                        const computed = (event as any).__newCursor as
                          | number
                          | undefined;
                        return typeof computed === 'number'
                          ? computed
                          : context.cursor;
                      },
                      libraryLoadId: () => uniqueId(),
                      streaming: () => true,
                    }),
                  },
                  SET_SCROLL_POSITION: {
                    actions: assign<LibraryState, AnyEventObject>({
                      scrollPosition: (context, event) => {
                        return event.position;
                      },
                    }),
                  },
                  REFRESH_LIBRARY: {
                    target: 'refreshing',
                  },
                },
              },
              refreshing: {
                invoke: {
                  src: (context) => {
                    console.log('[refresh] starting library refresh');
                    return invoke('refresh-library', [
                      {
                        initialFile: context.initialFile,
                        currentPaths: context.library.map((item) => item.path),
                        recursive: context.settings.recursive,
                      },
                    ]);
                  },
                  onDone: {
                    target: 'idle',
                    actions: [
                      assign<LibraryState, AnyEventObject>((context, event) => {
                        const { added, removed } = event.data as {
                          added: Item[];
                          removed: string[];
                        };

                        console.log('[refresh] complete:', {
                          added: added.length,
                          removed: removed.length,
                        });

                        if (added.length === 0 && removed.length === 0) {
                          // No changes, just return toast
                          const newToast = {
                            id: uniqueId(),
                            type: 'info' as const,
                            title: 'Library up to date',
                            message: 'No new or deleted files found',
                            timestamp: Date.now(),
                          };
                          return {
                            toasts: [...context.toasts, newToast],
                          };
                        }

                        // Build set of removed paths for filtering
                        const removedSet = new Set(
                          removed.map((p) => p.toLowerCase())
                        );

                        // Filter out removed files
                        let updatedLibrary = context.library.filter(
                          (item) => !removedSet.has(item.path.toLowerCase())
                        );

                        // Add new files
                        const existingPaths = new Set(
                          updatedLibrary.map((item) => item.path.toLowerCase())
                        );
                        const newItems = added.filter(
                          (item) => !existingPaths.has(item.path.toLowerCase())
                        );
                        updatedLibrary = updatedLibrary.concat(newItems);

                        // Adjust cursor if needed
                        const currentItem = context.library[context.cursor];
                        let newCursor = context.cursor;

                        if (currentItem) {
                          // Try to find the same item in the updated library
                          const newIndex = updatedLibrary.findIndex(
                            (item) =>
                              item.path.toLowerCase() ===
                              currentItem.path.toLowerCase()
                          );
                          if (newIndex !== -1) {
                            newCursor = newIndex;
                          } else if (newCursor >= updatedLibrary.length) {
                            // Current item was removed and cursor is past end
                            newCursor = Math.max(0, updatedLibrary.length - 1);
                          }
                        }

                        // Create toast message
                        const parts = [];
                        if (added.length > 0)
                          parts.push(`${added.length} added`);
                        if (removed.length > 0)
                          parts.push(`${removed.length} removed`);
                        const newToast = {
                          id: uniqueId(),
                          type: 'success' as const,
                          title: 'Library refreshed',
                          message: parts.join(', '),
                          timestamp: Date.now(),
                        };

                        // Persist the updated state
                        setSessionValues({
                          library: {
                            library: updatedLibrary,
                            initialFile: context.initialFile,
                          },
                          cursor: { cursor: newCursor },
                        });

                        return {
                          library: updatedLibrary,
                          libraryLoadId: uniqueId(),
                          cursor: newCursor,
                          toasts: [...context.toasts, newToast],
                        };
                      }),
                    ],
                  },
                  onError: {
                    target: 'idle',
                    actions: assign<LibraryState, AnyEventObject>((context) => {
                      console.error('[refresh] error during library refresh');
                      const newToast = {
                        id: uniqueId(),
                        type: 'error' as const,
                        title: 'Refresh failed',
                        message: 'Could not refresh library',
                        timestamp: Date.now(),
                      };
                      return {
                        toasts: [...context.toasts, newToast],
                      };
                    }),
                  },
                },
              },
            },
          },
          loadedFromDB: {
            initial: 'idle',
            // Invariant: in tag (DB) mode, description search is empty (tags
            // and search are mutually exclusive). Force textFilter='' on entry
            // so a stale search pill can't surface alongside tag results.
            entry: [
              (context, event) => console.log('loadedFromDB', context, event),
              assign<LibraryState, AnyEventObject>({
                currentStateType: () => 'db' as LibraryStateType,
                textFilter: () => '',
              }),
              (context) => updatePersistedState(context),
            ],
            on: {
              ...queryMutationOn,
              LOAD_FILES_BATCH: {
                actions: assign<LibraryState, AnyEventObject>({
                  library: (context, event) => {
                    // Use the sorted/filtered view to look up the current item,
                    // since cursor is an index into the sorted view, not the raw library
                    const currentView = filter(
                      context.libraryLoadId,
                      context.textFilter,
                      context.library,
                      context.settings.filters,
                      context.settings.sortBy
                    );
                    const previousSelectedPath =
                      context.pinnedPath ||
                      currentView[context.cursor]?.path;
                    // Use case-insensitive path matching to avoid duplicates on Windows
                    const existing = new Set(
                      context.library.map((item) => item.path.toLowerCase())
                    );
                    const incoming = (event.data?.files || []) as Item[];
                    const newItems = incoming.filter(
                      (f) => f?.path && !existing.has(f.path.toLowerCase())
                    );
                    const updatedLibrary = context.library.concat(newItems);
                    if (previousSelectedPath) {
                      const tempLibraryLoadId = 'batch-' + uniqueId();
                      const sorted = filter(
                        tempLibraryLoadId,
                        context.textFilter,
                        updatedLibrary,
                        context.settings.filters,
                        (context.streaming
                          ? 'stream'
                          : context.settings.sortBy) as any
                      );
                      const newIndex = sorted.findIndex(
                        (item: Item) =>
                          item?.path &&
                          path.normalize(item.path).toLowerCase() ===
                            path.normalize(previousSelectedPath).toLowerCase()
                      );
                      if (newIndex !== -1) {
                        updatePersistedCursor(context, newIndex);
                        (event as any).__newCursor = newIndex;
                      }
                    }
                    return updatedLibrary;
                  },
                  cursor: (context, event) => {
                    const computed = (event as any).__newCursor as
                      | number
                      | undefined;
                    return typeof computed === 'number'
                      ? computed
                      : context.cursor;
                  },
                  libraryLoadId: () => uniqueId(),
                  streaming: () => true,
                }),
              },
              SELECT_FILE: {
                target: 'selecting',
              },
              SELECT_DIRECTORY: {
                target: 'selectingDirectory',
              },
              SET_FILE: {
                target: 'loadingFromFS',
                actions: [
                  capturePrevious,
                  persistPreviousState,
                  assign<LibraryState, AnyEventObject>({
                    textFilter: () => '',
                    initialFile: (context, event) => event.path,
                  }),
                ],
              },
              SET_ACTIVE_CATEGORY: {
                actions: assign<LibraryState, AnyEventObject>({
                  activeCategory: (context, event) => {
                    console.log('SET_ACTIVE_CATEGORY', context, event);
                    store.set('activeCategory', event.data.category);
                    return event.data.category;
                  },
                }),
              },
              CHANGE_DB_PATH: {
                target: 'selectingDB',
              },
              UPDATE_FILE_PATH: {
                target: 'selectingFilePath',
              },
              CLEAR_QUERY_TAG: {
                // No actions — loadingFromPreviousLibrary's entry restores
                // dbQuery, textFilter, library, and cursor from previous*
                // and persists them. Writing dbQuery: { tags: [] } here
                // would be clobbered by the entry, and persisting that
                // intermediate state to disk could leave the on-disk query
                // in a bad shape if the app closes mid-transition.
                target: 'loadingFromPreviousLibrary',
              },
              // Remove a single tag from the active query. Falls back to
              // CLEAR semantics (restoring the previous library) when the
              // last tag is removed; otherwise reloads against the
              // remaining tag set.
              REMOVE_QUERY_TAG: [
                {
                  cond: (context, event) => {
                    const remaining = (context.dbQuery.tags || []).filter(
                      (t) => t !== event.data.tag
                    );
                    return remaining.length > 0;
                  },
                  // Target runningQuery (which uses setLibrary, not
                  // setLibraryWithPrevious) so the original entry-mode
                  // previous (e.g. FS → DB) is preserved across within-DB
                  // tag tweaks. Symmetric with SET_QUERY_TAG.
                  target: 'runningQuery',
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => ({
                        tags: (context.dbQuery.tags || []).filter(
                          (t) => t !== event.data.tag
                        ),
                      }),
                      query: (context, event) =>
                        removePredicate(
                          context.query,
                          predicateKey({
                            type: 'tag',
                            value: event.data.tag,
                            exclude: false,
                          })
                        ),
                    }),
                    (context, event) => {
                      const newTags = (context.dbQuery.tags || []).filter(
                        (t) => t !== event.data.tag
                      );
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: newTags },
                        query: removePredicate(
                          context.query,
                          predicateKey({
                            type: 'tag',
                            value: event.data.tag,
                            exclude: false,
                          })
                        ),
                      });
                      clearSessionKeys(['library', 'cursor']);
                    },
                  ],
                },
                {
                  target: 'loadingFromPreviousLibrary',
                },
              ],
              SHUFFLE: {
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: 0,
                  libraryLoadId: () => uniqueId(),
                  settings: (context, event) => {
                    console.log('SHUFFLE', context, event);
                    return {
                      ...context.settings,
                      sortBy: 'shuffle',
                    };
                  },
                }),
              },
              SORTED_SCORE: {
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: 0,
                  libraryLoadId: () => uniqueId(),
                  settings: (context) => ({
                    ...context.settings,
                    sortBy: 'similarity',
                  }),
                }),
              },
              UPDATE_MEDIA_ELO: {
                actions: assign<LibraryState, AnyEventObject>({
                  library: (context, event) => {
                    console.log('UPDATE_MEDIA_ELO', context, event);
                    const { path, elo } = event;
                    const library = [...context.library];
                    const item = library.find((item) => item.path === path);
                    if (item) {
                      item.elo = elo;
                    }
                    return library;
                  },
                  libraryLoadId: () => uniqueId(),
                }),
              },
              DELETE_FILE: {
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    library: (context, event) => {
                      console.log('DELETE_FILE', context, event);
                      try {
                        invoke('delete-file', [
                          event.data.path,
                        ]);
                      } catch (e) {
                        console.error(e);
                      }
                      const path = event.data.path;
                      const library = [...context.library];
                      const index = library.findIndex(
                        (item) => item.path === path
                      );
                      if (index > -1) {
                        library.splice(index, 1);
                      }
                      return library;
                    },
                    cursor: (context) => {
                      // If the cursor was on the last item in the library, decrement it.
                      if (context.cursor >= context.library.length - 1) {
                        return context.cursor - 1;
                      }
                      return context.cursor;
                    },
                    libraryLoadId: () => uniqueId(),
                  }),
                  assign<LibraryState, AnyEventObject>({
                    toasts: (context, event) => {
                      const filename =
                        event.data.path.split(/[\\/]/).pop() || event.data.path;
                      const newToast = {
                        id: uniqueId(),
                        type: 'info' as const,
                        title: 'File deleted',
                        message: filename,
                        timestamp: Date.now(),
                      };
                      return [...context.toasts, newToast];
                    },
                  }),
                ],
              },
              SET_QUERY_TAG: [
                {
                  target: 'runningQuery',
                  cond: willHaveTag,
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'SET QUERY TAG TO',
                          context,
                          event.data.tags
                        );
                        // Get active tags and index by tag.
                        const activeTags = context.dbQuery.tags.reduce(
                          (acc, tag) => {
                            acc[tag] = true;
                            return acc;
                          },
                          {} as { [key: string]: boolean }
                        );
                        // eslint-disable-next-line no-constant-condition
                        if (true) {
                          // DELETE ALL KEYS THAT DONT MATCH THE EVENT TAG
                          Object.keys(activeTags).forEach((tag) => {
                            if (
                              tag !== event.data.tag &&
                              context.settings.filteringMode === 'EXCLUSIVE'
                            ) {
                              delete activeTags[tag];
                            }
                          });
                        }
                        // If the tag is already active, remove it.
                        if (activeTags[event.data.tag]) {
                          delete activeTags[event.data.tag];
                        } else {
                          // Otherwise, add it.
                          activeTags[event.data.tag] = true;
                        }
                        return { tags: Object.keys(activeTags) };
                      },
                      // Mirror the tag click into the unified query model.
                      query: (context, event) =>
                        applyTagClick(
                          context.query,
                          event.data.tag,
                          context.settings.filteringMode
                        ),
                    }),
                    (context, event) => {
                      // Calculate the new tags for persistence
                      const activeTags = context.dbQuery.tags.reduce(
                        (acc, tag) => {
                          acc[tag] = true;
                          return acc;
                        },
                        {} as { [key: string]: boolean }
                      );
                      if (context.settings.filteringMode === 'EXCLUSIVE') {
                        Object.keys(activeTags).forEach((tag) => {
                          if (tag !== event.data.tag) {
                            delete activeTags[tag];
                          }
                        });
                      }
                      if (activeTags[event.data.tag]) {
                        delete activeTags[event.data.tag];
                      } else {
                        activeTags[event.data.tag] = true;
                      }
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: Object.keys(activeTags) },
                        query: applyTagClick(
                          context.query,
                          event.data.tag,
                          context.settings.filteringMode
                        ),
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      // Invalidate persisted library snapshot using session store
                      clearSessionKeys(['library', 'cursor']);
                    },
                  ],
                },
                {
                  // Toggling off the last tag — let
                  // loadingFromPreviousLibrary restore previous state.
                  target: 'loadingFromPreviousLibrary',
                  cond: willHaveNoTag,
                },
              ],
              DELETED_ASSIGNMENT: {
                // Removing a tag from an image refreshes the current view; it
                // doesn't navigate. Capture the current libraryLoadId so the
                // list can recognize the upcoming setLibrary as an in-place
                // refresh and preserve scroll instead of jumping to top.
                target: 'runningQuery',
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    preserveScrollFromLoadId: (context) => context.libraryLoadId,
                  }),
                  () => console.log('deleted assignment'),
                ],
              },
              SORTED_WEIGHTS: {
                target: 'runningQuery',
                actions: () => console.log('sorting weights'),
              },
            },
            states: {
              idle: {
                on: {
                  SET_SCROLL_POSITION: {
                    actions: assign<LibraryState, AnyEventObject>({
                      scrollPosition: (context, event) => {
                        return event.position;
                      },
                    }),
                  },
                  SET_VIDEO_TIME: {
                    target: 'idle',
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          timeStamp: event.timeStamp,
                          eventId: event.eventId,
                        };
                      },
                    }),
                  },
                  SET_ACTUAL_VIDEO_TIME: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          actualVideoTime: event.timeStamp,
                        };
                      },
                    }),
                  },
                  SET_PLAYING_STATE: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          playing: event.playing,
                        };
                      },
                    }),
                  },
                  LOOP_VIDEO: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          loopLength: event.loopLength,
                          loopStartTime: event.loopStartTime,
                        };
                      },
                    }),
                  },
                  SET_VIDEO_LENGTH: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        console.log('SET_VIDEO_LENGTH', context, event);
                        return {
                          ...context.videoPlayer,
                          videoLength: event.videoLength,
                        };
                      },
                    }),
                  },
                  SET_VIDEO_FRAME_RATE: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          frameRate: event.frameRate,
                        };
                      },
                    }),
                  },
                  SET_PLAYBACK_RATE: {
                    actions: assign<LibraryState, AnyEventObject>({
                      videoPlayer: (context, event) => {
                        return {
                          ...context.videoPlayer,
                          playbackRate: event.playbackRate,
                        };
                      },
                    }),
                  },
                },
              },
            },
          },
          selectingFilePath: {
            invoke: {
              src: (context, event) => {
                console.log('selectingFilePath', context, event);
                return invoke('select-new-path', [
                  event.path,
                  event.updateAll,
                ]);
              },
              onDone: {
                target: 'loadedFromDB',
                actions: ['updateFilePath'],
              },
              onError: {
                target: 'loadedFromDB',
              },
            },
          },
          loadingError: {
            entry: (context, event) =>
              console.log('error loading library', context, event),
          },
        },
      },
    },
  },
  {
    actions: {
      setLibrary,
      applySimilaritySort,
      addQueryErrorToast,
      setLibraryWithPrevious,
      setPath,
      setDB,
      // createJob removed - jobs now handled by external job runner service
      updateFilePath,
    },
  }
);

export const GlobalStateContext = createContext({
  libraryService: {} as InterpreterFrom<typeof libraryMachine>,
});

// Track session store initialization
let sessionStoreReady = false;
let sessionStoreInitPromise: Promise<void> | null = null;

/**
 * Initialize session store before state machine starts.
 * This should be called early in app startup (e.g., in App.tsx).
 */
export async function initializeSessionStore(): Promise<void> {
  if (sessionStoreReady) return;
  if (sessionStoreInitPromise) return sessionStoreInitPromise;

  sessionStoreInitPromise = initSessionStore().then(() => {
    sessionStoreReady = true;
  });

  return sessionStoreInitPromise;
}

// Inner component that only renders after session store is ready
const GlobalStateProviderInner = (props: Props) => {
  const libraryService = useInterpret(libraryMachine);

  React.useEffect(() => {
    const offBatch = on(
      'load-files-batch',
      (...args: unknown[]) => {
        const batch = (args[0] as { path: string; mtimeMs: number }[]) || [];
        libraryService.send({
          type: 'LOAD_FILES_BATCH',
          data: { files: batch },
        });
      }
    );
    const offDone = on('load-files-done', () => {
      libraryService.send({ type: 'LOAD_FILES_DONE' });
    });
    return () => {
      if (typeof offBatch === 'function') offBatch();
      if (typeof offDone === 'function') offDone();
    };
  }, [libraryService]);

  // Persist scroll position only when the app is about to close
  React.useEffect(() => {
    const handleBeforeUnload = () => {
      const { scrollPosition, cursor } = libraryService.getSnapshot().context;
      // Update session store and flush to ensure data is written
      setSessionValue('cursor', { cursor, scrollPosition });
      // Note: flushSession is async but beforeunload doesn't wait.
      // The session store on main process will handle this via 'before-quit' event.
      flushSession();
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [libraryService]);

  return (
    <GlobalStateContext.Provider value={{ libraryService }}>
      {props.children}
    </GlobalStateContext.Provider>
  );
};

export const GlobalStateProvider = (props: Props) => {
  const [isReady, setIsReady] = React.useState(sessionStoreReady);

  // Initialize session store on mount if not already done
  React.useEffect(() => {
    if (!sessionStoreReady) {
      initializeSessionStore().then(() => setIsReady(true));
    }
  }, []);

  // Don't render until session store is initialized
  // This ensures the state machine reads from populated cache
  if (!isReady) {
    return null;
  }

  return <GlobalStateProviderInner {...props} />;
};
