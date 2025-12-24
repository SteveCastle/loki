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
import filter from './filter';
// Job management removed - now handled by external job runner service

export type Item = {
  path: string;
  tagLabel?: string;
  mtimeMs: number;
  weight?: number;
  timeStamp?: number;
  elo?: number | null;
  description?: string;
};

type PersistedLibraryData = {
  library: Item[];
  initialFile: string;
};

type PersistedCursorData = {
  cursor: number;
  scrollPosition?: number;
};

type PersistedQueryData = {
  dbQuery: {
    tags: string[];
  };
  mostRecentTag: string;
  mostRecentCategory: string;
  textFilter: string;
};

type PersistedPreviousData = {
  previousLibrary: Item[];
  previousCursor: number;
};

type Props = {
  children: React.ReactNode;
};

type LibraryState = {
  initialFile: string;
  dbPath: string;
  library: Item[];
  libraryLoadId: string;
  initSessionId: string;
  previousLibrary: Item[];
  cursor: number;
  textFilter: string;
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
  };
  dbQuery: {
    tags: string[];
  };
  commandPalette: {
    display: boolean;
    position: { x: number; y: number };
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
};

const setLibrary = assign<LibraryState, AnyEventObject>({
  library: (context, event) => {
    const library = event.data.library;
    // Update library data
    window.electron.store.set('persistedLibrary', {
      library,
      initialFile: context.initialFile,
    });
    // Update cursor separately
    window.electron.store.set('persistedCursor', {
      cursor: event.data.cursor,
    });
    return library;
  },
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
});

const setLibraryWithPrevious = assign<LibraryState, AnyEventObject>({
  previousLibrary: (context) => context.library,
  previousCursor: (context) => context.cursor,
  library: (context, event) => {
    const library = event.data.library;
    const previousLibrary = context.library;
    const previousCursor = context.cursor;

    // Update library data
    window.electron.store.set('persistedLibrary', {
      library,
      initialFile: context.initialFile,
    });
    // Update cursor separately
    window.electron.store.set('persistedCursor', {
      cursor: event.data.cursor,
    });
    // Update previous state separately
    window.electron.store.set('persistedPrevious', {
      previousLibrary,
      previousCursor,
    });

    return library;
  },
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
  commandPalette: (context) => {
    return {
      ...context.commandPalette,
      display: false,
    };
  },
});

const clearPersistedLibrary = () => {
  window.electron.store.set('persistedLibrary', null);
  window.electron.store.set('persistedCursor', null);
  window.electron.store.set('persistedQuery', null);
  window.electron.store.set('persistedPrevious', null);
};

const updatePersistedCursor = (context: LibraryState, cursor: number) => {
  // Only update cursor - much faster!
  window.electron.store.set('persistedCursor', {
    cursor,
    scrollPosition: context.scrollPosition,
  });
};

const updatePersistedState = (context: LibraryState) => {
  // Update query state separately
  window.electron.store.set('persistedQuery', {
    dbQuery: context.dbQuery,
    mostRecentTag: context.mostRecentTag,
    mostRecentCategory: context.mostRecentCategory,
    textFilter: context.textFilter,
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
  settings: (context, event) => {
    return {
      ...context.settings,
      filters: 'all',
    };
  },
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
    window.electron.store.set('dbPath', event.data);
    return event.data;
  },
});

const hasInitialFile = (context: LibraryState) => !!context.initialFile;
const missingDb = (context: LibraryState) => !context.dbPath;
const hasPersistedLibrary = (context: LibraryState): boolean => {
  const persistedData = window.electron.store.get(
    'persistedLibrary',
    null
  ) as PersistedLibraryData | null;
  return !!(
    persistedData &&
    persistedData.library &&
    Array.isArray(persistedData.library) &&
    persistedData.library.length > 0
  );
};

const hasPersistedTextFilter = (context: LibraryState): boolean => {
  const persistedData = window.electron.store.get(
    'persistedQuery',
    null
  ) as PersistedQueryData | null;
  return !!(
    persistedData &&
    persistedData.textFilter &&
    persistedData.textFilter.length > 0
  );
};

const hasPersistedTags = (context: LibraryState): boolean => {
  const persistedData = window.electron.store.get(
    'persistedQuery',
    null
  ) as PersistedQueryData | null;
  return !!(
    persistedData &&
    persistedData.dbQuery &&
    persistedData.dbQuery.tags &&
    persistedData.dbQuery.tags.length > 0
  );
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
  const batched = (window.electron.store as any).getMany([
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
  ] as [string, any][]);

  return {
    initialFile: window.appArgs?.filePath || '',
    dbPath: batched['dbPath'] as string,
    library: [],
    libraryLoadId: '',
    initSessionId: '',
    textFilter: '',
    activeCategory: batched['activeCategory'] as string,
    storedCategories: batched['storedCategories'] as { [key: string]: string },
    storedTags: batched['storedTags'] as { [key: string]: string[] },
    mostRecentTag: '',
    mostRecentCategory: '',
    cursor: 0,
    previousLibrary: [],
    previousCursor: 0,
    scrollPosition: 0,
    previousScrollPosition: 0,
    videoPlayer: {
      eventId: 'initial',
      timeStamp: 0,
      playing: true,
      videoLength: 0,
      actualVideoTime: 0,
      loopLength: 0,
      loopStartTime: 0,
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
    },
    dbQuery: {
      tags: [],
    },
    commandPalette: {
      display: false,
      position: { x: 0, y: 0 },
    },
    // jobs: removed - now handled by external job runner service
    toasts: [],
    streaming: false,
    pinnedPath: null,
    savedSortByDuringStreaming: null,
    userMovedCursorDuringStreaming: false,
    scrollToCursorEventId: null,
  };
};

const libraryMachine = createMachine(
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
    },
    states: {
      // jobQueue: removed - jobs now handled by external job runner service
      settings: {
        on: {
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

                  window.electron.store.set(key, processedData[key]);
                  // Handle alwaysOnTop setting specially
                  if (key === 'alwaysOnTop') {
                    window.electron.ipcRenderer.sendMessage(
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
          CHANGE_HOTKEY: {
            actions: assign<LibraryState, AnyEventObject>({
              hotKeys: (context, event) => {
                console.log('CHANGE_HOTKEY', context, event);
                for (const key in event.data) {
                  window.electron.store.set(key, event.data[key]);
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
                  window.electron.store.set(`storedCategories`, {
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
                  window.electron.store.set(`storedTags`, {
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
                    path.normalize(item?.path) ===
                      path.normalize(event.currentItem?.path)
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
                    window.electron.store.set('dbPath', window.appArgs?.dbPath);
                    return window.appArgs?.dbPath;
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
                return window.electron.ipcRenderer.invoke('select-db', [
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
                return window.electron.ipcRenderer.invoke('load-db', [
                  context.dbPath,
                ]);
              },
              onDone: {
                target: 'init',
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
              { target: 'selecting' },
            ],
            entry: assign<LibraryState, AnyEventObject>({
              initSessionId: () => uniqueId(),
            }),
          },
          selecting: {
            invoke: {
              src: (context, event) => {
                const currentFile = context.initialFile;
                console.log('selecting', context, event);
                return window.electron.ipcRenderer.invoke('select-file', [
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
                return window.electron.ipcRenderer.invoke('select-directory', [
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
                return window.electron.ipcRenderer.invoke('load-files', [
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
                  'setLibrary',
                  // First: restore sort and compute final cursor while pinnedPath is still available
                  assign<LibraryState, AnyEventObject>({
                    // Restore original sort mode and selection
                    settings: (context) => ({
                      ...context.settings,
                      sortBy: (context.savedSortByDuringStreaming ||
                        'name') as any,
                    }),
                    cursor: (context, event) => {
                      const lib = (event.data?.library || []) as Item[];
                      if (!Array.isArray(lib) || lib.length === 0) {
                        return event.data?.cursor ?? context.cursor;
                      }
                      const restoredSort =
                        (context.savedSortByDuringStreaming || 'name') as any;
                      const tempLibraryLoadId = 'final-' + uniqueId();
                      const sorted = filter(
                        tempLibraryLoadId,
                        context.textFilter,
                        lib,
                        context.settings.filters,
                        restoredSort
                      );
                      const preferredPaths = [
                        context.pinnedPath || undefined,
                        context.library[context.cursor]?.path || undefined,
                      ].filter(Boolean) as string[];
                      for (const p of preferredPaths) {
                        const idx = sorted.findIndex(
                          (it: Item) =>
                            it?.path &&
                            path.normalize(it.path) === path.normalize(p)
                        );
                        if (idx !== -1) {
                          updatePersistedCursor(context, idx);
                          return idx;
                        }
                      }
                      return event.data?.cursor ?? context.cursor;
                    },
                  }),
                  // Second: now clear streaming flags and pinned state
                  assign<LibraryState, AnyEventObject>({
                    streaming: () => false,
                    pinnedPath: () => null,
                    userMovedCursorDuringStreaming: () => false,
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
                    const existing = new Set(
                      context.library.map((item) => item.path)
                    );
                    const incoming = (event.data?.files || []) as Item[];
                    const newItems = incoming.filter(
                      (f) => f?.path && !existing.has(f.path)
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
                    window.electron.store.set(
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
              src: (context, event) => {
                console.log('loading from DB', context, event);
                return window.electron.loadMediaFromDB(
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
              },
              onDone: {
                target: 'loadedFromDB',
                actions: ['setLibraryWithPrevious'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          loadingFromSearch: {
            invoke: {
              src: (context, event) => {
                console.log(
                  'initial search',
                  context.textFilter,
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
                return window.electron.loadMediaByDescriptionSearch(
                  context.textFilter,
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
              },
              onDone: {
                target: 'loadedFromSearch',
                actions: ['setLibraryWithPrevious'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          changingSearch: {
            invoke: {
              src: (context, event) => {
                console.log(
                  'changing Search',
                  context.textFilter,
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
                return window.electron.loadMediaByDescriptionSearch(
                  context.textFilter,
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
              },
              onDone: {
                target: 'loadedFromSearch',
                actions: ['setLibrary'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          switchingTag: {
            invoke: {
              src: (context, event) => {
                console.log('switchingTag', context, event);
                return window.electron.loadMediaFromDB(
                  context.dbQuery.tags,
                  context.settings.filteringMode
                );
              },
              onDone: {
                target: 'loadedFromDB',
                actions: ['setLibrary'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
          },
          loadingFromPersisted: {
            entry: assign<LibraryState, AnyEventObject>({
              library: (context) => {
                const persistedData = window.electron.store.get(
                  'persistedLibrary',
                  null
                ) as PersistedLibraryData | null;
                return persistedData ? persistedData.library : [];
              },
              initialFile: (context) => {
                const persistedData = window.electron.store.get(
                  'persistedLibrary',
                  null
                ) as PersistedLibraryData | null;
                return persistedData
                  ? persistedData.initialFile
                  : context.initialFile;
              },
              cursor: (context) => {
                const cursorData = window.electron.store.get(
                  'persistedCursor',
                  null
                ) as PersistedCursorData | null;
                return cursorData ? cursorData.cursor : 0;
              },
              scrollPosition: (context) => {
                const cursorData = window.electron.store.get(
                  'persistedCursor',
                  null
                ) as PersistedCursorData | null;
                return cursorData?.scrollPosition ?? 0;
              },
              previousLibrary: (context) => {
                const previousData = window.electron.store.get(
                  'persistedPrevious',
                  null
                ) as PersistedPreviousData | null;
                return previousData ? previousData.previousLibrary : [];
              },
              previousCursor: (context) => {
                const previousData = window.electron.store.get(
                  'persistedPrevious',
                  null
                ) as PersistedPreviousData | null;
                return previousData ? previousData.previousCursor : 0;
              },
              dbQuery: (context) => {
                const queryData = window.electron.store.get(
                  'persistedQuery',
                  null
                ) as PersistedQueryData | null;
                return queryData ? queryData.dbQuery : { tags: [] };
              },
              mostRecentTag: (context) => {
                const queryData = window.electron.store.get(
                  'persistedQuery',
                  null
                ) as PersistedQueryData | null;
                return queryData ? queryData.mostRecentTag : '';
              },
              mostRecentCategory: (context) => {
                const queryData = window.electron.store.get(
                  'persistedQuery',
                  null
                ) as PersistedQueryData | null;
                return queryData ? queryData.mostRecentCategory : '';
              },
              textFilter: (context) => {
                const queryData = window.electron.store.get(
                  'persistedQuery',
                  null
                ) as PersistedQueryData | null;
                return queryData ? queryData.textFilter : '';
              },
              libraryLoadId: () => uniqueId(),
            }),
            always: [
              { target: 'loadedFromSearch', cond: hasPersistedTextFilter },
              { target: 'loadedFromDB', cond: hasPersistedTags },
              { target: 'loadedFromFS' },
            ],
          },
          loadingFromPreviousLibrary: {
            entry: assign<LibraryState, AnyEventObject>({
              library: (context) => {
                const library = context.previousLibrary;
                // Persist the restored library using separate keys
                window.electron.store.set('persistedLibrary', {
                  library,
                  initialFile: context.initialFile,
                });
                window.electron.store.set('persistedCursor', {
                  cursor: context.previousCursor,
                });
                window.electron.store.set('persistedPrevious', {
                  previousLibrary: [],
                  previousCursor: 0,
                });
                updatePersistedState(context);
                return library;
              },
              libraryLoadId: () => uniqueId(),
              cursor: (context) => context.previousCursor,
            }),
            always: { target: 'loadedFromFS' },
          },
          loadedFromFS: {
            initial: 'idle',
            on: {
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

                    window.electron.store.set(
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
                        window.electron.ipcRenderer.invoke('delete-file', [
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

                      window.electron.store.set(key, processedData[key]);
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
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'SET QUERY TAG TO',
                          context,
                          event.data.tag
                        );
                        return { tags: [event.data.tag] };
                      },
                    }),
                    (context, event) => {
                      // Persist the updated state
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: [event.data.tag] },
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
              ],
              SET_TEXT_FILTER: {
                target: 'loadingFromSearch',
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    textFilter: (context, event) => {
                      console.log('SET_TEXT_FILTER', context, event);
                      return event.data.textFilter;
                    },
                  }),
                  (context, event) => {
                    updatePersistedState({
                      ...context,
                      textFilter: event.data.textFilter,
                    });
                    // Invalidate persisted library snapshot to avoid query/library mismatch
                    window.electron.store.set('persistedLibrary', null);
                    window.electron.store.set('persistedCursor', null);
                  },
                ],
              },
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
            },
            states: {
              idle: {
                on: {
                  LOAD_FILES_BATCH: {
                    actions: assign<LibraryState, AnyEventObject>({
                      library: (context, event) => {
                        const previousSelectedPath =
                          context.pinnedPath ||
                          context.library[context.cursor]?.path;
                        const existing = new Set(
                          context.library.map((item) => item.path)
                        );
                        const incoming = (event.data?.files || []) as Item[];
                        const newItems = incoming.filter(
                          (f) => f?.path && !existing.has(f.path)
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
                              path.normalize(item.path) ===
                                path.normalize(previousSelectedPath)
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
                },
              },
            },
          },
          loadedFromSearch: {
            initial: 'idle',
            entry: assign<LibraryState, AnyEventObject>({
              libraryLoadId: () => uniqueId(),
            }),
            states: {
              idle: {},
            },
            on: {
              LOAD_FILES_BATCH: {
                actions: assign<LibraryState, AnyEventObject>({
                  library: (context, event) => {
                    const previousSelectedPath =
                      context.pinnedPath ||
                      context.library[context.cursor]?.path;
                    const existing = new Set(
                      context.library.map((item) => item.path)
                    );
                    const incoming = (event.data?.files || []) as Item[];
                    const newItems = incoming.filter(
                      (f) => f?.path && !existing.has(f.path)
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
              SET_QUERY_TAG: [
                {
                  target: 'changingSearch',
                  cond: willHaveTag,
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'willHaveTag branch',
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
                    }),
                    (context, event) => {
                      // Calculate the new tags the same way as above for persistence
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
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
                {
                  target: 'changingSearch',
                  cond: willHaveNoTag,
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'will have no tag branch',
                          context,
                          event.data.tag
                        );
                        return { tags: [] };
                      },
                    }),
                    (context, event) => {
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: [] },
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
              ],
              SET_TEXT_FILTER: [
                {
                  cond: notEmpty,
                  target: 'changingSearch',
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      textFilter: (context, event) => {
                        console.log('Changing Search', context, event);
                        return event.data.textFilter;
                      },
                    }),
                    (context, event) => {
                      updatePersistedState({
                        ...context,
                        textFilter: event.data.textFilter,
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
                {
                  cond: isEmpty,
                  target: 'loadingFromPreviousLibrary',
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      textFilter: (context, event) => {
                        console.log('Clearing search', context, event);
                        return event.data.textFilter;
                      },
                    }),
                    (context, event) => {
                      updatePersistedState({
                        ...context,
                        textFilter: event.data.textFilter,
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
              ],
              DELETE_FILE: {
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    library: (context, event) => {
                      console.log('DELETE_FILE', context, event);
                      try {
                        window.electron.ipcRenderer.invoke('delete-file', [
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
              SET_ACTIVE_CATEGORY: {
                actions: assign<LibraryState, AnyEventObject>({
                  activeCategory: (context, event) => {
                    console.log('SET_ACTIVE_CATEGORY', context, event);
                    window.electron;
                    return event.data.category;
                  },
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
                  assign<LibraryState, AnyEventObject>({
                    previousLibrary: (context) => context.library,
                    previousCursor: (context) => context.cursor,
                    textFilter: () => '',
                    initialFile: (context, event) => event.path,
                  }),
                ],
              },
              CHANGE_DB_PATH: {
                target: 'selectingDB',
              },
              UPDATE_FILE_PATH: {
                target: 'selectingFilePath',
              },
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
              SET_SCROLL_POSITION: {
                actions: assign<LibraryState, AnyEventObject>({
                  scrollPosition: (context, event) => {
                    return event.position;
                  },
                }),
              },
            },
          },
          loadedFromDB: {
            initial: 'idle',
            entry: (context, event) =>
              console.log('loadedFromDB', context, event),
            on: {
              LOAD_FILES_BATCH: {
                actions: assign<LibraryState, AnyEventObject>({
                  library: (context, event) => {
                    const previousSelectedPath =
                      context.pinnedPath ||
                      context.library[context.cursor]?.path;
                    const existing = new Set(
                      context.library.map((item) => item.path)
                    );
                    const incoming = (event.data?.files || []) as Item[];
                    const newItems = incoming.filter(
                      (f) => f?.path && !existing.has(f.path)
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
                          path.normalize(item.path) ===
                            path.normalize(previousSelectedPath)
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
                  assign<LibraryState, AnyEventObject>({
                    previousLibrary: (context) => context.library,
                    previousCursor: (context) => context.cursor,
                    textFilter: () => '',
                    initialFile: (context, event) => event.path,
                  }),
                ],
              },
              SET_ACTIVE_CATEGORY: {
                actions: assign<LibraryState, AnyEventObject>({
                  activeCategory: (context, event) => {
                    console.log('SET_ACTIVE_CATEGORY', context, event);
                    window.electron;
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
                target: 'loadingFromPreviousLibrary',
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    dbQuery: (context, event) => {
                      console.log('CLEAR QUERY TAG', context, event);
                      return { tags: [] };
                    },
                  }),
                  (context, event) => {
                    updatePersistedState({
                      ...context,
                      dbQuery: { tags: [] },
                    });
                    // Invalidate persisted library snapshot to avoid query/library mismatch
                    window.electron.store.set('persistedLibrary', null);
                    window.electron.store.set('persistedCursor', null);
                  },
                ],
              },
              SET_TEXT_FILTER: {
                target: 'changingSearch',
                actions: [
                  assign<LibraryState, AnyEventObject>({
                    cursor: 0,
                    libraryLoadId: () => uniqueId(),
                    textFilter: (context, event) => {
                      console.log('SET_TEXT_FILTER', context, event);
                      return event.data.textFilter;
                    },
                  }),
                  (context, event) => {
                    updatePersistedState({
                      ...context,
                      textFilter: event.data.textFilter,
                    });
                    // Invalidate persisted library snapshot to avoid query/library mismatch
                    window.electron.store.set('persistedLibrary', null);
                    window.electron.store.set('persistedCursor', null);
                  },
                ],
              },
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
                        window.electron.ipcRenderer.invoke('delete-file', [
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
                  target: 'switchingTag',
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
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
                {
                  target: 'loadingFromPreviousLibrary',
                  cond: willHaveNoTag,
                  actions: [
                    assign<LibraryState, AnyEventObject>({
                      dbQuery: (context, event) => {
                        console.log(
                          'SET QUERY TAG TO',
                          context,
                          event.data.tag
                        );
                        return { tags: [] };
                      },
                    }),
                    (context, event) => {
                      updatePersistedState({
                        ...context,
                        dbQuery: { tags: [] },
                      });
                      // Invalidate persisted library snapshot to avoid query/library mismatch
                      window.electron.store.set('persistedLibrary', null);
                      window.electron.store.set('persistedCursor', null);
                    },
                  ],
                },
              ],
              DELETED_ASSIGNMENT: {
                target: 'switchingTag',
                actions: () => console.log('deleted assignment'),
              },
              SORTED_WEIGHTS: {
                target: 'switchingTag',
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
                },
              },
            },
          },
          selectingFilePath: {
            invoke: {
              src: (context, event) => {
                console.log('selectingFilePath', context, event);
                return window.electron.ipcRenderer.invoke('select-new-path', [
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

export const GlobalStateProvider = (props: Props) => {
  const libraryService = useInterpret(libraryMachine);

  React.useEffect(() => {
    const offBatch = window.electron.ipcRenderer.on(
      'load-files-batch',
      (...args: unknown[]) => {
        const batch = (args[0] as { path: string; mtimeMs: number }[]) || [];
        libraryService.send({
          type: 'LOAD_FILES_BATCH',
          data: { files: batch },
        });
      }
    );
    const offDone = window.electron.ipcRenderer.on('load-files-done', () => {
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
      window.electron.store.set('persistedCursor', {
        cursor,
        scrollPosition,
      });
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
