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
import { Job, JobQueue } from '../main/jobs';

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
  cursor: number;
  previousLibrary?: Item[];
  previousCursor?: number;
  dbQuery?: {
    tags: string[];
  };
  mostRecentTag?: string;
  mostRecentCategory?: string;
  textFilter?: string;
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
  jobs: JobQueue;
  toasts: {
    id: string;
    type: 'success' | 'error' | 'info';
    title: string;
    message?: string;
    timestamp: number;
  }[];
};

const setLibrary = assign<LibraryState, AnyEventObject>({
  library: (context, event) => {
    const library = event.data.library;
    window.electron.store.set('persistedLibrary', {
      library,
      initialFile: context.initialFile,
      cursor: event.data.cursor,
      previousLibrary: context.previousLibrary,
      previousCursor: context.previousCursor,
      dbQuery: context.dbQuery,
      mostRecentTag: context.mostRecentTag,
      mostRecentCategory: context.mostRecentCategory,
      textFilter: context.textFilter,
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
    window.electron.store.set('persistedLibrary', {
      library,
      initialFile: context.initialFile,
      cursor: event.data.cursor,
      previousLibrary,
      previousCursor,
      dbQuery: context.dbQuery,
      mostRecentTag: context.mostRecentTag,
      mostRecentCategory: context.mostRecentCategory,
      textFilter: context.textFilter,
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
};

const updatePersistedCursor = (context: LibraryState, cursor: number) => {
  const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
  if (persistedData) {
    window.electron.store.set('persistedLibrary', {
      ...persistedData,
      cursor,
      dbQuery: context.dbQuery,
      mostRecentTag: context.mostRecentTag,
      mostRecentCategory: context.mostRecentCategory,
      textFilter: context.textFilter,
    });
  }
};

const updatePersistedState = (context: LibraryState) => {
  const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
  if (persistedData) {
    window.electron.store.set('persistedLibrary', {
      ...persistedData,
      dbQuery: context.dbQuery,
      mostRecentTag: context.mostRecentTag,
      mostRecentCategory: context.mostRecentCategory,
      textFilter: context.textFilter,
    });
  }
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

const createJob = assign<LibraryState, AnyEventObject>({
  jobs: (context, event) => {
    console.log('createJob', event);
    const job = event.data as Job;
    const jobs = new Map(context.jobs);
    jobs.set(job.id, job);
    return jobs;
  },
});

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
  const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
  return !!(persistedData && 
         persistedData.library && 
         Array.isArray(persistedData.library) && 
         persistedData.library.length > 0);
};

const hasPersistedTextFilter = (context: LibraryState): boolean => {
  const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
  return !!(persistedData && persistedData.textFilter && persistedData.textFilter.length > 0);
};

const hasPersistedTags = (context: LibraryState): boolean => {
  const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
  return !!(persistedData && persistedData.dbQuery && persistedData.dbQuery.tags && persistedData.dbQuery.tags.length > 0);
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
const getInitialContext = (): LibraryState => ({
  initialFile: window.appArgs?.filePath || '',
  dbPath: window.electron.store.get('dbPath', null) as string,
  library: [],
  libraryLoadId: '',
  initSessionId: '',
  textFilter: '',
  activeCategory: window.electron.store.get('activeCategory', '') as string,
  storedCategories: window.electron.store.get('storedCategories', {}) as {
    [key: string]: string;
  },
  storedTags: window.electron.store.get('storedTags', {}) as {
    [key: string]: string[];
  },
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
    order: window.electron.store.get('sortOrder', 'asc') as 'asc' | 'desc',
    sortBy: window.electron.store.get('sortBy', 'name') as
      | 'name'
      | 'date'
      | 'weight'
      | 'elo'
      | 'shuffle',
    filters: 'all',
    recursive: false,
    scale: 1,
    comicMode: window.electron.store.get('comicMode', false) as boolean,
    showTagCount: window.electron.store.get('showTagCount', false) as boolean,
    battleMode: window.electron.store.get('battleMode', false) as boolean,
    libraryLayout: window.electron.store.get('libraryLayout', 'bottom') as
      | 'left'
      | 'bottom',
    applyTagPreview: window.electron.store.get(
      'applyTagPreview',
      true
    ) as boolean,
    filteringMode: window.electron.store.get(
      'filteringMode',
      'EXCLUSIVE'
    ) as FilterModeOption,
    applyTagToAll: window.electron.store.get('applyTagToAll', false) as boolean,
    scaleMode: window.electron.store.get('scaleMode', 'fit') as
      | 'fit'
      | 'cover'
      | number,
    playSound: window.electron.store.get('playSound', false) as boolean,
    followTranscript: window.electron.store.get(
      'followTranscript',
      true
    ) as boolean,
    showTags: window.electron.store.get('showTags', 'all') as
      | 'all'
      | 'list'
      | 'detail'
      | 'none',
    showFileInfo: window.electron.store.get('showFileInfo', 'none') as
      | 'all'
      | 'list'
      | 'detail'
      | 'none',
    showControls: window.electron.store.get('showControls', true) as boolean,
    gridSize: window.electron.store.get('gridSize', [4, 4]) as [number, number],
    listImageCache: window.electron.store.get(
      'listImageCache',
      'thumbnail_path_600'
    ) as 'thumbnail_path_1200' | 'thumbnail_path_600' | false,
    detailImageCache: window.electron.store.get(
      'detailImageCache',
      false
    ) as DetailImageCache,
    controlMode: window.electron.store.get('controlMode', 'mouse') as
      | 'mouse'
      | 'touchpad',
    autoPlay: window.electron.store.get('autoPlay', false) as boolean,
    autoPlayTime: window.electron.store.get('autoPlayTime', false) as
      | number
      | false,
    autoPlayVideoLoops: window.electron.store.get(
      'autoPlayVideoLoops',
      false
    ) as number | false,
    volume: window.electron.store.get('volume', 1.0) as number,
    alwaysOnTop: window.electron.store.get('alwaysOnTop', false) as boolean,
  },
  hotKeys: {
    incrementCursor: window.electron.store.get(
      'incrementCursor',
      'arrowright'
    ) as string,
    decrementCursor: window.electron.store.get(
      'decrementCursor',
      'arrowleft'
    ) as string,
    toggleTagPreview: window.electron.store.get(
      'toggleTagPreview',
      'shift'
    ) as string,
    toggleTagAll: window.electron.store.get(
      'toggleTagAll',
      'control'
    ) as string,
    moveToTop: window.electron.store.get('moveToTop', '[') as string,
    moveToEnd: window.electron.store.get('moveToEnd', ']') as string,
    minimize: window.electron.store.get('minimize', 'escape') as string,
    shuffle: window.electron.store.get('shuffle', 'x') as string,
    copyFile: window.electron.store.get('copyFilePath', 'c+control') as string,
    copyAllSelectedFiles: window.electron.store.get(
      'copyAllSelectedFiles',
      'c+control+shift'
    ) as string,
    deleteFile: window.electron.store.get('deleteFile', 'delete') as string,
    applyMostRecentTag: window.electron.store.get(
      'applyMostRecentTag',
      'a'
    ) as string,
    storeCategory1: window.electron.store.get(
      'storeCategory1',
      '1+alt'
    ) as string,
    storeCategory2: window.electron.store.get(
      'storeCategory2',
      '2+alt'
    ) as string,
    storeCategory3: window.electron.store.get(
      'storeCategory3',
      '3+alt'
    ) as string,
    storeCategory4: window.electron.store.get(
      'storeCategory4',
      '4+alt'
    ) as string,
    storeCategory5: window.electron.store.get(
      'storeCategory5',
      '5+alt'
    ) as string,
    storeCategory6: window.electron.store.get(
      'storeCategory6',
      '6+alt'
    ) as string,
    storeCategory7: window.electron.store.get(
      'storeCategory7',
      '7+alt'
    ) as string,
    storeCategory8: window.electron.store.get(
      'storeCategory8',
      '8+alt'
    ) as string,
    storeCategory9: window.electron.store.get(
      'storeCategory9',
      '9+alt'
    ) as string,
    tagCategory1: window.electron.store.get('tagCategory1', '!') as string,
    tagCategory2: window.electron.store.get('tagCategory2', '@') as string,
    tagCategory3: window.electron.store.get('tagCategory3', '#') as string,
    tagCategory4: window.electron.store.get('tagCategory4', '$') as string,
    tagCategory5: window.electron.store.get('tagCategory5', '%') as string,
    tagCategory6: window.electron.store.get('tagCategory6', '^') as string,
    tagCategory7: window.electron.store.get('tagCategory7', '&') as string,
    tagCategory8: window.electron.store.get('tagCategory8', '*') as string,
    tagCategory9: window.electron.store.get('tagCategory9', '(') as string,
    storeTag1: window.electron.store.get('storeTag1', '1+control') as string,
    storeTag2: window.electron.store.get('storeTag2', '2+control') as string,
    storeTag3: window.electron.store.get('storeTag3', '3+control') as string,
    storeTag4: window.electron.store.get('storeTag4', '4+control') as string,
    storeTag5: window.electron.store.get('storeTag5', '5+control') as string,
    storeTag6: window.electron.store.get('storeTag6', '6+control') as string,
    storeTag7: window.electron.store.get('storeTag7', '7+control') as string,
    storeTag8: window.electron.store.get('storeTag8', '8+control') as string,
    storeTag9: window.electron.store.get('storeTag9', '9+control') as string,
    applyTag1: window.electron.store.get('applyTag1', '1') as string,
    applyTag2: window.electron.store.get('applyTag2', '2') as string,
    applyTag3: window.electron.store.get('applyTag3', '3') as string,
    applyTag4: window.electron.store.get('applyTag4', '4') as string,
    applyTag5: window.electron.store.get('applyTag5', '5') as string,
    applyTag6: window.electron.store.get('applyTag6', '6') as string,
    applyTag7: window.electron.store.get('applyTag7', '7') as string,
    applyTag8: window.electron.store.get('applyTag8', '8') as string,
    applyTag9: window.electron.store.get('applyTag9', '9') as string,
    togglePlayPause: window.electron.store.get(
      'togglePlayPause',
      ' '
    ) as string,
  },
  dbQuery: {
    tags: [],
  },
  commandPalette: {
    display: false,
    position: { x: 0, y: 0 },
  },
  jobs: new Map<string, Job>(),
  toasts: [],
});

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
      jobQueue: {
        initial: 'waiting',

        states: {
          waiting: {
            on: {
              CREATE_JOB: {
                target: 'creatingJob',
              },
              UPDATE_JOB: {
                actions: assign<LibraryState, AnyEventObject>({
                  jobs: (context, event) => {
                    console.log('update job', context, event);
                    const jobs = new Map(context.jobs);
                    jobs.set(event.job.id, event.job);
                    return jobs;
                  },
                }),
              },
              COMPLETE_JOB: {
                actions: assign<LibraryState, AnyEventObject>({
                  jobs: (context, event) => {
                    console.log('complete job', context, event);
                    const jobs = new Map(context.jobs);
                    jobs.set(event.job.id, event.job);
                    return jobs;
                  },
                }),
              },
              CLEAR_JOB: {
                actions: assign<LibraryState, AnyEventObject>({
                  jobs: (context, event) => {
                    console.log('clear job', context, event);
                    const jobs = new Map(context.jobs);
                    jobs.delete(event.job.id);
                    return jobs;
                  },
                }),
              },
            },
          },
          creatingJob: {
            invoke: {
              src: (context, event) => {
                console.log('adding Job', context, event);
                return window.electron.ipcRenderer.invoke('create-job', [
                  event.paths,
                  event.jobType,
                  event.invalidations,
                ]);
              },
              onDone: {
                target: 'waiting',
                actions: ['createJob'],
              },
              onError: {
                target: 'waiting',
              },
            },
          },
        },
      },
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
                    mostRecentCategory: event.category
                  });
                  return newTag;
                },
                mostRecentCategory: (context, event) => {
                  console.log('SET_MOST_RECENT_CATEGORY', context, event);
                  return event.category;
                },
              })
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
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context) => {
                let newCursor;
                if (
                  context.cursor <
                  filter(
                    context.libraryLoadId,
                    context.textFilter,
                    context.library,
                    context.settings.filters,
                    context.settings.sortBy
                  ).length -
                    1
                ) {
                  newCursor = context.cursor + 1;
                } else {
                  newCursor = 0;
                }
                updatePersistedCursor(context, newCursor);
                return newCursor;
              },
            }),
          },
          DECREMENT_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context) => {
                let newCursor;
                if (context.cursor > 0) {
                  newCursor = context.cursor - 1;
                } else {
                  newCursor = filter(
                    context.libraryLoadId,
                    context.textFilter,
                    context.library,
                    context.settings.filters,
                    context.settings.sortBy
                  ).length - 1;
                }
                updatePersistedCursor(context, newCursor);
                return newCursor;
              },
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
          loadingFromFS: {
            entry: assign<LibraryState, AnyEventObject>({
              library: (context) => [{ path: context.initialFile, mtimeMs: 0 }],
              libraryLoadId: () => uniqueId(),
              cursor: 0,
              dbQuery: () => ({ tags: [] }),
            }),
            invoke: {
              src: (context, event) => {
                console.log('loadingFromFS', context, event);
                const { recursive } = context.settings;
                return window.electron.ipcRenderer.invoke('load-files', [
                  context.initialFile,
                  context.settings.sortBy,
                  recursive,
                ]);
              },
              onDone: {
                target: 'loadedFromFS',
                actions: ['setLibrary'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
            on: {
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
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? persistedData.library : [];
              },
              initialFile: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? persistedData.initialFile : context.initialFile;
              },
              cursor: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? persistedData.cursor : 0;
              },
              previousLibrary: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData && persistedData.previousLibrary ? persistedData.previousLibrary : [];
              },
              previousCursor: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? (persistedData.previousCursor || 0) : 0;
              },
              dbQuery: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData && persistedData.dbQuery ? persistedData.dbQuery : { tags: [] };
              },
              mostRecentTag: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? (persistedData.mostRecentTag || '') : '';
              },
              mostRecentCategory: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? (persistedData.mostRecentCategory || '') : '';
              },
              textFilter: (context) => {
                const persistedData = window.electron.store.get('persistedLibrary', null) as PersistedLibraryData | null;
                return persistedData ? (persistedData.textFilter || '') : '';
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
              library: (context) => context.previousLibrary,
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
                  actions: assign<LibraryState, AnyEventObject>({
                    dbQuery: (context, event) => {
                      console.log('SET QUERY TAG TO', context, event.data.tag);
                      return { tags: [event.data.tag] };
                    },
                  }),
                },
              ],
              SET_TEXT_FILTER: {
                target: 'loadingFromSearch',
                actions: assign<LibraryState, AnyEventObject>({
                  textFilter: (context, event) => {
                    console.log('SET_TEXT_FILTER', context, event);
                    return event.data.textFilter;
                  },
                }),
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
              SET_QUERY_TAG: [
                {
                  target: 'changingSearch',
                  cond: willHaveTag,
                  actions: assign<LibraryState, AnyEventObject>({
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
                },
                {
                  target: 'changingSearch',
                  cond: willHaveNoTag,
                  actions: assign<LibraryState, AnyEventObject>({
                    dbQuery: (context, event) => {
                      console.log(
                        'will have no tag branch',
                        context,
                        event.data.tag
                      );
                      return { tags: [] };
                    },
                  }),
                },
              ],
              SET_TEXT_FILTER: [
                {
                  cond: notEmpty,
                  target: 'changingSearch',
                  actions: assign<LibraryState, AnyEventObject>({
                    textFilter: (context, event) => {
                      console.log('Changing Search', context, event);
                      return event.data.textFilter;
                    },
                  }),
                },
                {
                  cond: isEmpty,
                  target: 'loadingFromPreviousLibrary',
                  actions: assign<LibraryState, AnyEventObject>({
                    textFilter: (context, event) => {
                      console.log('Clearing search', context, event);
                      return event.data.textFilter;
                    },
                  }),
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
              SET_FILE: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  textFilter: () => '',
                  initialFile: (context, event) => event.path,
                }),
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
              SELECT_FILE: {
                target: 'selecting',
              },
              SET_FILE: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  textFilter: () => '',
                  initialFile: (context, event) => event.path,
                }),
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
                actions: assign<LibraryState, AnyEventObject>({
                  dbQuery: (context, event) => {
                    console.log('CLEAR QUERY TAG', context, event);
                    return { tags: [] };
                  },
                }),
              },
              SET_TEXT_FILTER: {
                target: 'changingSearch',
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: 0,
                  libraryLoadId: () => uniqueId(),
                  textFilter: (context, event) => {
                    console.log('SET_TEXT_FILTER', context, event);
                    return event.data.textFilter;
                  },
                }),
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
                  actions: assign<LibraryState, AnyEventObject>({
                    dbQuery: (context, event) => {
                      console.log('SET QUERY TAG TO', context, event.data.tags);
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
                },
                {
                  target: 'loadingFromPreviousLibrary',
                  cond: willHaveNoTag,
                  actions: assign<LibraryState, AnyEventObject>({
                    dbQuery: (context, event) => {
                      console.log('SET QUERY TAG TO', context, event.data.tag);
                      return { tags: [] };
                    },
                  }),
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
      createJob,
      updateFilePath,
    },
  }
);

export const GlobalStateContext = createContext({
  libraryService: {} as InterpreterFrom<typeof libraryMachine>,
});

export const GlobalStateProvider = (props: Props) => {
  const libraryService = useInterpret(libraryMachine);

  return (
    <GlobalStateContext.Provider value={{ libraryService }}>
      {props.children}
    </GlobalStateContext.Provider>
  );
};
