import React, { createContext } from 'react';
import { useInterpret } from '@xstate/react';
import { uniqueId } from 'lodash';
import path from 'path-browserify';
import { AnyEventObject, assign, createMachine, InterpreterFrom } from 'xstate';
import { Settings } from 'settings';
import filter from './filter';
import { Job, JobQueue } from '../main/jobs';

export type Item = {
  path: string;
  mtimeMs: number;
  weight?: number;
  timeStamp?: number;
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
};

const setLibrary = assign<LibraryState, AnyEventObject>({
  library: (_, event) => event.data.library,
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
});

const setLibraryWithPrevious = assign<LibraryState, AnyEventObject>({
  previousLibrary: (context) => context.library,
  previousCursor: (context) => context.cursor,
  library: (_, event) => event.data.library,
  libraryLoadId: () => uniqueId(),
  cursor: (_, event) => event.data.cursor,
  commandPalette: (context) => {
    return {
      ...context.commandPalette,
      display: false,
    };
  },
});

const setPath = assign<LibraryState, AnyEventObject>({
  initialFile: (context, event) => {
    console.log('setPath', event);
    if (!event.data) {
      return context.initialFile;
    }
    return event.data;
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
  }
  return newTagList.length !== 0;
};

const noTag = (context: LibraryState, event: AnyEventObject) => {
  // Detect if the result of toggling a tag is an empty tag list.
  // If so return true.
  const tag = event.data.tag;
  const tagList = context.dbQuery.tags;
  const index = tagList.indexOf(tag);
  const newTagList = [...tagList];
  if (index > -1) {
    newTagList.splice(index, 1);
  }
  return newTagList.length === 0;
};

const libraryMachine = createMachine(
  {
    id: 'library',
    predictableActionArguments: true,
    type: 'parallel',
    context: {
      initialFile: window.appArgs?.filePath || '',
      dbPath: window.electron.store.get('dbPath', null),
      library: [],
      libraryLoadId: '',
      initSessionId: '',
      textFilter: '',
      activeCategory: window.electron.store.get('activeCategory', ''),
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
        order: window.electron.store.get('sortOrder', 'asc'),
        sortBy: window.electron.store.get('sortBy', 'name'),
        filters: window.electron.store.get('filters', 'all'),
        recursive: false,
        scale: 1,
        comicMode: window.electron.store.get('comicMode', false),
        applyTagPreview: window.electron.store.get('applyTagPreview', true),
        filteringMode: window.electron.store.get('filteringMode', 'EXCLUSIVE'),
        applyTagToAll: window.electron.store.get('applyTagToAll', false),
        scaleMode: window.electron.store.get('scaleMode', 'fit'),
        playSound: window.electron.store.get('playSound', false),
        followTranscript: window.electron.store.get('followTranscript', true),
        showTags: window.electron.store.get('showTags', 'all'),
        showFileInfo: window.electron.store.get('showFileInfo', 'none'),
        showControls: window.electron.store.get('showControls', true),
        gridSize: window.electron.store.get('gridSize', [4, 4]),
        listImageCache: window.electron.store.get(
          'listImageCache',
          'thumbnail_path_600'
        ),
        detailImageCache: window.electron.store.get('detailImageCache', false),
      },
      hotKeys: {
        incrementCursor: window.electron.store.get(
          'incrementCursor',
          'arrowright'
        ),
        decrementCursor: window.electron.store.get(
          'decrementCursor',
          'arrowleft'
        ),
        toggleTagPreview: window.electron.store.get(
          'toggleTagPreview',
          'shift'
        ),
        toggleTagAll: window.electron.store.get('toggleTagAll', 'control'),
        moveToTop: window.electron.store.get('moveToTop', '['),
        moveToEnd: window.electron.store.get('moveToEnd', ']'),
        minimize: window.electron.store.get('minimize', 'escape'),
        shuffle: window.electron.store.get('shuffle', 'x'),
        copyFile: window.electron.store.get('copyFilePath', 'c+control'),
        copyAllSelectedFiles: window.electron.store.get(
          'copyAllSelectedFiles',
          'c+control+shift'
        ),
      },
      dbQuery: {
        tags: [],
      },
      commandPalette: {
        display: false,
        position: { x: 0, y: 0 },
      },
      jobs: new Map<string, Job>(),
    } as LibraryState,
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
                for (const key in event.data) {
                  window.electron.store.set(key, event.data[key]);
                }
                return {
                  ...context.settings,
                  ...event.data,
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
      cursor: {
        on: {
          SET_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context, event) => event.idx,
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
                    path.normalize(item?.path) ===
                    path.normalize(event.currentItem?.path)
                );
                console.log('index of current item', event.currentItem, cursor);
                return cursor > -1 ? cursor : 0;
              },
            }),
          },
          INCREMENT_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context) => {
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
                  return context.cursor + 1;
                }
                return context.cursor;
              },
            }),
          },
          DECREMENT_CURSOR: {
            actions: assign<LibraryState, AnyEventObject>({
              cursor: (context) => {
                if (context.cursor > 0) {
                  return context.cursor - 1;
                }
                return context.cursor;
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
                return window.electron.ipcRenderer.invoke('select-db', []);
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
              { target: 'selecting' },
            ],
            entry: assign<LibraryState, AnyEventObject>({
              initSessionId: () => uniqueId(),
            }),
          },
          selecting: {
            invoke: {
              src: (context, event) => {
                console.log('selecting', context, event);
                return window.electron.ipcRenderer.invoke('select-file', []);
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
            }),
            invoke: {
              src: (context, event) => {
                console.log('loadingFromFS', context, event);
                const { recursive } = context.settings;
                return window.electron.loadFiles(
                  context.initialFile,
                  context.settings.filters,
                  context.settings.sortBy,
                  recursive
                );
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
          switchingTag: {
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
                actions: ['setLibrary'],
              },
              onError: {
                target: 'loadedFromFS',
              },
            },
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
            entry: assign<LibraryState, AnyEventObject>({
              cursor: (context) => {
                console.log('loadedFromFS', context);
                const cursor = filter(
                  context.libraryLoadId,
                  context.textFilter,
                  context.library,
                  context.settings.filters,
                  context.settings.sortBy
                ).findIndex((item: Item) => item?.path === context.initialFile);
                return cursor > -1 ? cursor : 0;
              },
              dbQuery: () => ({ tags: [] }),
            }),
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
                  initialFile: (context, event) => event.path,
                }),
              },
              CHANGE_SETTING_AND_RELOAD: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  settings: (context, event) => {
                    console.log('CHANGE_SETTING_AND_RELOAD', context, event);
                    for (const key in event.data) {
                      window.electron.store.set(key, event.data[key]);
                    }
                    return {
                      ...context.settings,
                      ...event.data,
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
                actions: assign<LibraryState, AnyEventObject>({
                  cursor: (context, event) => {
                    return context.textFilter === event.data.textFilter
                      ? context.cursor
                      : 0;
                  },
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
              CLEAR_QUERY_TAG: {
                target: 'loadingFromFS',
                actions: assign<LibraryState, AnyEventObject>({
                  dbQuery: (context, event) => {
                    console.log('CLEAR QUERY TAG', context, event);
                    return { tags: [] };
                  },
                }),
              },
              SET_TEXT_FILTER: {
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
                  cond: noTag,
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
