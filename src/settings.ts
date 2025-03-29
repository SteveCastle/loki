export type ScaleModeOption = 'cover' | 'fit' | 'actual' | number;

export type OrderingOption = 'asc' | 'desc';

export type SortByOption = 'name' | 'date' | 'weight' | 'shuffle' | 'elo';

export type FilterOption = 'all' | 'static' | 'video';

export type FilterModeOption = 'AND' | 'OR' | 'EXCLUSIVE';

export type ShowTagOptions = 'all' | 'list' | 'detail' | 'none';

export type ShowFileInfoOptions = 'all' | 'list' | 'detail' | 'none';

export type ControlMode = 'mouse' | 'touchpad';

export type LibraryLayout = 'left' | 'bottom';

export type ListImageCache =
  | 'thumbnail_path_1200'
  | 'thumbnail_path_600'
  | false;
export type DetailImageCache =
  | 'thumbnail_path_1200'
  | 'thumbnail_path_600'
  | false;

export type SettingKey =
  | 'scaleMode'
  | 'sortBy'
  | 'filters'
  | 'playSound'
  | 'comicMode'
  | 'showTagCount'
  | 'libraryLayout'
  | 'battleMode'
  | 'followTranscript'
  | 'showTags'
  | 'showFileInfo'
  | 'showControls'
  | 'recursive'
  | 'controlMode';

export type Settings = {
  scaleMode: ScaleModeOption;
  order: OrderingOption;
  sortBy: SortByOption;
  filters: FilterOption;
  playSound: boolean;
  comicMode: boolean;
  showTagCount: boolean;
  libraryLayout: LibraryLayout;
  battleMode: boolean;
  showTags: ShowTagOptions;
  showFileInfo: ShowFileInfoOptions;
  showControls: boolean;
  recursive: boolean;
  applyTagToAll: boolean;
  followTranscript: boolean;
  applyTagPreview: boolean;
  filteringMode: FilterModeOption;
  listImageCache: ListImageCache;
  detailImageCache: DetailImageCache;
  controlMode: ControlMode;
  gridSize: [number, number];
};

export const SCALE_MODES = {
  title: 'Scale Mode',
  reload: false,
  display: 'image',
  key: 'scaleMode',
  options: {
    cover: {
      label: 'Cover',
      value: 'cover',
    },
    fit: {
      label: 'Fit',
      value: 'fit',
    },
    overscan: {
      label: 'Overscan',
      value: 140,
    },
    actual: {
      label: 'Actual',
      value: 'actual',
    },
  },
};

export const ORDERING = {
  title: 'Ordering',
  reload: false,
  display: 'image',
  options: {
    asc: {
      label: 'Ascending',
      value: 'asc',
    },
    desc: {
      label: 'Descending',
      value: 'desc',
    },
  },
};

export const SORT_BY = {
  title: 'Sort By',
  reload: false,
  display: 'image',
  resetCursor: true,
  options: {
    name: {
      label: 'Name',
      value: 'name',
    },
    date: {
      label: 'Date',
      value: 'date',
    },
    weight: {
      label: 'Custom',
      value: 'weight',
    },
    elo: {
      label: 'Elo',
      value: 'elo',
    },
    shuffle: {
      label: 'Shuffle',
      value: 'shuffle',
    },
  },
};

export const FILTERS = {
  title: 'Media Types',
  display: 'image',
  reload: false,
  resetCursor: true,
  options: {
    all: {
      label: 'All',
      value: 'all',
    },
    static: {
      label: 'Still',
      value: 'static',
    },
    video: {
      label: 'Motion',
      value: 'video',
    },
  },
};

type SettingsObject = {
  [key in SettingKey]: {
    title: string;
    display: string;
    reload: boolean;
    resetCursor?: boolean;
    options: {
      [key: string]: {
        label: string;
        value: string | number | boolean;
      };
    };
  };
};

export const PLAY_SOUND = {
  title: 'Play Sound',
  reload: false,
  display: 'none',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const COMIC_MODE = {
  title: 'Comic Mode',
  reload: false,
  display: 'image',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const SHOW_TAG_COUNT = {
  title: 'Show Tag Count',
  reload: false,
  display: 'general',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const BATTLE_MODE = {
  title: 'Battle Mode',
  reload: false,
  display: 'general',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const SHOW_TAGS = {
  title: 'Show Tags',
  reload: false,
  display: 'general',
  options: {
    all: {
      label: 'All',
      value: 'all',
    },
    list: {
      label: 'List',
      value: 'list',
    },
    detail: {
      label: 'Detail',
      value: 'detail',
    },
    none: {
      label: 'None',
      value: 'none',
    },
  },
};

export const SHOW_FILE_INFO = {
  title: 'Show File Info',
  reload: false,
  display: 'general',
  options: {
    all: {
      label: 'All',
      value: 'all',
    },
    list: {
      label: 'List',
      value: 'list',
    },
    detail: {
      label: 'Detail',
      value: 'detail',
    },
    none: {
      label: 'None',
      value: 'none',
    },
  },
};

export const SHOW_CONTROLS = {
  title: 'Show Controls',
  reload: false,
  display: 'none',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const FOLLOW_TRANSCRIPT = {
  title: 'Follow Transcript',
  reload: false,
  display: 'none',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const RECURSIVE = {
  title: 'Recursive',
  reload: true,
  display: 'none',
  options: {
    name: {
      label: 'Yes',
      value: true,
    },
    date: {
      label: 'No',
      value: false,
    },
  },
};

export const CONTROL_MODE = {
  title: 'Control Mode',
  reload: false,
  display: 'general',
  options: {
    name: {
      label: 'Mouse',
      value: 'mouse',
    },
    date: {
      label: 'TouchPad',
      value: 'touchpad',
    },
  },
};

export const LIBRARY_LAYOUT = {
  title: 'Library Position',
  reload: false,
  display: 'general',
  options: {
    name: {
      label: 'Left',
      value: 'left',
    },
    date: {
      label: 'Bottom',
      value: 'bottom',
    },
  },
};

export function getNextFilterMode(
  currentMode: FilterModeOption
): FilterModeOption {
  switch (currentMode) {
    case 'AND':
      return 'OR';
    case 'OR':
      return 'EXCLUSIVE';
    case 'EXCLUSIVE':
      return 'AND';
    default:
      throw new Error(`Invalid filter mode: ${currentMode}`);
  }
}

export const SETTINGS: SettingsObject = {
  scaleMode: SCALE_MODES,
  sortBy: SORT_BY,
  filters: FILTERS,
  playSound: PLAY_SOUND,
  comicMode: COMIC_MODE,
  showTagCount: SHOW_TAG_COUNT,
  battleMode: BATTLE_MODE,
  libraryLayout: LIBRARY_LAYOUT,
  followTranscript: FOLLOW_TRANSCRIPT,
  showTags: SHOW_TAGS,
  showFileInfo: SHOW_FILE_INFO,
  showControls: SHOW_CONTROLS,
  recursive: RECURSIVE,
  controlMode: CONTROL_MODE,
};
