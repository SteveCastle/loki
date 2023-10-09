export type ScaleModeOption = 'cover' | 'fit' | 'actual' | number;

export type OrderingOption = 'asc' | 'desc';

export type SortByOption = 'name' | 'date' | 'weight' | 'shuffle';

export type FilterOption = 'all' | 'static' | 'video';

export type FilterModeOption = 'AND' | 'OR' | 'EXCLUSIVE';

export type ShowTagOptions = 'all' | 'list' | 'detail' | 'none';

export type ShowFileInfoOptions = 'all' | 'list' | 'detail' | 'none';

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
  | 'followTranscript'
  | 'showTags'
  | 'showFileInfo'
  | 'showControls'
  | 'recursive';

export type Settings = {
  scaleMode: ScaleModeOption;
  order: OrderingOption;
  sortBy: SortByOption;
  filters: FilterOption;
  playSound: boolean;
  comicMode: boolean;
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
  gridSize: [number, number];
};

export const SCALE_MODES = {
  title: 'Scale Mode',
  reload: false,
  display: true,
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
  display: true,
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
  display: true,
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
      label: 'Weight',
      value: 'weight',
    },
    shuffle: {
      label: 'Shuffle',
      value: 'shuffle',
    },
  },
};

export const FILTERS = {
  title: 'Media Types',
  display: true,
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
    display: boolean;
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
  display: false,
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
  display: true,
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
  display: true,
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
  display: true,
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
  display: false,
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
  display: false,
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
  display: false,
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
  followTranscript: FOLLOW_TRANSCRIPT,
  showTags: SHOW_TAGS,
  showFileInfo: SHOW_FILE_INFO,
  showControls: SHOW_CONTROLS,
  recursive: RECURSIVE,
};
