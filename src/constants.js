export const HOT_KEY_DEFAULTS = {
  f: "fileOptions.changeFile",
  z: "fileOptions.toggleRecursion",
  x: "fileOptions.shuffle",
  ArrowUp: "fileOptions.nextImage",
  ArrowDown: "fileOptions.previousImage",
  " ": "windowOptions.minimize",
  "]": "windowOptions.openDevTools",
  c: "windowOptions.toggleFullscreen",
  v: "windowOptions.toggleAlwaysOnTop",
  1: "listOptions.toggleSortOrder",
  2: "listOptions.showALL",
  3: "listOptions.showSTATIC",
  4: "listOptions.showVIDEO",
  5: "listOptions.showGIF",
  6: "listOptions.showMOTION",
  q: "imageOptions.toggleSizing",
  w: "imageOptions.sizeOVERSCAN",
  e: "imageOptions.sizeACTUAL",
  r: "imageOptions.sizeFIT",
  t: "imageOptions.sizeCOVER",
  a: "imageOptions.toggleAudio",
  s: "imageOptions.toggleVideoControls",
  "/": "controlOptions.toggleControls",
};

export const VIEW = {
  DETAIL: "DETAIL",
  LIST: "LIST",
};

export const CONTROL_MODE = {
  MOUSE: { title: "Mouse and Scroll Wheel", key: "MOUSE" },
  TRACK_PAD: { title: "Track Pad or Magic Mouse", key: "TRACK_PAD" },
};

export const SIZE = {
  OVERSCAN: { title: "Over Scan Mode", className: "overscan", key: "OVERSCAN" },
  ACTUAL: { title: "Actual Size", className: "actual", key: "ACTUAL" },
  FIT: { title: "Fit", className: "fit", key: "FIT" },
  COVER: { title: "Cover", className: "cover", key: "COVER" },
};

export const LIST_SIZE = {
  OVERSCAN: { title: "Over Scan", className: "overscan", key: "OVERSCAN" },
  FIT: { title: "Fit Screen", className: "fit", key: "FIT" },
};

export const SORT = {
  ALPHA: { title: "Alphabetical", key: "ALPHA" },
  CREATE_DATE: { title: "Create Date", key: "CREATE_DATE" },
};

export const FILTER = {
  ALL: {
    title: "All",
    key: "ALL",
    value: /jpg$|jpeg$|jfif$|webp$|png$|webm$|mp4$|mpeg$|gif$/i,
  },
  STATIC: {
    title: "Static",
    key: "STATIC",
    value: /jpg$|jpeg$|webp$|jfif$|png$/i,
  },
  VIDEO: {
    title: "Videos",
    key: "VIDEO",
    value: /mp4$|mpeg$/i,
  },
  GIF: {
    title: "Gifs",
    key: "GIF",
    value: /gif$/i,
  },
  MOTION: {
    title: "Motion",
    key: "MOTION",
    value: /gif$|mp4$|mpeg$/i,
  },
};

export const EXTENSIONS = {
  img: [".jpg", ".jpeg", ".jfif", ".gif", ".png", ".webp"],
  video: [".webm", ".avi", ".mpg", ".mpeg", ".mp4"],
};

export function getNext(obj, currentKey) {
  const keys = Object.keys(obj);
  const position = keys.findIndex((k) => k === currentKey);
  return obj[keys[(position + 1) % keys.length]];
}
