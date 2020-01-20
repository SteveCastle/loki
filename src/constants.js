export const VIEW = {
  DETAIL: "DETAIL",
  LIST: "LIST"
};

export const CONTROL_MODE = {
  MOUSE: "MOUSE",
  TRACK_PAD: "TRACK_PAD"
};

export const SIZE = {
  OVERSCAN: "OVERSCAN",
  ACTUAL: "ACTUAL"
};

export const SORT = {
  ALPHA: "ALPHA",
  CREATE_DATE: "CREATE_DATE"
};

export const FILTER = {
  ALL: ["**.jpg", "**.gif", "**.jpeg", "**.png", "**.webm", "**.mp4"],
  STATIC: ["**.jpg", "**.jpeg", "**.png"],
  VIDEO: ["**.webm", "**.mp4", "**.mpeg"],
  GIF: ["**.gif"]
};

export const EXTENSIONS = {
  img: [".jpg", ".jpeg", ".gif", ".png"],
  video: [".webm", ".avi", ".mpg", ".mpeg", ".mp4"]
};

export function getNext(obj, currentKey) {
  Object.keys(obj).reduce((acc, k) => k);
}
