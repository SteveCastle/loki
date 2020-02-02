export const VIEW = {
  DETAIL: "DETAIL",
  LIST: "LIST"
};

export const CONTROL_MODE = {
  MOUSE: "MOUSE",
  TRACK_PAD: "TRACK_PAD"
};

export const SIZE = {
  OVERSCAN: { title: "Over Scan Mode", className: "overscan", key: "OVERSCAN" },
  ACTUAL: { title: "Actual Size", className: "actual", key: "ACTUAL" }
};

export const LIST_SIZE = {
  OVERSCAN: { title: "Over Scan", className: "overscan", key: "OVERSCAN" },
  FIT: { title: "Fit Screen", className: "fit", key: "FIT" }
};

export const SORT = {
  ALPHA: "ALPHA",
  CREATE_DATE: "CREATE_DATE"
};

export const FILTER = {
  ALL: {
    title: "All",
    key: "ALL",
    value: /jpg$|jpeg$|png$|webm$|mp4$|mpeg$|gif$/
  },
  STATIC: {
    title: "Only Static",
    key: "STATIC",
    value: /jpg$|jpeg$|png$/
  },
  VIDEO: {
    title: "Videos",
    key: "VIDEO",
    value: /mp4$|mpeg$/
  },
  GIF: {
    title: "Animated Gifs",
    key: "GIF",
    value: /gif$/
  },
  MOTION: {
    title: "Motion",
    key: "MOTION",
    value: /gif$|mp4$|mpeg$/
  }
};

export const EXTENSIONS = {
  img: [".jpg", ".jpeg", ".gif", ".png"],
  video: [".webm", ".avi", ".mpg", ".mpeg", ".mp4"]
};

export function getNext(obj, currentKey) {
  console.log(obj, currentKey);
  const keys = Object.keys(obj);
  const position = keys.findIndex(k => k === currentKey);
  console.log(keys[position]);
  return obj[keys[(position + 1) % keys.length]];
}
