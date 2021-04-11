const settings = window.require("electron-settings");

const commands = {
  fileOptions: {
    changeFile: {
      action: () => {
        console.log("change file");
      },
    },
    toggleRecursion: {
      action: () => {
        console.log("toggle recursion");
      },
    },
    shuffle: { action: () => {} },
  },
  listOptions: {
    toggleSortOrder: { action: () => {} },
    showAll: { action: () => {} },
    showStatic: { action: () => {} },
    showVideos: { action: () => {} },
    showGifs: { action: () => {} },
    showMotion: { action: () => {} },
  },
  imageOptions: {
    toggleSizing: { action: () => {} },
    sizeOverscan: { action: () => {} },
    sizeActual: { action: () => {} },
    sizeFit: { action: () => {} },
    sizeCover: { action: () => {} },
    toggleAudio: { action: () => {} },
    toggleVideoControls: { action: () => {} },
  },
  controlOptions: { toggleControls: { action: () => {} } },
  windowOptions: {
    toggleFullscreen: { action: () => {} },
    toggleAlwaysOnTop: { action: () => {} },
  },
};

const hotKeyDefaults = {
  s: "fileOptions.changeFile",
  r: "fileOptions.toggleRecursion",
};

if (!settings.has("settings.hotKeys")) {
  settings.set("settings.hotKeys", hotKeyDefaults);
}

const hotKeys = settings.get("settings.hotKeys");
export { hotKeys, commands };
