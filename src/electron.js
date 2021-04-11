const electron = require("electron");
const app = electron.app;
const path = require("path");
const { encode } = require("js-base64");
const BrowserWindow = electron.BrowserWindow;
const settings = require("electron-settings");
const isDev = require("electron-is-dev");
var is = require("electron-is");

const { MenuBuilder } = require("./menu");

const HOT_KEY_DEFAULTS = {
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

let mainWindow;
let filePath = "";

function createWindow() {
  //   If this is windows get the filePath to use from argv.
  if (process.argv.length >= 2 && process.argv[1] && is.windows()) {
    filePath = process.argv[1];
    //open, read, handle file
  }
  if (!settings.has("settings")) {
    settings.set("settings", {
      alwaysOnTop: true,
      openFullScreen: true,
      defaultSort: "ALPHA",
      defaultView: "DETAIL",
      defaultFilter: "ALL",
      listScaleMode: "OVERSCAN",
      controlMode: "TRACK_PAD",
      scaleMode: "OVERSCAN",
      starts: 0,
    });
  }

  if (!settings.has("settings.hotKeys")) {
    settings.set("settings.hotKeys", HOT_KEY_DEFAULTS);
  }

  settings.set("settings.starts", settings.get("settings.starts") + 1);
  settings.set("settings.lastStart", new Date());

  // Configure new Window options.
  const options = {
    name: "Lowkey Image Viewer",
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      webSecurity: false,
    },
    frame: false,
    alwaysOnTop: settings.get("settings.alwaysOnTop"),
    width: 900,
    height: 680,
  };

  if (settings.get("settings.openFullScreen")) {
    options.fullscreen = true;
  }
  mainWindow = new BrowserWindow(options);

  // Load the react app html in the window.
  mainWindow.loadURL(
    isDev
      ? `http://localhost:3000?${encode(filePath)}`
      : `file://${path.join(
          __dirname,
          `../build/index.html?${encode(filePath)}`
        )}`
  );
  mainWindow.on("closed", () => (mainWindow = null));

  // Register Menu
  const menuBuilder = new MenuBuilder(mainWindow);
  menuBuilder.buildMenu();
}

app.on("will-finish-launching", (info) => {
  // Handle file open event from OS.
  app.on("open-file", (event, newFilePath) => {
    event.preventDefault();
    filePath = newFilePath;
    mainWindow &&
      mainWindow.loadURL(
        `file://${path.join(
          __dirname,
          `../build/index.html?${encode(filePath)}`
        )}`
      );
  });
});

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  app.quit();
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});
