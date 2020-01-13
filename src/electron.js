const electron = require("electron");
const app = electron.app;
const path = require("path");
const os = require("os");
var btoa = require("btoa");
const BrowserWindow = electron.BrowserWindow;
const settings = require("electron-settings");
const isDev = require("electron-is-dev");

let mainWindow;
let filePath = "/Users/tracer/Pictures/Ix4oPwv.mp4";

function createWindow() {
  // Initialize React Dev Tools
  isDev &&
    BrowserWindow.addDevToolsExtension(
      path.join(
        os.homedir(),
        "/Library/Application Support/Google/Chrome/Default/Extensions/fmkadmapgofadopljbjfkapdkoienihi/4.3.0_0"
      )
    );
  if (!settings.has("settings")) {
    settings.set("settings", {
      alwaysOnTop: true,
      openFullScreen: true,
      defaultSort: "ALPHA",
      defaultView: "DETAIL",
      defaultFilter: "ALL",
      listScaleMode: "OVERSCAN",
      controlMode: "TRACKPAD",
      scaleMode: "OVERSCAN"
    });
  }
  // Configure new Window options.
  mainWindow = new BrowserWindow({
    name: "Lowkey Image Viewer",
    webPreferences: {
      nodeIntegration: true,
      webSecurity: false
    },
    fullscreen: settings.get("settings.openFullScreen"),
    alwaysOnTop: settings.get("settings.alwaysOnTop"),
    width: 900,
    height: 680
  });

  // Load the react app html in the window.
  mainWindow.loadURL(
    isDev
      ? `http://localhost:3000?${btoa(filePath)}`
      : `file://${path.join(
          __dirname,
          `../build/index.html?${btoa(filePath)}`
        )}`
  );
  mainWindow.on("closed", () => (mainWindow = null));
}

app.on("will-finish-launching", info => {
  console.log("will finish launching");

  // Handle file open event from OS.
  app.on("open-file", (event, newFilePath) => {
    event.preventDefault();
    console.log("on file open fired", newFilePath);
    filePath = newFilePath;
    mainWindow &&
      mainWindow.loadURL(
        `file://${path.join(
          __dirname,
          `../build/index.html?${btoa(filePath)}`
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
