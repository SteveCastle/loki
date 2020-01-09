const electron = require("electron");
const app = electron.app;
const path = require("path");
const os = require("os");
var btoa = require("btoa");
const BrowserWindow = electron.BrowserWindow;

const isDev = require("electron-is-dev");

let mainWindow;
let filePath = "/Users/tracer/Pictures/Astronomy/1000863413l.jpg";

function createWindow() {
  // Initialize React Dev Tools
  BrowserWindow.addDevToolsExtension(
    path.join(
      os.homedir(),
      "/Library/Application Support/Google/Chrome/Default/Extensions/fmkadmapgofadopljbjfkapdkoienihi/4.3.0_0"
    )
  );

  // Configure new Window options.
  mainWindow = new BrowserWindow({
    webPreferences: {
      nodeIntegration: true,
      webSecurity: false
    },
    fullscreen: true,
    alwaysOnTop: true,
    width: 900,
    height: 680
  });

  // Load the react app html in the window.
  mainWindow.loadURL(
    isDev
      ? `http://localhost:3000/${btoa(filePath)}`
      : `file://${path.join(__dirname, "../build/index.html", btoa(filePath))}`
  );
  mainWindow.on("closed", () => (mainWindow = null));
}

app.on("will-finish-launching", info => {
  console.log("will finish launching");

  // Handle file open event from OS.
  app.on("open-file", (event, path) => {
    event.preventDefault();
    console.log("on file open fired", path);
    filePath = path;
    mainWindow &&
      mainWindow.loadURL(`file://${__dirname}/app.html#/${btoa(filePath)}`);
  });
});

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});
