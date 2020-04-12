var is = window.require("electron-is");
const settings = window.require("electron-settings");

export const getFolder = (filePath) => {
  const matchDirectory = is.windows() ? "\\" : "/";

  const folderPath = filePath.substring(
    0,
    filePath.lastIndexOf(matchDirectory)
  );
  return folderPath;
};

export function saveCurrentSettings({
  controlMode,
  defaultSort,
  scaleMode,
  defaultFilter,
  isAlwaysOnTop,
  isFullScreen,
}) {
  if (controlMode) {
    settings.set("settings.controlMode", controlMode);
  }
  if (defaultSort) {
    settings.set("settings.defaultSort", defaultSort);
  }
  if (scaleMode) {
    settings.set("settings.scaleMode", scaleMode);
  }
  if (defaultFilter) {
    settings.set("settings.defaultFilter", defaultFilter);
  }
  settings.set("settings.alwaysOnTop", isAlwaysOnTop);

  settings.set("settings.openFullScreen", isFullScreen);
}
