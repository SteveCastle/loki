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

export const getFile = (filePath) => {
  const matchDirectory = is.windows() ? "\\" : "/";

  const folderPath = filePath.substring(
    filePath.lastIndexOf(matchDirectory) + 1
  );
  return folderPath;
};

export function saveCurrentSettings({
  controlMode,
  defaultSort,
  scaleMode,
  defaultFilter,
  audio,
  videoControls,
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
  if (audio != null) {
    settings.set("settings.audio", audio);
  }
  if (videoControls != null) {
    settings.set("settings.videoControls", videoControls);
  }
  if (defaultFilter) {
    settings.set("settings.defaultFilter", defaultFilter);
  }
  settings.set("settings.alwaysOnTop", isAlwaysOnTop);

  settings.set("settings.openFullScreen", isFullScreen);
}
