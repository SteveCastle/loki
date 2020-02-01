var is = window.require("electron-is");

export const getFolder = filePath => {
  const matchDirectory = is.windows() ? "\\" : "/";

  const folderPath = filePath.substring(
    0,
    filePath.lastIndexOf(matchDirectory)
  );
  return folderPath;
};
