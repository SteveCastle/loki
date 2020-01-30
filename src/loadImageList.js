import { getFolder } from "./fsTools";

const readdir = window.require("readdir-enhanced");

const sorts = {
  CREATE_DATE: (a, b) => b.modified - a.modified,
  ALPHA: (a, b) => a.fileName.localeCompare(b.fileName)
};

export default async function loadImageList({
  filePath,
  filter,
  sortOrder,
  recursive = false
}) {
  const folderPath = getFolder(filePath);
  console.log(filePath, folderPath);
  let items = await readdir.async(folderPath, {
    filter,
    deep: recursive,
    basePath: folderPath,
    stats: true
  });

  let sortedItems = items
    .map(item => ({
      fileName: item.path,
      modified: item.mtimeMs
    }))
    .sort(sorts[sortOrder]);
  // Get the position of the initial item in the results, unless its not found, then return 0;
  let cursor = sortedItems.findIndex(item => {
    return item.fileName === filePath;
  });
  if (cursor < 0) {
    cursor = 0;
  }
  return { items: sortedItems, cursor };
}
