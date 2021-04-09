var memoize = window.require("memoizee");
const readdir = window.require("readdir-enhanced");
const path = window.require("path");

const sorts = {
  CREATE_DATE: (a, b) => b.modified - a.modified,
  ALPHA: (a, b) => a.fileName.localeCompare(b.fileName),
};

export default async function loadImageList(
  folderPath,
  filter,
  sortOrder,
  recursive = false
) {
  console.log(folderPath);
  // Use readdir-enhanced to get a list of files at a path given the settings passed in.
  let items = await readdir.async(folderPath, {
    filter,
    deep: recursive,
    basePath: folderPath,
    stats: true,
  });

  // Do not include hidden items, then map to object with uri and date.
  // Finally sort by the selected sort key.
  let sortedItems = items
    .filter((item) => path.basename(item.path)[0] !== ".")
    .map((item) => ({
      fileName: item.path,
      modified: item.mtimeMs,
    }))
    .sort(sorts[sortOrder]);

  return { items: sortedItems };
}
