var memoize = window.require("memoizee");
const readdir = window.require("readdir-enhanced");
const path = window.require("path");
const fs = window.require("fs");

const sorts = {
  CREATE_DATE: (a, b) => b.modified - a.modified,
  ALPHA: (a, b) => a.fileName.localeCompare(b.fileName),
};

export default async function loadImageList(
  folderPath,
  filter,
  sortOrder,
  recursive = false,
  bucket
) {
  if (bucket) {
    const response = await fetch(bucket);
    const images = await response.json();
    console.log("BUCKET IMAGES", images);
    const bucketItems = images
      .filter((item) => path.basename(item.Path)[0] !== ".")
      .map((item) => ({
        fileName: item.Path,
        modified: item.Modified,
      }))
      .sort(sorts[sortOrder.key]);
    return { items: bucketItems };
  }

  console.log("Loading Images from:", folderPath);
  let items, sortedItems;
  // Use readdir-enhanced to get a list of files at a path given the settings passed in.
  try {
    items = await readdir.async(folderPath, {
      filter,
      deep: recursive,
      basePath: folderPath,
      stats: true,
    });
    // Do not include hidden items, then map to object with uri and date.
    // Finally sort by the selected sort key.
    sortedItems = items
      .filter((item) => path.basename(item.path)[0] !== ".")
      .map((item) => ({
        fileName: item.path,
        modified: item.mtimeMs,
      }))
      .sort(sorts[sortOrder.key]);
  } catch (err) {
    items = await readdir.async(folderPath, {
      filter,
      deep: recursive,
      basePath: folderPath,
    });
    // Do not include hidden items, then map to object with uri and date.
    // Finally sort by the selected sort key.
    sortedItems = items
      .filter((item) => path.basename(item)[0] !== ".")
      .map((item) => ({
        fileName: item,
        modified: fs.statSync(item).mtimeMs,
      }))
      .sort(sorts[sortOrder.key]);
  }
  return { items: sortedItems };
}
