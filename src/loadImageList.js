var memoize = window.require("memoizee");
const readdir = window.require("readdir-enhanced");

const sorts = {
  CREATE_DATE: (a, b) => b.modified - a.modified,
  ALPHA: (a, b) => a.fileName.localeCompare(b.fileName)
};

export default memoize(
  async function loadImageList(
    folderPath,
    filter,
    sortOrder,
    recursive = false
  ) {
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

    return { items: sortedItems };
  },
  { length: 4 }
);
