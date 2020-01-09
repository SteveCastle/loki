import { Promise } from "bluebird";
const readdir = window.require("readdir");
const fs = window.require("fs");

const sorts = {
  CREATE_DATE: (a, b) => b.modified - a.modified,
  ALPHA: (a, b) => a.fileName.localeCompare(b.fileName)
};

const asyncRead = Promise.promisify(readdir.read);

export default async function loadImageList({
  path,
  filter,
  sortOrder,
  recursive = false
}) {
  const folderPath = path.substring(0, path.lastIndexOf("/"));
  let items = await asyncRead(
    folderPath,
    filter,
    readdir.ABSOLUTE_PATHS +
      readdir.INCLUDE_DIRECTORIES +
      (!recursive && readdir.NON_RECURSIVE)
  );
  let sortedItems = items
    .map(item => ({
      fileName: item,
      modified: fs.statSync(item).mtime.getTime()
    }))
    .sort(sorts[sortOrder]);
  return sortedItems;
}
