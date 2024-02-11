import { Database } from './database';
import path from 'path';
import crypto from 'crypto';
const clipboardEx = require('electron-clipboard-ex');
import type Store from 'electron-store';
import { asyncCreateThumbnail } from './image-processing';
import { getFileType } from '../file-types';
import { IpcMainInvokeEvent } from 'electron';
type LoadMediaInput = [string[], string];

function createHash(input: string) {
  const hash = crypto.createHash('sha256');
  hash.update(input);
  return hash.digest('hex');
}

function getMediaCachePath(
  mediaPath: string,
  basePath: string,
  cache: string,
  timeStamp: number
) {
  // Parts of the thumbnail path. The filename is a sha256 hash of the input path.
  const thumbnailBasePath = path.join(basePath, cache);
  const thumbnailFileName = createHash(
    mediaPath + (timeStamp > 0 ? timeStamp.toString() : '')
  );
  const thumbnailFullPath = thumbnailBasePath + '/' + thumbnailFileName;
  const isVideo = getFileType(mediaPath, true) === 'video';
  if (isVideo) {
    return thumbnailFullPath + '.mp4';
  }
  return thumbnailFullPath;
}

function checkIfMediaCacheExists(cachePath: string) {
  return require('fs').existsSync(cachePath);
}

const loadMediaByTags =
  (db: Database) => async (_: IpcMainInvokeEvent, args: LoadMediaInput) => {
    const tableName = 'media_tag_by_category';
    let sql = `SELECT * FROM ${tableName}`;
    const tags = args[0];
    const mode = args[1];
    const params: string[] = [];
    const conditions = tags.map((tag: string) => {
      return {
        column: 'tag_label',
        operator: '=',
        value: tag,
      };
    });

    if (conditions && conditions.length > 0) {
      sql += ' WHERE';

      // Generate the WHERE clause dynamically
      conditions.forEach((condition: any, index: number) => {
        const { column, operator, value } = condition;
        sql += ` ${column} ${operator} $${index + 1}`;
        params.push(value);

        if (index !== conditions.length - 1) {
          sql += ' OR';
        }
      });
    }
    sql += ` GROUP BY media_path`;
    if (conditions.length <= 1) {
      sql += `, time_stamp`;
    }
    sql += ` HAVING COUNT(DISTINCT tag_label) = ${
      mode === 'AND' ? conditions.length : 1
    }`;
    sql += ` ORDER BY weight;`;
    console.log('SQL', sql, params);
    try {
      const media = await db.all(sql, params);

      const library = media.map((media) => ({
        path: media.media_path,
        weight: media.weight,
        mtimeMs: 0,
        timeStamp: media.time_stamp,
        tagLabel: media.tag_label,
      }));
      return { library, cursor: 0 };
    } catch (e) {
      console.log(e);
    }
  };

type FetchMediaPreviewInput = [string, string?, timeStamp?: number];
const fetchMediaPreview =
  (store: Store) =>
  async (_: IpcMainInvokeEvent, args: FetchMediaPreviewInput) => {
    const filePath = args[0];
    const cache = args[1] || 'thumbnail_path_600';
    const timeStamp = args[2] || 0;
    const userHomeDirectory = require('os').homedir();
    const defaultBasePath = path.join(path.join(userHomeDirectory, '.lowkey'));
    const dbPath = store.get('dbPath', defaultBasePath) as string;

    // Parts of the thumbnail path. The filename is a sha256 hash of the input path.
    const basePath = path.dirname(dbPath);
    const thumbnailFullPath = getMediaCachePath(
      filePath,
      basePath,
      cache,
      timeStamp
    );
    const thumbnailExists = checkIfMediaCacheExists(thumbnailFullPath);
    if (!thumbnailExists) {
      await asyncCreateThumbnail(filePath, basePath, cache, timeStamp);
    }
    return thumbnailFullPath;
  };

type CopyFileIntoClipboardInput = [string];
const copyFileIntoClipboard =
  () => async (_: IpcMainInvokeEvent, args: CopyFileIntoClipboardInput) => {
    const filePaths = args[0];
    console.log('copying file into clipboard', filePaths);
    // Copies the file into the clipboard
    clipboardEx.writeFilePaths(filePaths);
    console.log('copied file into clipboard');
  };

export { loadMediaByTags, fetchMediaPreview, copyFileIntoClipboard };
