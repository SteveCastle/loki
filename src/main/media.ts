import { Database } from './database';
import path from 'path';
import crypto from 'crypto';
const clipboardEx = require('electron-clipboard-ex');
import type Store from 'electron-store';
import { asyncCreateThumbnail } from './image-processing';
import { getFileType } from '../file-types';
import { IpcMainInvokeEvent, shell } from 'electron';
import fs from 'fs';
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

function parseSearchString(search: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < search.length; i++) {
    const c = search[i];

    if (c === '"') {
      // Toggle inQuotes on/off
      inQuotes = !inQuotes;
    } else if (/\s/.test(c) && !inQuotes) {
      // Whitespace outside quotes => new token boundary
      if (current.trim()) {
        tokens.push(current);
      }
      current = '';
    } else {
      // Accumulate character
      current += c;
    }
  }

  // Push the last token if present
  if (current.trim()) {
    tokens.push(current);
  }

  return tokens;
}

const loadMediaByDescriptionSearch =
  (db: Database) => async (_: IpcMainInvokeEvent, args: any[]) => {
    const search = args[0] || '';
    const tags = args[1] || [];
    const filteringMode = args[2] || 'EXCLUSIVE';
    console.log('loadMediaByDescriptionSearch', search, tags, filteringMode);
    // 1) Tokenize: respect quoted phrases
    const rawTokens = parseSearchString(search);

    // 2) Build up WHERE clauses
    const whereClauses: string[] = [];
    const params: string[] = [];

    for (let token of rawTokens) {
      if (!token) continue;

      // Check if token is excluded (leading '-')
      let isExclude = false;
      if (token.startsWith('-')) {
        isExclude = true;
        token = token.slice(1).trim();
      }

      // Identify columns to search
      let columns: Array<'description' | 'path' | 'tag' | 'hash'> = [];

      // Check for column-specific prefixes
      if (token.startsWith('description:')) {
        columns = ['description'];
        token = token.replace(/^description:/, '').trim();
      } else if (token.startsWith('tag:')) {
        columns = ['tag'];
        token = token.replace(/^tag:/, '').trim();
      } else if (token.startsWith('path:')) {
        columns = ['path'];
        token = token.replace(/^path:/, '').trim();
      } else if (token.startsWith('hash:')) {
        columns = ['hash'];
        token = token.replace(/^hash:/, '').trim();
      } else {
        // No prefix => search in all three
        columns = ['description', 'path', 'tag'];
      }

      // If the token ended up empty after stripping prefixes, skip
      if (!token) continue;

      // We'll accumulate sub-expressions for these columns
      const subExprs: string[] = [];
      const likeParam = `%${token}%`;

      for (const col of columns) {
        if (col === 'description') {
          if (isExclude) {
            subExprs.push('(media.description NOT LIKE ?)');
            params.push(likeParam);
          } else {
            subExprs.push('(media.description LIKE ?)');
            params.push(likeParam);
          }
        } else if (col === 'path') {
          if (isExclude) {
            subExprs.push('(media.path NOT LIKE ?)');
            params.push(likeParam);
          } else {
            subExprs.push('(media.path LIKE ?)');
            params.push(likeParam);
          }
        } else if (col === 'hash') {
          if (isExclude) {
            subExprs.push('(media.hash NOT LIKE ?)');
            params.push(likeParam);
          } else {
            subExprs.push('(media.hash LIKE ?)');
            params.push(likeParam);
          }
        } else if (col === 'tag') {
          if (isExclude) {
            // Must NOT have this tag
            subExprs.push(`
            NOT EXISTS (
              SELECT 1 FROM media_tag_by_category mtc
              WHERE mtc.media_path = media.path
                AND mtc.tag_label LIKE ?
            )
          `);
            params.push(likeParam);
          } else {
            // Must have this tag
            subExprs.push(`
            EXISTS (
              SELECT 1 FROM media_tag_by_category mtc
              WHERE mtc.media_path = media.path
                AND mtc.tag_label LIKE ?
            )
          `);
            params.push(likeParam);
          }
        }
      }

      // If multiple columns, for "includes" we OR them; for "excludes" we AND them.
      // e.g. a no-prefix token "foo" => (description LIKE ? OR path LIKE ? OR tag LIKE ?)
      //      an exclude token "-foo" => (description NOT LIKE ? AND path NOT LIKE ? AND NOT EXISTS ... )
      let combined: string;
      if (columns.length > 1) {
        if (isExclude) {
          combined = '(' + subExprs.join(' AND ') + ')';
        } else {
          combined = '(' + subExprs.join(' OR ') + ')';
        }
      } else {
        // Only one sub-expression
        combined = subExprs[0];
      }

      whereClauses.push(combined);
    }

    // 3) Add tag filtering conditions if tags are provided
    if (tags && tags.length > 0) {
      if (filteringMode === 'EXCLUSIVE') {
        // For exclusive mode, media must have ALL specified tags
        for (const tag of tags) {
          whereClauses.push(`
            EXISTS (
              SELECT 1 FROM media_tag_by_category mtc
              WHERE mtc.media_path = media.path
                AND mtc.tag_label = ?
            )
          `);
          params.push(tag);
        }
      } else {
        // For inclusive mode, media must have ANY of the specified tags
        const tagConditions = tags.map(() => `mtc.tag_label = ?`).join(' OR ');
        whereClauses.push(`
          EXISTS (
            SELECT 1 FROM media_tag_by_category mtc
            WHERE mtc.media_path = media.path
              AND (${tagConditions})
          )
        `);
        params.push(...tags);
      }
    }

    // Combine all conditions with AND
    const whereClause = whereClauses.length
      ? `WHERE ${whereClauses.join(' AND ')}`
      : '';

    // Final SQL: we do subqueries for tags instead of an explicit join
    const sql = `
    SELECT DISTINCT media.*
    FROM media
    ${whereClause}
  `;

    try {
      const mediaRows = await db.all(sql, params);

      const library = mediaRows.map((m: any) => ({
        path: m.path,
        weight: m.weight,
      }));

      return { library, cursor: 0 };
    } catch (error) {
      console.error('Error in loadMediaByDescriptionSearch:', error);
      throw error;
    }
  };

const loadMediaByTags =
  (db: Database) => async (_: IpcMainInvokeEvent, args: LoadMediaInput) => {
    const tableName = 'media_tag_by_category';
    let sql = `SELECT mtc.media_path, mtc.tag_label, mtc.category_label, mtc.weight, mtc.time_stamp, mtc.created_at, m.height, m.width FROM ${tableName} mtc left join media m on m.path = mtc.media_path`;
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
        mtimeMs: media.created_at || 0,
        timeStamp: media.time_stamp,
        elo: media.elo,
        tagLabel: media.tag_label,
        height: media.height,
        width: media.width,
      }));
      return { library, cursor: 0 };
    } catch (e) {
      console.log(e);
    }
  };

type FetchMediaPreviewInput = [string, string?, timeStamp?: number];
const fetchMediaPreview =
  (db: Database, store: Store) =>
  async (_: IpcMainInvokeEvent, args: FetchMediaPreviewInput) => {
    const filePath = args[0];
    const cache = args[1] || 'thumbnail_path_600';
    const timeStamp = args[2] || 0;
    const userHomeDirectory = require('os').homedir();
    const defaultBasePath = path.join(path.join(userHomeDirectory, '.lowkey'));
    const dbPath = store.get('dbPath', defaultBasePath) as string;
    const regenerateMediaCache = store.get(
      'regenerateMediaCache',
      false
    ) as boolean;
    // Parts of the thumbnail path. The filename is a sha256 hash of the input path.
    const basePath = path.dirname(dbPath);
    const thumbnailFullPath = getMediaCachePath(
      filePath,
      basePath,
      cache,
      timeStamp
    );
    insertMedia(db, filePath);

    const thumbnailExists = checkIfMediaCacheExists(thumbnailFullPath);
    if (!thumbnailExists || regenerateMediaCache) {
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

type UpdateEloInput = [string, number, string, number];
const updateElo =
  (db: Database) => async (_: IpcMainInvokeEvent, args: UpdateEloInput) => {
    const winningPath = args[0];
    const newWinnerElo = args[1];
    const losingPath = args[2];
    const newLoserElo = args[3];
    // Update the elo in the database if the path isn't already there create it.
    const updateElo = `INSERT INTO media (path, elo) VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET elo = ?`;
    await db.run(updateElo, [winningPath, newWinnerElo, newWinnerElo]);
    await db.run(updateElo, [losingPath, newLoserElo, newLoserElo]);
  };

type UpdateDescriptionInput = [string, string];

const updateDescription =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: UpdateDescriptionInput) => {
    const filePath = args[0];
    const description = args[1];
    db.run('UPDATE media SET description = ? WHERE path = ?', [
      description,
      filePath,
    ]);
  };

type DeleteMediaInput = [string];
const deleteMedia =
  (db: Database) => async (_: IpcMainInvokeEvent, args: DeleteMediaInput) => {
    const filePath = args[0];
    shell
      .trashItem(filePath)
      .then(
        () => {
          console.log('File was moved to the trash');
          db.run('DELETE FROM media WHERE path = ?', [filePath]);
          db.run('DELETE FROM media_tag_by_category WHERE media_path = ?', [
            filePath,
          ]);
        },
        () => {
          console.error('Error deleting file trying unlink:');
          fs.unlinkSync(filePath);
          db.run('DELETE FROM media WHERE path = ?', [filePath]);
          db.run('DELETE FROM media_tag_by_category WHERE media_path = ?', [
            filePath,
          ]);
        }
      )
      .catch(() => {
        console.error('Error deleting file trying unlink:');
        fs.unlinkSync(filePath);
        db.run('DELETE FROM media WHERE path = ?', [filePath]);
        db.run('DELETE FROM media_tag_by_category WHERE media_path = ?', [
          filePath,
        ]);
      });
  };

// Function to calculate file hash
async function calculateFileHash(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const maxBytes = 3 * 1024 * 1024; // 3MB

    // Create a read stream with a limit of 3MB
    const stream = fs.createReadStream(filePath, {
      start: 0,
      end: maxBytes - 1,
    });

    stream.on('data', (data: Buffer) => {
      const view = new Uint8Array(
        data.buffer,
        data.byteOffset,
        data.byteLength
      );
      hash.update(view);
    });
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', (err) => reject(err));
  });
}
export async function insertBulkMedia(
  db: Database,
  filePaths: string[]
): Promise<void> {
  await db.run('BEGIN TRANSACTION');
  const insertStatement = await db.prepare(
    `
    INSERT INTO media (path)
    VALUES (?)
    ON CONFLICT(path)
    DO NOTHING
    `
  );

  for (const filePath of filePaths) {
    await insertStatement.run(filePath);
  }

  await db.run('COMMIT');
}

// Main function
async function insertMedia(db: Database, filePath: string): Promise<void> {
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      console.error('File does not exist:', filePath);
      // Remove path from database
      db.run('DELETE FROM media WHERE path = ?', [filePath]);
      return;
    }

    // Get file stats
    const stats = fs.statSync(filePath);
    const fileSize = stats.size;

    // Calculate file hash
    const fileHash = await calculateFileHash(filePath);

    db.run(
      `
        INSERT INTO media (path, size, hash, views)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(path)
        DO UPDATE SET
          size = excluded.size,
          hash = excluded.hash,
          views = views + 1
        `,
      [filePath, fileSize, fileHash]
    );
  } catch (error) {
    console.error(
      'Error processing file:',
      error instanceof Error ? error.message : error
    );
  }
}

// Finds other media entries that share the same hash as the provided path
// Excludes the provided path itself and ignores null/empty hashes
const loadDuplicatesByPath =
  (db: Database) => async (_: IpcMainInvokeEvent, args: [string]) => {
    const targetPath = args[0];
    try {
      // Ensure the base file exists in DB with a hash; insert/update like fetchMediaPreview does
      await insertMedia(db, targetPath);
      const sql = `
        SELECT m2.path, m2.elo
        FROM media AS m
        JOIN media AS m2 ON m2.hash = m.hash AND m2.path <> m.path
        WHERE m.path = ?
          AND m.hash IS NOT NULL AND m.hash <> ''
        ORDER BY m2.path ASC
      `;
      const rows = await db.all(sql, [targetPath]);
      const library = rows.map((r: any) => ({
        path: r.path,
        elo: r.elo,
        mtimeMs: 0,
      }));
      return { library, cursor: 0 };
    } catch (error) {
      console.error('Error in loadDuplicatesByPath:', error);
      throw error;
    }
  };

// Merge: copy tags from all duplicates into target, then delete the duplicates
type MergeDuplicatesInput = [string];
const mergeDuplicatesByPath =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: MergeDuplicatesInput) => {
    const targetPath = args[0];
    try {
      // Ensure target is inserted/updated to have hash
      await insertMedia(db, targetPath);

      // Find other duplicates by same hash
      const dupSql = `
        SELECT m2.path AS path
        FROM media AS m
        JOIN media AS m2 ON m2.hash = m.hash AND m2.path <> m.path
        WHERE m.path = ?
          AND m.hash IS NOT NULL AND m.hash <> ''
        ORDER BY m2.path ASC
      `;
      const dupRows: { path: string }[] = await db.all(dupSql, [targetPath]);
      const duplicatePaths = dupRows.map((r) => r.path);

      if (duplicatePaths.length === 0) {
        return { mergedInto: targetPath, deleted: [], copiedTags: 0 };
      }

      // Copy tags from each duplicate into target
      let copiedTags = 0;
      await db.run('BEGIN TRANSACTION');
      const insertTagStmt = await db.prepare(
        `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp, created_at)
         SELECT $1, tag_label, category_label, weight, time_stamp, created_at
         FROM media_tag_by_category
         WHERE media_path = $2
         ON CONFLICT(media_path, tag_label, category_label, time_stamp) DO NOTHING`
      );

      for (const dupPath of duplicatePaths) {
        try {
          // Insert-select returns no rowcount here; optionally count source rows
          const countRow = await db.get(
            `SELECT COUNT(*) AS cnt FROM media_tag_by_category WHERE media_path = $1`,
            [dupPath]
          );
          await insertTagStmt.run(targetPath, dupPath);
          copiedTags += Number(countRow?.cnt || 0);
        } catch (e) {
          console.error('Error copying tags from', dupPath, e);
        }
      }
      await db.run('COMMIT');

      // After copying, delete the duplicates (file + db rows)
      const deleted: string[] = [];
      for (const dupPath of duplicatePaths) {
        try {
          await shell
            .trashItem(dupPath)
            .then(
              async () => {
                await db.run('DELETE FROM media WHERE path = ?', [dupPath]);
                await db.run(
                  'DELETE FROM media_tag_by_category WHERE media_path = ?',
                  [dupPath]
                );
                deleted.push(dupPath);
              },
              async () => {
                console.error('Error deleting file trying unlink:', dupPath);
                fs.unlinkSync(dupPath);
                await db.run('DELETE FROM media WHERE path = ?', [dupPath]);
                await db.run(
                  'DELETE FROM media_tag_by_category WHERE media_path = ?',
                  [dupPath]
                );
                deleted.push(dupPath);
              }
            )
            .catch(async () => {
              console.error('Error deleting file trying unlink:', dupPath);
              try {
                fs.unlinkSync(dupPath);
              } catch (_) {}
              await db.run('DELETE FROM media WHERE path = ?', [dupPath]);
              await db.run(
                'DELETE FROM media_tag_by_category WHERE media_path = ?',
                [dupPath]
              );
              deleted.push(dupPath);
            });
        } catch (e) {
          console.error('Failed to delete duplicate', dupPath, e);
        }
      }

      return { mergedInto: targetPath, deleted, copiedTags };
    } catch (error) {
      console.error('Error in mergeDuplicatesByPath:', error);
      throw error;
    }
  };

export {
  loadMediaByTags,
  loadMediaByDescriptionSearch,
  fetchMediaPreview,
  copyFileIntoClipboard,
  deleteMedia,
  updateElo,
  updateDescription,
  loadDuplicatesByPath,
  mergeDuplicatesByPath,
};

// New helpers to manage thumbnails for a given media path
type ListThumbnailsInput = [string];
type ThumbnailInfo = {
  cache: 'thumbnail_path_100' | 'thumbnail_path_600' | 'thumbnail_path_1200';
  path: string;
  exists: boolean;
  size: number;
};

const caches: ThumbnailInfo['cache'][] = [
  'thumbnail_path_100',
  'thumbnail_path_600',
  'thumbnail_path_1200',
];

const listThumbnails =
  (store: Store) =>
  async (_: IpcMainInvokeEvent, args: ListThumbnailsInput) => {
    const filePath = args[0];
    const userHomeDirectory = require('os').homedir();
    const defaultBasePath = path.join(path.join(userHomeDirectory, '.lowkey'));
    const dbPath = (store.get('dbPath', defaultBasePath) as string) || '';
    const basePath = path.dirname(dbPath);

    const timeStamp = 0; // images ignore timestamp hashing; videos use it

    const results: ThumbnailInfo[] = caches.map((cache) => {
      const thumbnailFullPath = getMediaCachePath(
        filePath,
        basePath,
        cache,
        timeStamp
      );
      const exists = checkIfMediaCacheExists(thumbnailFullPath);
      let size = 0;
      try {
        if (exists) {
          const stat = fs.statSync(thumbnailFullPath);
          size = stat.size || 0;
        }
      } catch (_) {}
      return { cache, path: thumbnailFullPath, exists, size };
    });

    return results;
  };

type RegenerateThumbnailInput = [string, ThumbnailInfo['cache'], number?];
const regenerateThumbnail =
  (store: Store) =>
  async (_: IpcMainInvokeEvent, args: RegenerateThumbnailInput) => {
    const [filePath, cache, timeStampArg] = args;
    const timeStamp = typeof timeStampArg === 'number' ? timeStampArg : 0;

    const userHomeDirectory = require('os').homedir();
    const defaultBasePath = path.join(path.join(userHomeDirectory, '.lowkey'));
    const dbPath = (store.get('dbPath', defaultBasePath) as string) || '';
    const basePath = path.dirname(dbPath);

    const thumbnailFullPath = getMediaCachePath(
      filePath,
      basePath,
      cache,
      timeStamp
    );
    // Delete existing file if it exists to force regeneration
    try {
      if (checkIfMediaCacheExists(thumbnailFullPath)) {
        fs.unlinkSync(thumbnailFullPath);
      }
    } catch (e) {
      console.warn('Could not delete existing thumbnail', thumbnailFullPath, e);
    }

    await asyncCreateThumbnail(filePath, basePath, cache, timeStamp);
    return thumbnailFullPath;
  };

export { listThumbnails, regenerateThumbnail };
