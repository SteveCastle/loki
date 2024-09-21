import path from 'path';
import { Database } from './database';
import type Store from 'electron-store';
import { IpcMainInvokeEvent, dialog } from 'electron';

import { asyncCreateThumbnail } from './image-processing';
import { getFileType } from '../file-types';

const loadTaxonomy = (db: Database) => async () => {
  try {
    const categories = await db.all(
      `SELECT
    c.label AS category_label,
    c.weight AS category_weight,
    json_group_array(
      json_object(
        'label', t.label,
        'category', t.category_label,
        'weight', t.weight
      )
    ) AS tags
  FROM category c
  LEFT JOIN tag t ON c.label = t.category_label
  GROUP BY c.label, c.weight ORDER BY c.weight;`
    );
    const taxonomy = categories.map((category) => {
      return {
        label: category.category_label,
        weight: category.category_weight,
        tags: JSON.parse(category.tags),
      };
    });

    const taxonomyByLabel = taxonomy.reduce((acc, category) => {
      acc[category.label] = category;
      return acc;
    }, {} as { [key: string]: any });
    return taxonomyByLabel;
  } catch (e) {
    console.log(e);
  }
};

type TagInput = [string, string];

const createTag =
  (db: Database) => async (_: IpcMainInvokeEvent, args: TagInput) => {
    const [label, categoryLabel] = args;
    const results = await db.get(`SELECT COUNT(*) AS count FROM tag`);
    const newWeight = results.count + 1;
    await db.run(
      `INSERT INTO tag (label, category_label, weight) VALUES ($1, $2, $3) ON CONFLICT(label) DO NOTHING`,
      [label, categoryLabel, newWeight]
    );
  };

type CategoryInput = [string, number];
const createCategory =
  (db: Database) => async (_: IpcMainInvokeEvent, args: CategoryInput) => {
    const [label, weight] = args;
    await db.run(
      `INSERT INTO category (label, weight) VALUES ($1, $2) ON CONFLICT(label) DO NOTHING`,
      [label, weight]
    );
  };

type AssignmentInput = [string[], string, string, number, boolean];
const createAssignment =
  (db: Database, store: Store) =>
  async (_: IpcMainInvokeEvent, args: AssignmentInput) => {
    // eslint-disable-next-line prefer-const
    let [mediaPaths, tagLabel, categoryLabel, timeStamp, applyTagPreview] =
      args;
    // Open mediaPaths and create thumbnail as a blog to store in the database.
    await db.run('BEGIN TRANSACTION');
    const insertStatement = await db.prepare(
      `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES ($1, $2, $3, $4, $5) ON CONFLICT(media_path, tag_label, category_label, time_stamp) DO NOTHING`
    );
    const results = await db.get(
      `SELECT COUNT(*) AS count FROM media_tag_by_category WHERE tag_label = $2`,
      [tagLabel]
    );
    let newWeight = results.count;

    if (mediaPaths.length > 1 || !timeStamp) {
      timeStamp = 0;
    }

    for (const mediaPath of mediaPaths) {
      try {
        newWeight = newWeight + 1;
        if (getFileType(mediaPath) === 'image' || !timeStamp) {
          timeStamp = 0;
        }
        await insertStatement.run(
          mediaPath,
          tagLabel,
          categoryLabel,
          newWeight,
          timeStamp
        );
      } catch (e) {
        console.log(e);
      }
    }
    await db.run('COMMIT');

    // Save the preview in the database, use the tags table.
    const userHomeDirectory = require('os').homedir();
    const defaultBasePath = path.join(path.join(userHomeDirectory, '.lowkey'));
    const dbPath = store.get('dbPath', defaultBasePath) as string;
    const basePath = path.dirname(dbPath);
    try {
      if (applyTagPreview) {
        const thumbnail_path_600 = await asyncCreateThumbnail(
          mediaPaths[0],
          basePath,
          'thumbnail_path_600',
          timeStamp
        );
        await db.run(
          `UPDATE tag
  SET thumbnail_path_600 = $1
  WHERE label = $2;`,
          [thumbnail_path_600, tagLabel]
        );
      }
    } catch (e) {
      console.log(e);
    }

    //If there was more than one mediaPath wait one second before returning.
    if (mediaPaths.length > 1) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  };

type Tag = {
  tag_label: string;
  time_stamp: number;
};

type DeleteAssignmentInput = [string, Tag];
const deleteAssignment =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: DeleteAssignmentInput) => {
    const [mediaPath, tag] = args;
    const { tag_label: tagLabel, time_stamp: timeStamp } = tag;
    if (timeStamp) {
      await db.run(
        `DELETE FROM media_tag_by_category WHERE media_path = $1 AND tag_label = $2 AND time_stamp = $3`,
        [mediaPath, tagLabel, timeStamp]
      );
      return;
    }
    await db.run(
      `DELETE FROM media_tag_by_category WHERE media_path = $1 AND tag_label = $2`,
      [mediaPath, tagLabel]
    );
  };

type UpdateAssignmentWeightInput = [string, string, number, number];
const updateAssignmentWeight =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: UpdateAssignmentWeightInput) => {
    const [mediaPath, tagLabel, weight, mediaTimeStamp] = args;
    const normalizedMediaPath = path.normalize(mediaPath);
    if (mediaTimeStamp) {
      await db.run(
        `UPDATE media_tag_by_category SET weight = $1 WHERE media_path = $2 AND tag_label = $3 AND time_stamp = $4;`,
        [weight, normalizedMediaPath, tagLabel, mediaTimeStamp]
      );
      return;
    }

    await db.run(
      `UPDATE media_tag_by_category SET weight = $1 WHERE media_path = $2 AND tag_label = $3;`,
      [weight, normalizedMediaPath, tagLabel]
    );
  };

type UpdateTagWeightInput = [string, number];
const updateTagWeight =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: UpdateTagWeightInput) => {
    const [tagLabel, weight] = args;
    const results = await db.run(
      `UPDATE tag SET weight = $1 WHERE label = $2`,
      [weight, tagLabel]
    );
    console.log(results);
  };

type FetchTagPreviewInput = [string];
const fetchTagPreview =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: FetchTagPreviewInput) => {
    const [tagLabel] = args;
    const results = await db.get(
      `SELECT thumbnail_path_600 FROM tag WHERE label = $1`,
      [tagLabel]
    );
    return results?.thumbnail_path_600;
  };

type RenameCategoryInput = [string, string];
const renameCategory =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: RenameCategoryInput) => {
    const [oldCategoryLabel, newCategoryLabel] = args;
    await db.run(`UPDATE category SET label = $1 WHERE label = $2`, [
      newCategoryLabel,
      oldCategoryLabel,
    ]);
    await db.run(
      'UPDATE media_tag_by_category SET category_label = $1 WHERE category_label = $2',
      [newCategoryLabel, oldCategoryLabel]
    );
    await db.run(
      'UPDATE tag SET category_label = $1 WHERE category_label = $2',
      [newCategoryLabel, oldCategoryLabel]
    );
  };
type DeleteCategoryInput = [string];
const deleteCategory =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: DeleteCategoryInput) => {
    const [categoryLabel] = args;
    await db.run(`DELETE FROM category WHERE label = $1`, [categoryLabel]);
    await db.run(
      'DELETE FROM media_tag_by_category WHERE category_label = $1',
      [categoryLabel]
    );
    await db.run('DELETE FROM tag WHERE category_label = $1', [categoryLabel]);
  };

type RenameTagInput = [string, string];
const renameTag =
  (db: Database) => async (_: IpcMainInvokeEvent, args: RenameTagInput) => {
    const [oldTagLabel, newTagLabel] = args;
    await db.run(`UPDATE tag SET label = $1 WHERE label = $2`, [
      newTagLabel,
      oldTagLabel,
    ]);
    await db.run(
      'UPDATE media_tag_by_category SET tag_label = $1 WHERE tag_label = $2',
      [newTagLabel, oldTagLabel]
    );
  };

type MoveTagInput = [string, string];
const moveTag =
  (db: Database) => async (_: IpcMainInvokeEvent, args: MoveTagInput) => {
    const [tagLabel, categoryLabel] = args;
    await db.run(`UPDATE tag SET category_label = $1 WHERE label = $2`, [
      categoryLabel,
      tagLabel,
    ]);
    await db.run(
      'UPDATE media_tag_by_category SET category_label = $1 WHERE tag_label = $2',
      [categoryLabel, tagLabel]
    );
  };

type DeleteTagInput = [string];
const deleteTag =
  (db: Database) => async (_: IpcMainInvokeEvent, args: DeleteTagInput) => {
    const [tagLabel] = args;
    await db.run(`DELETE FROM tag WHERE label = $1`, [tagLabel]);
    await db.run('DELETE FROM media_tag_by_category WHERE tag_label = $1', [
      tagLabel,
    ]);
  };

type Media = {
  path: string;
  timeStamp: number;
};
type LoadTagsByMediaPath = [Media];
const loadTagsByMediaPath =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: LoadTagsByMediaPath) => {
    let tags = [];
    try {
      const [media] = args;
      if (!media.timeStamp) {
        const query = `SELECT * FROM media_tag_by_category WHERE media_path = $1 ORDER BY weight ASC`;
        tags = await db.all(query, [media.path]);
      } else {
        const query = `SELECT * FROM media_tag_by_category WHERE media_path = $1 ORDER BY weight ASC`;
        tags = await db.all(query, [media.path]);
      }

      if (!tags) {
        return null;
      }
      return {
        tags: tags.map((tag) => tag),
      };
    } catch (e) {
      console.log(e);
    }
  };

type SelectNewPathInput = [string, boolean];
const selectNewPath =
  (db: Database, mainWindow: Electron.BrowserWindow | null) =>
  async (_: IpcMainInvokeEvent, args: SelectNewPathInput) => {
    if (!mainWindow) {
      return null;
    }
    const targetPath = args[0];
    const updateAll = args[1];
    const result = await dialog.showOpenDialog(mainWindow, {
      defaultPath: targetPath,
      properties: ['openFile'],
      filters: [
        {
          name: 'Media',
          extensions: [
            'jpg',
            'jpeg',
            'png',
            'gif',
            'bmp',
            'svg',
            'jfif',
            'pjpeg',
            'pjp',
            'webp',
            'mp4',
            'mkv',
            'webm',
            'gltf',
            'glb',
          ],
        },

        {
          name: 'Images',
          extensions: [
            'jpg',
            'jpeg',
            'png',
            'gif',
            'bmp',
            'svg',
            'jfif',
            'pjpeg',
            'pjp',
            'webp',
          ],
        },
        { name: 'Movies', extensions: ['mp4', 'mkv', 'webm'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (!result.canceled) {
      // Update the media_tag_by_category table with the new path.
      const newPath = result.filePaths[0];
      const pathWithoutFile = path.dirname(targetPath);
      const newPathWithoutFile = path.dirname(newPath);

      // If not updating all, only update the media_path, if updating all, update all media_paths that start with the pathWithoutFile
      if (!updateAll) {
        await db.run(
          `UPDATE media_tag_by_category SET media_path = $1 WHERE media_path = $2`,
          [newPath, targetPath]
        );
      } else {
        console.log(
          'Updating all media paths that start with',
          pathWithoutFile
        );
        await db.run(
          `UPDATE media_tag_by_category
          SET media_path = REPLACE(media_path, $1, $2)
          WHERE media_path LIKE $3;`,
          [pathWithoutFile, newPathWithoutFile, pathWithoutFile + '%']
        );
      }

      return { newPath, path: targetPath, updateAll };
    } else {
      return null;
    }
  };

type OrderTagInput = [string];
const orderTags =
  (db: Database) => async (_: IpcMainInvokeEvent, args: OrderTagInput) => {
    const [categoryLabel] = args;
    const tags = await db.all(
      `WITH RankedLabels AS (
          SELECT
            label,
            ROW_NUMBER() OVER (ORDER BY label ASC) AS rank
          FROM
            tag
          WHERE
            category_label = $1
        )

        UPDATE tag
        SET weight = (SELECT rank FROM RankedLabels WHERE tag.label = RankedLabels.label)
        WHERE
          category_label = $1
          AND EXISTS (SELECT 1 FROM RankedLabels WHERE tag.label = RankedLabels.label);
`,
      [categoryLabel]
    );
    return tags;
  };

type ApplyELOInput = [string];
const applyELO =
  (db: Database) => async (_: IpcMainInvokeEvent, args: ApplyELOInput) => {
    const [tagLabel] = args;
    // Update the weights based on the elo rating. Do this by sorting by elo then assigning the index to the weight.
    console.log('Applying ELO to tag', tagLabel);
    await db.run(
      `      WITH RankedElo AS (
          SELECT
          	path,
            elo,
            ROW_NUMBER() OVER (ORDER BY elo DESC) AS rank
          FROM
            media
          WHERE
            tag_label = $1
        )
        UPDATE media_tag_by_category
        SET weight = (SELECT rank FROM RankedElo WHERE media_tag_by_category.media_path = RankedElo.path)
        WHERE
          tag_label = $1
          AND EXISTS (SELECT 1 FROM RankedElo WHERE media_tag_by_category.media_path = RankedElo.path);
;`,
      [tagLabel]
    );
    return;
  };

type ApplyWeightInput = [string];
const applyWeight =
  (db: Database) => async (_: IpcMainInvokeEvent, args: ApplyWeightInput) => {
    const [categoryLabel] = args;
    console.log('Applying weight to category', categoryLabel);
    return;
  };

export {
  loadTaxonomy,
  createTag,
  createAssignment,
  createCategory,
  deleteAssignment,
  updateAssignmentWeight,
  updateTagWeight,
  fetchTagPreview,
  renameCategory,
  deleteCategory,
  renameTag,
  moveTag,
  orderTags,
  applyELO,
  applyWeight,
  deleteTag,
  loadTagsByMediaPath,
  selectNewPath,
};
