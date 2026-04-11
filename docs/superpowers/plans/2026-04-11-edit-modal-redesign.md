# Edit Tag / Category Modal Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the edit tag and edit category modals with a cleaner layout, description fields, and extensible action rows (Apply ELO ordering, Consolidate files).

**Architecture:** Add `description` columns to the `tag` and `category` database tables. Update the `loadTaxonomy` query to include descriptions. Redesign both modal components with a structured layout (properties section + actions section + footer). Add new IPC handlers for description updates, ELO ordering, and file consolidation. Use the existing toast system for action feedback.

**Tech Stack:** Electron, React, TypeScript, SQLite (via `sqlite` npm package), XState, React Query

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/main/database.ts` | Modify | Add migration for `description` column on `tag` and `category` tables |
| `src/main/taxonomy.ts` | Modify | Update `loadTaxonomy` query to include descriptions; add `updateTagDescription`, `updateCategoryDescription`, `consolidateTagFiles`, `consolidateCategoryFiles`, `applyEloOrdering` handlers |
| `src/main/main.ts` | Modify | Register new IPC handlers |
| `src/main/preload.ts` | Modify | Add new channel names to `Channels` type |
| `src/renderer/components/taxonomy/taxonomy.tsx` | Modify | Update `Concept` and `Category` types to include `description`; pass description to modals |
| `src/renderer/components/taxonomy/new-tag-modal.tsx` | Modify | Redesign with properties section, description field, action rows, footer |
| `src/renderer/components/taxonomy/new-category-modal.tsx` | Modify | Redesign with properties section, description field, action rows, footer |
| `src/renderer/components/taxonomy/new-modal.css` | Modify | New layout styles for wider modal, sections, action rows, footer |

---

### Task 1: Database Migration — Add description columns

**Files:**
- Modify: `src/main/database.ts:172-192`

- [ ] **Step 1: Add tag description migration**

In `src/main/database.ts`, find the existing tag table migration block (line 172-192). Add `description` to the `columnsToMigrate` array:

```typescript
    const columnsToMigrate = [
      { name: 'preview', type: 'BLOB' },
      { name: 'thumbnail_path_600', type: 'INTEGER' },
      { name: 'description', type: 'TEXT' },
    ];
```

- [ ] **Step 2: Add category description migration**

After the tag migration block (after line 192), add a new migration block for the category table. Follow the same pattern used for tag and media migrations:

```typescript
  // Migrate existing category table if needed
  const categoryTable = await db.get(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='category'`
  );
  if (categoryTable) {
    const tableInfo = await db.all(`PRAGMA table_info(category)`);
    const columnsToMigrate = [
      { name: 'description', type: 'TEXT' },
    ];
    for (const column of columnsToMigrate) {
      const columnExists = tableInfo.some(
        (tableColumn: any) => tableColumn.name === column.name
      );
      if (!columnExists) {
        await db.run(
          `ALTER TABLE category ADD COLUMN ${column.name} ${column.type}`
        );
      }
    }
  }
```

- [ ] **Step 3: Verify migration runs**

Run: `npm start` (or the project's dev command), open the app, and confirm no errors in the console. The migration is idempotent so it's safe to run repeatedly.

- [ ] **Step 4: Commit**

```bash
git add src/main/database.ts
git commit -m "feat: add description column migration for tag and category tables"
```

---

### Task 2: Update loadTaxonomy query to include descriptions

**Files:**
- Modify: `src/main/taxonomy.ts:9-42`
- Modify: `src/renderer/components/taxonomy/taxonomy.tsx:25-38`

- [ ] **Step 1: Update the SQL query in loadTaxonomy**

In `src/main/taxonomy.ts`, update the `loadTaxonomy` function's SQL query (lines 11-24) to include description fields:

```typescript
const loadTaxonomy = (db: Database) => async () => {
  try {
    const categories = await db.all(
      `SELECT
    c.label AS category_label,
    c.weight AS category_weight,
    c.description AS category_description,
    json_group_array(
      json_object(
        'label', t.label,
        'category', t.category_label,
        'weight', t.weight,
        'description', t.description
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
        description: category.category_description || '',
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
```

- [ ] **Step 2: Update the Concept and Category types in taxonomy.tsx**

In `src/renderer/components/taxonomy/taxonomy.tsx`, update the type definitions (lines 25-38):

```typescript
type Concept = {
  label: string;
  category: string;
  weight: number;
  description: string;
};

type Category = {
  label: string;
  tags: Concept[];
  description: string;
};

type Taxonomy = {
  [key: string]: Category;
};
```

- [ ] **Step 3: Verify data loads correctly**

Run the app. Open dev tools console. Confirm taxonomy data now includes `description` fields (they will be empty strings or null for existing data).

- [ ] **Step 4: Commit**

```bash
git add src/main/taxonomy.ts src/renderer/components/taxonomy/taxonomy.tsx
git commit -m "feat: include description in taxonomy query and types"
```

---

### Task 3: Add description update IPC handlers

**Files:**
- Modify: `src/main/taxonomy.ts:479-499` (exports)
- Modify: `src/main/main.ts:309` (handler registration area)
- Modify: `src/main/preload.ts:15-62` (Channels type)

- [ ] **Step 1: Add updateTagDescription handler in taxonomy.ts**

Add before the `export` block at the bottom of `src/main/taxonomy.ts` (before line 479):

```typescript
type UpdateTagDescriptionInput = [string, string];
const updateTagDescription =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: UpdateTagDescriptionInput) => {
    const [label, description] = args;
    await db.run(`UPDATE tag SET description = $1 WHERE label = $2`, [
      description,
      label,
    ]);
  };

type UpdateCategoryDescriptionInput = [string, string];
const updateCategoryDescription =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: UpdateCategoryDescriptionInput) => {
    const [label, description] = args;
    await db.run(`UPDATE category SET description = $1 WHERE label = $2`, [
      description,
      label,
    ]);
  };
```

- [ ] **Step 2: Export the new handlers**

In the `export` block at the bottom of `src/main/taxonomy.ts`, add the new handlers:

```typescript
export {
  loadTaxonomy,
  getTagCount,
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
  deleteTag,
  loadTagsByMediaPath,
  selectNewPath,
  updateTimestamp,
  removeTimestamp,
  updateTagDescription,
  updateCategoryDescription,
};
```

- [ ] **Step 3: Register handlers in main.ts**

In `src/main/main.ts`, find the taxonomy handler registration area (around line 344 where `rename-tag` is registered). Add after the existing taxonomy handlers:

```typescript
  ipcMain.handle(
    'update-tag-description',
    taxonomyModule.updateTagDescription(db)
  );
  ipcMain.handle(
    'update-category-description',
    taxonomyModule.updateCategoryDescription(db)
  );
```

Also add the corresponding `removeHandler` calls in the cleanup section (around line 238):

```typescript
  ipcMain.removeHandler('update-tag-description');
  ipcMain.removeHandler('update-category-description');
```

- [ ] **Step 4: Add channel names to preload.ts**

In `src/main/preload.ts`, add the new channel names to the `Channels` type (after `'order-tags'` on line 46):

```typescript
  | 'update-tag-description'
  | 'update-category-description'
```

- [ ] **Step 5: Commit**

```bash
git add src/main/taxonomy.ts src/main/main.ts src/main/preload.ts
git commit -m "feat: add IPC handlers for tag and category description updates"
```

---

### Task 4: Add applyEloOrdering IPC handler

**Files:**
- Modify: `src/main/taxonomy.ts`
- Modify: `src/main/main.ts`
- Modify: `src/main/preload.ts`

- [ ] **Step 1: Add applyEloOrdering handler in taxonomy.ts**

Add before the `export` block in `src/main/taxonomy.ts`:

```typescript
type ApplyEloOrderingInput = [string];
const applyEloOrdering =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: ApplyEloOrderingInput) => {
    const [tagLabel] = args;

    // Get all media items for this tag, sorted by ELO descending
    const mediaItems = await db.all(
      `SELECT m.path, m.elo
       FROM media m
       JOIN media_tag_by_category mtc ON m.path = mtc.media_path
       WHERE mtc.tag_label = $1
       ORDER BY COALESCE(m.elo, -1) DESC`,
      [tagLabel]
    );

    // Assign incrementing weights: highest ELO gets highest weight
    for (let i = 0; i < mediaItems.length; i++) {
      const weight = mediaItems.length - i;
      await db.run(
        `UPDATE media_tag_by_category SET weight = $1 WHERE media_path = $2 AND tag_label = $3`,
        [weight, mediaItems[i].path, tagLabel]
      );
    }

    return { count: mediaItems.length };
  };
```

- [ ] **Step 2: Export the handler**

Add `applyEloOrdering` to the export block in `src/main/taxonomy.ts`.

- [ ] **Step 3: Register handler in main.ts**

In `src/main/main.ts`, add with the other taxonomy handlers:

```typescript
  ipcMain.handle('apply-elo-ordering', taxonomyModule.applyEloOrdering(db));
```

And in the cleanup section:

```typescript
  ipcMain.removeHandler('apply-elo-ordering');
```

- [ ] **Step 4: Add channel name to preload.ts**

Add to the `Channels` type in `src/main/preload.ts`:

```typescript
  | 'apply-elo-ordering'
```

- [ ] **Step 5: Commit**

```bash
git add src/main/taxonomy.ts src/main/main.ts src/main/preload.ts
git commit -m "feat: add apply-elo-ordering IPC handler"
```

---

### Task 5: Add consolidateTagFiles and consolidateCategoryFiles IPC handlers

**Files:**
- Modify: `src/main/taxonomy.ts`
- Modify: `src/main/main.ts`
- Modify: `src/main/preload.ts`

- [ ] **Step 1: Add consolidateTagFiles handler in taxonomy.ts**

Add the `fs` and `path` imports at the top of `src/main/taxonomy.ts` if not already present:

```typescript
import * as fs from 'fs';
import * as path from 'path';
```

Add before the `export` block:

```typescript
type ConsolidateTagFilesInput = [string, string];
const consolidateTagFiles =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: ConsolidateTagFilesInput) => {
    const [tagLabel, targetDir] = args;

    const mediaItems = await db.all(
      `SELECT DISTINCT m.path
       FROM media m
       JOIN media_tag_by_category mtc ON m.path = mtc.media_path
       WHERE mtc.tag_label = $1`,
      [tagLabel]
    );

    let copied = 0;
    let errors = 0;

    for (const item of mediaItems) {
      const sourcePath = item.path;
      let fileName = path.basename(sourcePath);
      let destPath = path.join(targetDir, fileName);

      // Handle filename collisions with numeric suffix
      let counter = 1;
      const ext = path.extname(fileName);
      const base = path.basename(fileName, ext);
      while (fs.existsSync(destPath) && destPath !== sourcePath) {
        fileName = `${base}_${counter}${ext}`;
        destPath = path.join(targetDir, fileName);
        counter++;
      }

      // Skip if source and destination are the same
      if (destPath === sourcePath) {
        continue;
      }

      try {
        fs.copyFileSync(sourcePath, destPath);

        // Update media path
        await db.run(`UPDATE media SET path = $1 WHERE path = $2`, [
          destPath,
          sourcePath,
        ]);

        // Update all media_tag_by_category references
        await db.run(
          `UPDATE media_tag_by_category SET media_path = $1 WHERE media_path = $2`,
          [destPath, sourcePath]
        );

        copied++;
      } catch (e) {
        console.error(`Failed to copy ${sourcePath} to ${destPath}:`, e);
        errors++;
      }
    }

    return { copied, errors, total: mediaItems.length };
  };
```

- [ ] **Step 2: Add consolidateCategoryFiles handler**

Add before the `export` block:

```typescript
type ConsolidateCategoryFilesInput = [string, string];
const consolidateCategoryFiles =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: ConsolidateCategoryFilesInput) => {
    const [categoryLabel, targetDir] = args;

    const mediaItems = await db.all(
      `SELECT DISTINCT m.path
       FROM media m
       JOIN media_tag_by_category mtc ON m.path = mtc.media_path
       WHERE mtc.category_label = $1`,
      [categoryLabel]
    );

    let copied = 0;
    let errors = 0;

    for (const item of mediaItems) {
      const sourcePath = item.path;
      let fileName = path.basename(sourcePath);
      let destPath = path.join(targetDir, fileName);

      let counter = 1;
      const ext = path.extname(fileName);
      const base = path.basename(fileName, ext);
      while (fs.existsSync(destPath) && destPath !== sourcePath) {
        fileName = `${base}_${counter}${ext}`;
        destPath = path.join(targetDir, fileName);
        counter++;
      }

      if (destPath === sourcePath) {
        continue;
      }

      try {
        fs.copyFileSync(sourcePath, destPath);

        await db.run(`UPDATE media SET path = $1 WHERE path = $2`, [
          destPath,
          sourcePath,
        ]);

        await db.run(
          `UPDATE media_tag_by_category SET media_path = $1 WHERE media_path = $2`,
          [destPath, sourcePath]
        );

        copied++;
      } catch (e) {
        console.error(`Failed to copy ${sourcePath} to ${destPath}:`, e);
        errors++;
      }
    }

    return { copied, errors, total: mediaItems.length };
  };
```

- [ ] **Step 3: Export both handlers**

Add `consolidateTagFiles` and `consolidateCategoryFiles` to the export block.

- [ ] **Step 4: Register handlers in main.ts**

```typescript
  ipcMain.handle(
    'consolidate-tag-files',
    taxonomyModule.consolidateTagFiles(db)
  );
  ipcMain.handle(
    'consolidate-category-files',
    taxonomyModule.consolidateCategoryFiles(db)
  );
```

And cleanup:

```typescript
  ipcMain.removeHandler('consolidate-tag-files');
  ipcMain.removeHandler('consolidate-category-files');
```

- [ ] **Step 5: Add channel names to preload.ts**

```typescript
  | 'consolidate-tag-files'
  | 'consolidate-category-files'
```

- [ ] **Step 6: Commit**

```bash
git add src/main/taxonomy.ts src/main/main.ts src/main/preload.ts
git commit -m "feat: add consolidate files IPC handlers for tags and categories"
```

---

### Task 6: Update modal CSS for new layout

**Files:**
- Modify: `src/renderer/components/taxonomy/new-modal.css`

- [ ] **Step 1: Replace the modal CSS**

Replace the entire contents of `src/renderer/components/taxonomy/new-modal.css`:

```css
.input-modal {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.input-modal-content {
  width: 550px;
  background-color: var(--controls-background-color);
  border-radius: 5px;
}

.input-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.input-modal-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  font-weight: 900;
  letter-spacing: 0.5px;
}

.input-modal-close {
  cursor: pointer;
  opacity: 0.8;
  transition: opacity 0.2s ease-in-out;
}

.input-modal-close:hover {
  opacity: 1;
}

/* Properties section (name + description) */
.input-modal-properties {
  padding: 20px;
}

.input-modal-properties label {
  font-size: 0.65rem;
  text-transform: uppercase;
  color: var(--controls-label-color, rgba(255, 255, 255, 0.4));
  letter-spacing: 0.5px;
  display: block;
  margin-bottom: 6px;
}

.input-modal-properties input {
  width: 100%;
  padding: 10px 12px;
  border-radius: 4px;
  border: none;
  background-color: var(--controls-input-color);
  color: inherit;
  font-size: 14px;
  margin-bottom: 16px;
  box-sizing: border-box;
}

.input-modal-properties textarea {
  width: 100%;
  padding: 10px 12px;
  border-radius: 4px;
  border: none;
  background-color: var(--controls-input-color);
  color: inherit;
  font-size: 13px;
  min-height: 60px;
  resize: vertical;
  font-family: inherit;
  line-height: 1.4;
  box-sizing: border-box;
}

/* Divider between sections */
.input-modal-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.06);
  margin: 0 20px;
}

/* Actions section */
.input-modal-actions {
  padding: 20px;
}

.input-modal-actions-label {
  font-size: 0.65rem;
  text-transform: uppercase;
  color: var(--controls-label-color, rgba(255, 255, 255, 0.4));
  letter-spacing: 0.5px;
  margin-bottom: 12px;
}

.action-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  background-color: var(--controls-input-color);
  border-radius: 4px;
  margin-bottom: 8px;
}

.action-row:last-child {
  margin-bottom: 0;
}

.action-row-text {
  flex: 1;
  min-width: 0;
  margin-right: 12px;
}

.action-row-title {
  font-size: 13px;
  font-weight: 500;
}

.action-row-description {
  font-size: 11px;
  color: var(--controls-label-color, rgba(255, 255, 255, 0.4));
  margin-top: 2px;
}

.action-row button {
  background-color: var(--controls-button-color);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 14px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  opacity: 0.8;
  transition: opacity 0.2s ease-in-out;
}

.action-row button:hover {
  opacity: 1;
}

/* Footer with Cancel + Save */
.input-modal-footer {
  display: flex;
  justify-content: flex-end;
  padding: 16px 20px;
  gap: 8px;
}

.input-modal-footer button {
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 13px;
  cursor: pointer;
  opacity: 0.8;
  transition: opacity 0.2s ease-in-out;
}

.input-modal-footer button:hover {
  opacity: 1;
}

.input-modal-footer .btn-cancel {
  background: transparent;
  color: var(--controls-label-color, rgba(255, 255, 255, 0.4));
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.input-modal-footer .btn-save {
  background-color: var(--controls-button-color);
  color: white;
  border: none;
  font-weight: 500;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/taxonomy/new-modal.css
git commit -m "feat: update modal CSS for redesigned layout"
```

---

### Task 7: Redesign NewTagModal component

**Files:**
- Modify: `src/renderer/components/taxonomy/new-tag-modal.tsx`

- [ ] **Step 1: Update the Props type to accept description**

The modal needs the current description when editing. Update `src/renderer/components/taxonomy/new-tag-modal.tsx`:

```typescript
type Props = {
  categoryLabel: string;
  handleClose: () => void;
  currentValue?: string;
  currentDescription?: string;
};
```

- [ ] **Step 2: Rewrite the component**

Replace the entire component in `src/renderer/components/taxonomy/new-tag-modal.tsx`:

```typescript
import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

import './new-modal.css';

type Props = {
  categoryLabel: string;
  handleClose: () => void;
  currentValue?: string;
  currentDescription?: string;
};

export default function NewTagModal({
  categoryLabel,
  handleClose,
  currentValue = '',
  currentDescription = '',
}: Props) {
  const [newLabel, setNewLabel] = useState<string>(currentValue);
  const [description, setDescription] = useState<string>(currentDescription);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const isEditing = Boolean(currentValue);

  function handleSubmit() {
    async function submit() {
      if (isEditing) {
        if (newLabel !== currentValue) {
          await invoke('rename-tag', [currentValue, newLabel]);
        }
        if (description !== currentDescription) {
          await invoke('update-tag-description', [
            newLabel,
            description,
          ]);
        }
      } else {
        await invoke('create-tag', [newLabel, categoryLabel, 0]);
        if (description) {
          await invoke('update-tag-description', [newLabel, description]);
        }
      }
      setNewLabel('');
      handleClose();
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    }
    submit();
  }

  async function handleApplyElo() {
    try {
      const result = await invoke('apply-elo-ordering', [currentValue]);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: `Reordered ${(result as any).count} items by ELO ranking`,
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to apply ELO ordering',
          message: String(e),
        },
      });
    }
  }

  async function handleConsolidateFiles() {
    try {
      const targetDir = await invoke('select-directory', [undefined]);
      if (!targetDir) return;

      const result = await invoke('consolidate-tag-files', [
        currentValue,
        targetDir,
      ]);
      const { copied, errors } = result as any;
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: errors > 0 ? 'error' : 'success',
          title: `Copied ${copied} files to ${targetDir}`,
          message: errors > 0 ? `${errors} files failed to copy` : undefined,
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to consolidate files',
          message: String(e),
        },
      });
    }
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {isEditing ? 'Edit Tag' : 'New Tag'}
          </div>
          <div
            className="input-modal-close"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            <img src={cancel} />
          </div>
        </div>

        <div className="input-modal-properties">
          <label>Name</label>
          <input
            autoFocus
            type="text"
            onChange={(e) => {
              e.stopPropagation();
              setNewLabel(e.currentTarget.value);
            }}
            value={newLabel}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
          />
          <label>Description</label>
          <textarea
            value={description}
            onChange={(e) => {
              e.stopPropagation();
              setDescription(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
            }}
            placeholder="Optional notes about this tag..."
          />
        </div>

        {isEditing && (
          <>
            <div className="input-modal-divider" />
            <div className="input-modal-actions">
              <div className="input-modal-actions-label">Actions</div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Apply ELO as custom order
                  </div>
                  <div className="action-row-description">
                    Seed custom sort weights from Battle Mode ELO rankings
                  </div>
                </div>
                <button onClick={handleApplyElo}>Apply</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Consolidate files to directory
                  </div>
                  <div className="action-row-description">
                    Copy tagged files into a chosen folder and update references
                  </div>
                </div>
                <button onClick={handleConsolidateFiles}>Choose...</button>
              </div>
            </div>
          </>
        )}

        <div className="input-modal-divider" />
        <div className="input-modal-footer">
          <button
            className="btn-cancel"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            Cancel
          </button>
          <button className="btn-save" onClick={handleSubmit}>
            {isEditing ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/components/taxonomy/new-tag-modal.tsx
git commit -m "feat: redesign NewTagModal with description and action rows"
```

---

### Task 8: Redesign NewCategoryModal component

**Files:**
- Modify: `src/renderer/components/taxonomy/new-category-modal.tsx`

- [ ] **Step 1: Rewrite the component**

Replace the entire contents of `src/renderer/components/taxonomy/new-category-modal.tsx`:

```typescript
import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';

import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

import './new-modal.css';

type Props = {
  handleClose: () => void;
  setCategory: (category: string) => void;
  currentValue?: string;
  currentDescription?: string;
};

export default function NewCategoryModal({
  handleClose,
  setCategory,
  currentValue = '',
  currentDescription = '',
}: Props) {
  const [newLabel, setNewLabel] = useState<string>(currentValue);
  const [description, setDescription] = useState<string>(currentDescription);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });

  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const isEditing = Boolean(currentValue);

  function handleSubmit() {
    async function submit() {
      if (isEditing) {
        if (newLabel !== currentValue) {
          await invoke('rename-category', [currentValue, newLabel]);
        }
        if (description !== currentDescription) {
          await invoke('update-category-description', [
            newLabel,
            description,
          ]);
        }
      } else {
        await invoke('create-category', [newLabel, 0]);
        if (description) {
          await invoke('update-category-description', [
            newLabel,
            description,
          ]);
        }
      }
      setNewLabel('');
      setCategory(newLabel);
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      handleClose();
    }
    submit();
  }

  async function handleResetOrdering() {
    try {
      await invoke('order-tags', [currentValue]);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'Tag order reset to alphabetical',
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to reset tag order',
          message: String(e),
        },
      });
    }
  }

  async function handleConsolidateFiles() {
    try {
      const targetDir = await invoke('select-directory', [undefined]);
      if (!targetDir) return;

      const result = await invoke('consolidate-category-files', [
        currentValue,
        targetDir,
      ]);
      const { copied, errors } = result as any;
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: errors > 0 ? 'error' : 'success',
          title: `Copied ${copied} files to ${targetDir}`,
          message: errors > 0 ? `${errors} files failed to copy` : undefined,
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to consolidate files',
          message: String(e),
        },
      });
    }
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {isEditing ? 'Edit Category' : 'New Category'}
          </div>
          <div
            className="input-modal-close"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            <img src={cancel} />
          </div>
        </div>

        <div className="input-modal-properties">
          <label>Name</label>
          <input
            autoFocus
            type="text"
            onChange={(e) => {
              e.stopPropagation();
              setNewLabel(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
            value={newLabel}
          />
          <label>Description</label>
          <textarea
            value={description}
            onChange={(e) => {
              e.stopPropagation();
              setDescription(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
            }}
            placeholder="Optional notes about this category..."
          />
        </div>

        {isEditing && (
          <>
            <div className="input-modal-divider" />
            <div className="input-modal-actions">
              <div className="input-modal-actions-label">Actions</div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">Reset tag order</div>
                  <div className="action-row-description">
                    Alphabetically sort all tags in this category
                  </div>
                </div>
                <button onClick={handleResetOrdering}>Reset</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Consolidate files to directory
                  </div>
                  <div className="action-row-description">
                    Copy all files in this category into a chosen folder
                  </div>
                </div>
                <button onClick={handleConsolidateFiles}>Choose...</button>
              </div>
            </div>
          </>
        )}

        <div className="input-modal-divider" />
        <div className="input-modal-footer">
          <button
            className="btn-cancel"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            Cancel
          </button>
          <button className="btn-save" onClick={handleSubmit}>
            {isEditing ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/taxonomy/new-category-modal.tsx
git commit -m "feat: redesign NewCategoryModal with description and action rows"
```

---

### Task 9: Wire up description props in Taxonomy.tsx

**Files:**
- Modify: `src/renderer/components/taxonomy/taxonomy.tsx`

- [ ] **Step 1: Find the tag description for the editing tag**

In `src/renderer/components/taxonomy/taxonomy.tsx`, find where `NewTagModal` is rendered (around line 339-345). The `editingTag` state holds the tag label. We need to look up the description from the taxonomy data. Update the rendering:

```typescript
{activeCategory && editingTag ? (
  <NewTagModal
    categoryLabel={activeCategory}
    handleClose={() => setEditingTag(null)}
    currentValue={editingTag}
    currentDescription={
      taxonomy?.[activeCategory]?.tags?.find(
        (t: Concept) => t.label === editingTag
      )?.description || ''
    }
  />
) : null}
```

- [ ] **Step 2: Pass description to NewCategoryModal**

Find where `NewCategoryModal` is rendered (around line 346-352). Update:

```typescript
{editingCategory ? (
  <NewCategoryModal
    handleClose={() => setEditingCategory(null)}
    setCategory={setActiveCategory}
    currentValue={editingCategory}
    currentDescription={
      taxonomy?.[editingCategory]?.description || ''
    }
  />
) : null}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/components/taxonomy/taxonomy.tsx
git commit -m "feat: pass description props to edit modals"
```

---

### Task 10: Manual verification

- [ ] **Step 1: Start the app**

Run the dev server and open the app.

- [ ] **Step 2: Test create tag flow**

Click "+" to create a new tag. Verify:
- Modal is wider (~550px)
- Name input and description textarea are visible
- Actions section is NOT visible
- Cancel and Create buttons in footer
- Creating a tag with a description saves both

- [ ] **Step 3: Test edit tag flow**

Click the pencil icon on an existing tag. Verify:
- Modal shows "Edit Tag" title
- Name and description fields are populated
- Actions section is visible with "Apply ELO as custom order" and "Consolidate files to directory"
- Renaming the tag works
- Adding/editing description persists after reopening the modal

- [ ] **Step 4: Test Apply ELO action**

On a tag that has media with ELO scores (from Battle Mode), click "Apply". Verify:
- Success toast appears with item count
- Media order within the tag reflects ELO ranking

- [ ] **Step 5: Test Consolidate Files action**

Click "Choose..." on a tag. Select a directory. Verify:
- Native OS folder picker opens
- Files are copied to the chosen directory
- Success toast shows count
- Media items now point to the new paths

- [ ] **Step 6: Test edit category flow**

Click pencil icon on a category. Verify:
- "Edit Category" title
- Name and description fields
- "Reset tag order" and "Consolidate files to directory" action rows
- Reset tag order shows toast and reorders tags alphabetically

- [ ] **Step 7: Test create category flow**

Click "+" for new category. Verify:
- Only name + description visible (no actions)
- Create button works

- [ ] **Step 8: Test dismiss behaviors**

Verify modal closes on:
- X button click
- Clicking outside the modal
- Cancel button click

- [ ] **Step 9: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address issues found during manual testing"
```
