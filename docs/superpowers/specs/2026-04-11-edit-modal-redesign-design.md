# Edit Tag / Category Modal Redesign

## Problem

The current edit tag and edit category modals are barebones — a single name input and a save button. They need to support descriptions, growing sets of actions, and a cleaner layout that scales as new features are added.

## Design

### Modal Layout

Both modals share the same structural layout (~550px wide, centered):

1. **Header** — title ("Edit Tag" / "Edit Category") and close button
2. **Properties section** — name input and description textarea
3. **Actions section** — list of action rows, each with title, description, and action button. Hidden in create mode.
4. **Footer** — Cancel and Save buttons

The Save button persists name and description changes. Actions execute independently via their own buttons.

### Database Changes

Add a `description TEXT` column to both the `tag` and `category` tables via `ALTER TABLE ADD COLUMN`. Nullable, no default — existing rows get `NULL`.

### New IPC Handlers

| Handler | Args | Purpose |
|---------|------|---------|
| `update-tag-description` | `[label, description]` | Save tag description |
| `update-category-description` | `[label, description]` | Save category description |
| `select-directory` | none | Open native OS folder picker, return chosen path or null |
| `consolidate-tag-files` | `[tagLabel, targetDir]` | Copy files for a tag to target directory, update DB paths |
| `consolidate-category-files` | `[categoryLabel, targetDir]` | Copy files for all tags in a category to target directory, update DB paths |
| `apply-elo-ordering` | `[tagLabel]` | Overwrite media_tag_by_category weights with ELO-derived ordering |

### Edit Tag Actions

1. **Apply ELO as custom order** — Takes ELO scores from the `media` table for all media items with this tag. Sorts by ELO descending and assigns incrementing `weight` values on the `media_tag_by_category` rows, so the highest ELO item gets the highest weight. Items with no ELO score are placed at the end. This seeds custom sort order from Battle Mode rankings, which the user can then fine-tune via drag-and-drop.

2. **Consolidate files to directory** — Opens the native OS folder picker. Copies all media files associated with this tag into the chosen directory. Updates `media.path` and `media_tag_by_category.media_path` in the database to point to the new copies. Filename collisions are handled with numeric suffixes (`photo.jpg` → `photo_1.jpg`).

### Edit Category Actions

1. **Reset tag order** — Existing functionality, moved from an inline button to an action row. Alphabetically sorts all tags in the category by updating their weights.

2. **Consolidate files to directory** — Same as the tag version, but operates on all media files across all tags in the category.

### Create Mode

When creating a new tag or category (no `currentValue`), the modal shows only the properties section (name + description) and footer. The actions section is hidden since actions only apply to existing items.

### Action Feedback

Actions use the existing toast system via `libraryService.send('ADD_TOAST', ...)`. Success toasts show result counts (e.g., "Copied 42 files to /path/to/dir", "Reordered 87 items by ELO ranking"). Errors show `type: 'error'` with a message.

### Consolidate Files Backend Logic

1. Query all `media_path` values from `media_tag_by_category` for the given tag/category
2. For each file, copy to `targetDir` using `fs.copyFile`
3. Handle filename collisions with numeric suffix
4. Update `media.path` to the new path
5. Update `media_tag_by_category.media_path` to the new path
6. Return count of files copied and any errors

### Apply ELO Ordering Backend Logic

1. Query all media for the tag: `SELECT m.path, m.elo FROM media m JOIN media_tag_by_category mtc ON m.path = mtc.media_path WHERE mtc.tag_label = ?`
2. Sort by `m.elo` descending
3. Assign incrementing weights so highest ELO = highest weight
4. Update each row: `UPDATE media_tag_by_category SET weight = ? WHERE media_path = ? AND tag_label = ?`
5. Return count of items reordered

### Data Fetching

The modals need the current description when opening in edit mode. The existing taxonomy query (React Query, key `['taxonomy']`) fetches tags and categories. This query's backend handler and return type must be updated to include `description` fields so they're available when the modal opens. The `Concept` / taxonomy types in the renderer need to be updated accordingly.

### Files Changed

| File | Change |
|------|--------|
| `src/main/media.ts` | DB migration, new IPC handlers |
| `src/renderer/components/taxonomy/new-tag-modal.tsx` | Redesigned component with description + actions |
| `src/renderer/components/taxonomy/new-category-modal.tsx` | Redesigned component with description + actions |
| `src/renderer/components/taxonomy/new-modal.css` | Updated styles for wider modal, sections, action rows |

No new files. Existing trigger pattern from `Taxonomy.tsx` unchanged.

### CSS Approach

Update `new-modal.css` with:
- Wider `.input-modal-content` (550px)
- Full opacity (remove the `opacity: 0.7`)
- New classes for sections: `.input-modal-properties`, `.input-modal-actions`, `.input-modal-footer`
- Action row styling: `.action-row` with flex layout, background, title/subtitle/button
- Divider styling between sections
- Textarea styling for description field
- Footer with right-aligned Cancel + Save buttons
