#!/usr/bin/env bash
#
# normalize-downloads-case.sh
#
# Normalizes the case of the "Downloads" directory segment in the Lowkey Media
# Viewer library (dream-x.sqlite) onto the lowercase form  Z:\downloads\ .
#
# Windows filesystems are case-insensitive, so  Z:\Downloads\foo.jpg  and
# Z:\downloads\foo.jpg  are the SAME physical file. The DB recorded both casings
# at different times, producing logical duplicates. This script collapses the
# capitalized variant onto the lowercase one:
#
#   * DUPLICATE  - capital path whose lowercase twin already exists:
#                  merge the capital row's tag assignments into the lowercase
#                  survivor (collision-safe), then delete the capital media row
#                  and its leftover tag rows.
#   * RENAME     - capital path with no lowercase twin:
#                  rewrite media.path and media_tag_by_category.media_path from
#                  Z:\Downloads\...  ->  Z:\downloads\...
#
# Only the leading "Downloads" segment is changed; the rest of each path
# (sub-folders / filename) is left exactly as-is.
#
# Usage:
#   ./normalize-downloads-case.sh           # dry run: report what would change
#   ./normalize-downloads-case.sh --apply   # backup DB, then apply in one txn
#
set -euo pipefail

DB="C:\\Users\\steph\\AppData\\Roaming\\Lowkey Media Viewer\\dream-x.sqlite"
CAP_PREFIX='Z:\Downloads\'   # 13 chars: Z : \ D o w n l o a d s \
LOW_PREFIX='Z:\downloads\'
TARGET_EXPR="'${LOW_PREFIX}' || substr(path,14)"          # for media.path
TARGET_EXPR_MT="'${LOW_PREFIX}' || substr(media_path,14)" # for junction

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

echo "DB: $DB"
echo

echo "=== Current state (case-sensitive, binary collation) ==="
sqlite3 "file:${DB}?mode=ro" -readonly "
WITH cap AS (
  SELECT path AS cpath, ${TARGET_EXPR} AS target
  FROM media WHERE substr(path,1,13) = '${CAP_PREFIX}'
)
SELECT
  (SELECT COUNT(*) FROM cap)                                          AS capital_media_rows,
  (SELECT COUNT(*) FROM cap WHERE target IN (SELECT path FROM media)) AS duplicates_to_delete,
  (SELECT COUNT(*) FROM cap WHERE target NOT IN (SELECT path FROM media)) AS rows_to_rename,
  (SELECT COUNT(*) FROM media_tag_by_category
     WHERE substr(media_path,1,13)='${CAP_PREFIX}')                  AS capital_tag_rows;
"
echo

if [[ $APPLY -eq 0 ]]; then
  echo "Dry run only. Re-run with --apply to back up the DB and perform the changes."
  exit 0
fi

# --- Backup -------------------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
BACKUP="${DB%.sqlite}.backup-${TS}.sqlite"
echo "=== Backing up DB ==="
echo "  -> $BACKUP"
cp -- "$DB" "$BACKUP"
echo "Backup complete ($(stat -c%s "$BACKUP" 2>/dev/null || wc -c < "$BACKUP") bytes)."
echo

# --- Apply (single transaction, FK enforcement off) ---------------------------
echo "=== Applying normalization ==="
sqlite3 "$DB" <<SQL
PRAGMA foreign_keys = OFF;
BEGIN;

-- Snapshot the work set so 'exists?' is decided before we start mutating media.
CREATE TEMP TABLE cap_map AS
  SELECT path AS cpath,
         ${TARGET_EXPR} AS target,
         (${TARGET_EXPR}) IN (SELECT path FROM media) AS is_dup
  FROM media
  WHERE substr(path,1,13) = '${CAP_PREFIX}';
CREATE INDEX tmp_cap_cpath  ON cap_map(cpath);
CREATE INDEX tmp_cap_isdup  ON cap_map(is_dup);

-- 1) RENAMES: capital paths with no lowercase twin. Repoint junction, then media.
UPDATE media_tag_by_category
   SET media_path = (SELECT target FROM cap_map WHERE cpath = media_path)
 WHERE media_path IN (SELECT cpath FROM cap_map WHERE is_dup = 0);

UPDATE media
   SET path = (SELECT target FROM cap_map WHERE cpath = path)
 WHERE path IN (SELECT cpath FROM cap_map WHERE is_dup = 0);

-- 2) DUPLICATES: merge capital tag rows into the existing lowercase survivor,
--    skipping rows that collide on the junction PK (media_path,tag,category,ts).
INSERT OR IGNORE INTO media_tag_by_category
  (media_path, tag_label, category_label, weight, time_stamp, job_id, created_at)
SELECT cm.target, mt.tag_label, mt.category_label, mt.weight,
       mt.time_stamp, mt.job_id, mt.created_at
  FROM media_tag_by_category mt
  JOIN cap_map cm ON cm.cpath = mt.media_path
 WHERE cm.is_dup = 1;

DELETE FROM media_tag_by_category
 WHERE media_path IN (SELECT cpath FROM cap_map WHERE is_dup = 1);

DELETE FROM media
 WHERE path IN (SELECT cpath FROM cap_map WHERE is_dup = 1);

DROP TABLE cap_map;
COMMIT;
PRAGMA foreign_key_check;
SQL
echo "Apply complete."
echo

# --- Verify -------------------------------------------------------------------
echo "=== Verification (all should be 0) ==="
sqlite3 "file:${DB}?mode=ro" -readonly "
SELECT
  (SELECT COUNT(*) FROM media
     WHERE substr(path,1,13)='${CAP_PREFIX}')              AS remaining_capital_media,
  (SELECT COUNT(*) FROM media_tag_by_category
     WHERE substr(media_path,1,13)='${CAP_PREFIX}')        AS remaining_capital_tag_rows;
"
echo
echo "=== Orphan tag rows (media_path not present in media) ==="
sqlite3 "file:${DB}?mode=ro" -readonly "
SELECT COUNT(*) AS orphan_tag_rows
FROM media_tag_by_category mt
WHERE NOT EXISTS (SELECT 1 FROM media m WHERE m.path = mt.media_path)
  AND substr(mt.media_path,1,13) IN ('${LOW_PREFIX}','${CAP_PREFIX}');
"
echo "Done. Backup left at: $BACKUP"
