CREATE TABLE IF NOT EXISTS media (
  path TEXT PRIMARY KEY);


CREATE TABLE IF NOT EXISTS category (
  label TEXT PRIMARY KEY,
  weight INTEGER);

CREATE TABLE IF NOT EXISTS tag (
  label TEXT PRIMARY KEY,
  category_label TEXT,
  weight INTEGER,
  FOREIGN KEY (category_label) REFERENCES category (label)
  );

CREATE TABLE IF NOT EXISTS media_tag_by_category (
  media_path TEXT,
  tag_label TEXT,
  category_label TEXT,
  weight REAL,
  -- time_stamp is the in-media offset (seconds into a video) the tag applies
  -- to; 0 means the media in general. It is NOT a creation time.
  time_stamp REAL,
  -- created_at is the wall-clock moment the tag was applied (Unix epoch).
  created_at INTEGER,
  PRIMARY KEY (media_path, tag_label, category_label, time_stamp),
  FOREIGN KEY (media_path) REFERENCES media (path),
  FOREIGN KEY (tag_label) REFERENCES tag (label),
  FOREIGN KEY (category_label) REFERENCES category (label)
)
