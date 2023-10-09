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
  weight INTEGER,
  PRIMARY KEY (media_path, tag_label, category_label),
  FOREIGN KEY (media_path) REFERENCES media (path),
  FOREIGN KEY (tag_label) REFERENCES tag (label),
  FOREIGN KEY (category_label) REFERENCES category (label)
)
