package media

import (
	"database/sql"

	"github.com/stevecastle/shrike/embedvec"
)

// StoredEmbedding pairs a media path with its decoded vector.
type StoredEmbedding struct {
	Path string
	Vec  []float32
}

// UpsertEmbedding writes (or replaces) the embedding for path under model.
func UpsertEmbedding(db *sql.DB, path, model string, vec []float32, createdAt int64) error {
	_, err := db.Exec(
		`INSERT INTO media_embedding (media_path, model, dim, vector, created_at)
		 VALUES (?,?,?,?,?)
		 ON CONFLICT(media_path, model)
		 DO UPDATE SET dim=excluded.dim, vector=excluded.vector, created_at=excluded.created_at`,
		path, model, len(vec), embedvec.Encode(vec), createdAt,
	)
	return err
}

// HasEmbedding reports whether path already has an embedding for model.
func HasEmbedding(db *sql.DB, path, model string) (bool, error) {
	var one int
	err := db.QueryRow(
		`SELECT 1 FROM media_embedding WHERE media_path=? AND model=? LIMIT 1`,
		path, model,
	).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	return err == nil, err
}

// GetEmbedding returns the decoded vector for path/model.
func GetEmbedding(db *sql.DB, path, model string) ([]float32, bool, error) {
	var blob []byte
	err := db.QueryRow(
		`SELECT vector FROM media_embedding WHERE media_path=? AND model=?`,
		path, model,
	).Scan(&blob)
	if err == sql.ErrNoRows {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	v, err := embedvec.Decode(blob)
	return v, true, err
}

// LoadAllEmbeddings returns every stored embedding for model. Used by the
// brute-force scan (Phase 1) and the index builder (Phase 2).
func LoadAllEmbeddings(db *sql.DB, model string) ([]StoredEmbedding, error) {
	rows, err := db.Query(
		`SELECT media_path, vector FROM media_embedding WHERE model=?`, model,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []StoredEmbedding
	for rows.Next() {
		var path string
		var blob []byte
		if err := rows.Scan(&path, &blob); err != nil {
			return nil, err
		}
		v, err := embedvec.Decode(blob)
		if err != nil {
			return nil, err
		}
		out = append(out, StoredEmbedding{Path: path, Vec: v})
	}
	return out, rows.Err()
}
