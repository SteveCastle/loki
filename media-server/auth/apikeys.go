package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"strings"
	"time"
)

// APIKeyPrefix marks a credential as an API key so callers can route it to
// VerifyAPIKey instead of JWT parsing.
const APIKeyPrefix = "lk_"

var ErrInvalidAPIKey = errors.New("invalid API key")

type APIKey struct {
	ID         int64  `json:"id"`
	Username   string `json:"username"`
	Name       string `json:"name"`
	Prefix     string `json:"prefix"`
	CreatedAt  int64  `json:"created_at"`
	LastUsedAt int64  `json:"last_used_at"`
}

func hashAPIKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// CreateAPIKey mints a key for username and returns the plaintext exactly
// once; only its SHA-256 hash is stored.
func (s *AuthService) CreateAPIKey(username, name string) (string, *APIKey, error) {
	if username == DefaultAdminUsername {
		return "", nil, errors.New("cannot create API keys for the temporary admin account")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return "", nil, errors.New("key name is required")
	}

	var userID int64
	err := s.db.QueryRow("SELECT id FROM users WHERE username = ?", username).Scan(&userID)
	if err == sql.ErrNoRows {
		return "", nil, ErrUserNotFound
	} else if err != nil {
		return "", nil, err
	}

	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", nil, err
	}
	key := APIKeyPrefix + hex.EncodeToString(raw)
	// Stored so listings can identify a key without revealing it.
	prefix := key[:len(APIKeyPrefix)+8]

	now := time.Now().Unix()
	res, err := s.db.Exec(
		"INSERT INTO api_keys (user_id, name, key_hash, prefix, created_at, last_used_at) VALUES (?, ?, ?, ?, ?, 0)",
		userID, name, hashAPIKey(key), prefix, now,
	)
	if err != nil {
		return "", nil, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return "", nil, err
	}
	return key, &APIKey{ID: id, Username: username, Name: name, Prefix: prefix, CreatedAt: now}, nil
}

// VerifyAPIKey resolves a plaintext API key to its owning user, returning
// Claims so callers can treat keys and JWTs uniformly.
func (s *AuthService) VerifyAPIKey(key string) (*Claims, error) {
	if !strings.HasPrefix(key, APIKeyPrefix) {
		return nil, ErrInvalidAPIKey
	}
	var (
		id       int64
		username string
		lastUsed int64
	)
	err := s.db.QueryRow(`
		SELECT k.id, u.username, k.last_used_at
		FROM api_keys k JOIN users u ON u.id = k.user_id
		WHERE k.key_hash = ?`, hashAPIKey(key)).Scan(&id, &username, &lastUsed)
	if err == sql.ErrNoRows {
		return nil, ErrInvalidAPIKey
	} else if err != nil {
		return nil, err
	}

	// Best-effort last-used stamp, throttled so hot paths (thumbnails, HLS)
	// don't issue a write per request.
	if now := time.Now().Unix(); now-lastUsed > 60 {
		_, _ = s.db.Exec("UPDATE api_keys SET last_used_at = ? WHERE id = ?", now, id)
	}
	return &Claims{Username: username}, nil
}

func (s *AuthService) ListAPIKeys() ([]APIKey, error) {
	rows, err := s.db.Query(`
		SELECT k.id, u.username, k.name, k.prefix, k.created_at, k.last_used_at
		FROM api_keys k JOIN users u ON u.id = k.user_id
		ORDER BY k.created_at DESC, k.id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var keys []APIKey
	for rows.Next() {
		var k APIKey
		if err := rows.Scan(&k.ID, &k.Username, &k.Name, &k.Prefix, &k.CreatedAt, &k.LastUsedAt); err != nil {
			return nil, err
		}
		keys = append(keys, k)
	}
	return keys, rows.Err()
}

func (s *AuthService) DeleteAPIKey(id int64) error {
	res, err := s.db.Exec("DELETE FROM api_keys WHERE id = ?", id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return errors.New("API key not found")
	}
	return nil
}
