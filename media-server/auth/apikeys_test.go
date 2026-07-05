package auth

import (
	"database/sql"
	"strings"
	"testing"

	_ "modernc.org/sqlite"
)

func newTestService(t *testing.T) *AuthService {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			created_at INTEGER
		)`,
		`CREATE TABLE api_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			name TEXT NOT NULL,
			key_hash TEXT UNIQUE NOT NULL,
			prefix TEXT NOT NULL,
			created_at INTEGER,
			last_used_at INTEGER
		)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("create table: %v", err)
		}
	}
	s := NewAuthService(db, "test-secret")
	if err := s.registerInternal("steve", "pw"); err != nil {
		t.Fatalf("register: %v", err)
	}
	return s
}

func TestCreateAndVerifyAPIKey(t *testing.T) {
	s := newTestService(t)

	plaintext, key, err := s.CreateAPIKey("steve", "lokictl")
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if !strings.HasPrefix(plaintext, APIKeyPrefix) {
		t.Errorf("key %q missing %q prefix", plaintext, APIKeyPrefix)
	}
	if !strings.HasPrefix(plaintext, key.Prefix) {
		t.Errorf("stored prefix %q does not match key %q", key.Prefix, plaintext)
	}
	if key.Username != "steve" || key.Name != "lokictl" || key.ID == 0 {
		t.Errorf("unexpected key metadata: %+v", key)
	}

	claims, err := s.VerifyAPIKey(plaintext)
	if err != nil {
		t.Fatalf("VerifyAPIKey: %v", err)
	}
	if claims.Username != "steve" {
		t.Errorf("username = %q, want steve", claims.Username)
	}
}

func TestVerifyAPIKeyRejectsBadKeys(t *testing.T) {
	s := newTestService(t)
	if _, _, err := s.CreateAPIKey("steve", "k"); err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	for _, bad := range []string{
		"",
		"not-a-key",
		"eyJhbGciOi.jwt.looking",                // JWT-shaped, no lk_ prefix
		APIKeyPrefix + strings.Repeat("00", 32), // right shape, wrong secret
	} {
		if _, err := s.VerifyAPIKey(bad); err == nil {
			t.Errorf("VerifyAPIKey(%q) succeeded, want error", bad)
		}
	}
}

func TestCreateAPIKeyValidation(t *testing.T) {
	s := newTestService(t)
	if _, _, err := s.CreateAPIKey("nobody", "k"); err != ErrUserNotFound {
		t.Errorf("unknown user: err = %v, want ErrUserNotFound", err)
	}
	if _, _, err := s.CreateAPIKey("steve", "   "); err == nil {
		t.Error("blank name accepted, want error")
	}
	if _, _, err := s.CreateAPIKey(DefaultAdminUsername, "k"); err == nil {
		t.Error("default admin accepted, want error")
	}
}

func TestDeleteAPIKey(t *testing.T) {
	s := newTestService(t)
	plaintext, key, err := s.CreateAPIKey("steve", "k")
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if err := s.DeleteAPIKey(key.ID); err != nil {
		t.Fatalf("DeleteAPIKey: %v", err)
	}
	if _, err := s.VerifyAPIKey(plaintext); err == nil {
		t.Error("revoked key still verifies")
	}
	if err := s.DeleteAPIKey(key.ID); err == nil {
		t.Error("deleting a missing key succeeded, want error")
	}
}

func TestDeleteUserRevokesKeys(t *testing.T) {
	s := newTestService(t)
	if err := s.registerInternal("other", "pw"); err != nil {
		t.Fatalf("register: %v", err)
	}
	plaintext, _, err := s.CreateAPIKey("other", "k")
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if err := s.DeleteUser("other"); err != nil {
		t.Fatalf("DeleteUser: %v", err)
	}
	if _, err := s.VerifyAPIKey(plaintext); err == nil {
		t.Error("key of a deleted user still verifies")
	}
}

func TestListAPIKeys(t *testing.T) {
	s := newTestService(t)
	if _, _, err := s.CreateAPIKey("steve", "a"); err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if _, _, err := s.CreateAPIKey("steve", "b"); err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	keys, err := s.ListAPIKeys()
	if err != nil {
		t.Fatalf("ListAPIKeys: %v", err)
	}
	if len(keys) != 2 {
		t.Fatalf("len(keys) = %d, want 2", len(keys))
	}
	for _, k := range keys {
		if k.Username != "steve" || k.Prefix == "" || k.CreatedAt == 0 {
			t.Errorf("incomplete key row: %+v", k)
		}
	}
}
