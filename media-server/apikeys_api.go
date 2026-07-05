package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/auth"
)

// requestAuthToken extracts the API credential from a request: the X-API-Key
// header or the Authorization Bearer value.
func requestAuthToken(r *http.Request) string {
	if h := strings.TrimSpace(r.Header.Get("X-API-Key")); h != "" {
		return h
	}
	if ah := r.Header.Get("Authorization"); strings.HasPrefix(ah, "Bearer ") {
		return strings.TrimPrefix(ah, "Bearer ")
	}
	return ""
}

// verifyCredential authenticates either credential kind: lk_-prefixed API
// keys against the api_keys table, anything else as a JWT.
func verifyCredential(deps *Dependencies, token string) (*auth.Claims, error) {
	if strings.HasPrefix(token, auth.APIKeyPrefix) {
		return deps.Auth.VerifyAPIKey(token)
	}
	return deps.Auth.VerifyToken(token)
}

// requestUsername resolves the authenticated user for a request that already
// passed the auth middleware (header credential first, then cookie).
func requestUsername(deps *Dependencies, r *http.Request) string {
	if token := requestAuthToken(r); token != "" {
		if claims, err := verifyCredential(deps, token); err == nil {
			return claims.Username
		}
	}
	if cookie, err := r.Cookie("auth_token"); err == nil {
		if claims, err := deps.Auth.VerifyToken(cookie.Value); err == nil {
			return claims.Username
		}
	}
	return ""
}

// apiKeysHandler implements /auth/keys: GET lists keys, POST creates one
// (plaintext returned exactly once), DELETE ?id= revokes. Registered behind
// RoleAdmin, unlike /auth/users which must stay public for first-run setup.
func apiKeysHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			keys, err := deps.Auth.ListAPIKeys()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if keys == nil {
				keys = []auth.APIKey{}
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"keys": keys})

		case http.MethodPost:
			var req struct {
				Name     string `json:"name"`
				Username string `json:"username"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}
			if strings.TrimSpace(req.Name) == "" {
				http.Error(w, "Key name required", http.StatusBadRequest)
				return
			}
			username := req.Username
			if username == "" {
				username = requestUsername(deps, r)
			}
			if username == "" {
				http.Error(w, "Username required", http.StatusBadRequest)
				return
			}
			plaintext, key, err := deps.Auth.CreateAPIKey(username, req.Name)
			if err != nil {
				status := http.StatusInternalServerError
				if err == auth.ErrUserNotFound {
					status = http.StatusBadRequest
				}
				http.Error(w, err.Error(), status)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "created",
				"key":      plaintext,
				"id":       key.ID,
				"name":     key.Name,
				"username": key.Username,
				"prefix":   key.Prefix,
			})

		case http.MethodDelete:
			id, err := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
			if err != nil {
				http.Error(w, "Key id required", http.StatusBadRequest)
				return
			}
			if err := deps.Auth.DeleteAPIKey(id); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"deleted"}`))

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}
