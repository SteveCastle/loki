package main

import (
	"net/http"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/auth"
)

// isAdminRequest reports whether r carries valid admin credentials — a
// header credential (Authorization Bearer JWT / lk_ API key / X-API-Key,
// see requestAuthToken and verifyCredential) or the auth_token cookie.
// Mirrors authMiddleware's checks, including the setup-required
// default-admin lockout, but never writes a response.
func isAdminRequest(deps *Dependencies, r *http.Request) bool {
	if tok := requestAuthToken(r); tok != "" {
		if claims, err := verifyCredential(deps, tok); err == nil {
			return !isSetupLockedAdmin(deps, claims)
		}
	}
	cookie, err := r.Cookie("auth_token")
	if err != nil {
		return false
	}
	claims, err := deps.Auth.VerifyToken(cookie.Value)
	if err != nil {
		return false
	}
	return !isSetupLockedAdmin(deps, claims)
}

// isSetupLockedAdmin matches authMiddleware's gate: the bootstrap default
// admin account doesn't count as an admin while first-run setup is still
// required.
func isSetupLockedAdmin(deps *Dependencies, claims *auth.Claims) bool {
	if claims.Username != auth.DefaultAdminUsername {
		return false
	}
	setupRequired, _ := deps.Auth.IsSetupRequired()
	return setupRequired
}

// pathAllowedForRequest scopes media-path access for non-admin requesters:
// s3:// paths must live inside a configured storage root, and LOCAL paths
// outside every root are admin-only (an anonymous public-access visitor
// must never read arbitrary files off the server's filesystem). Admins keep
// today's unrestricted behavior — the Electron viewer serves arbitrary
// local files through its own server.
func pathAllowedForRequest(deps *Dependencies, r *http.Request, path string) bool {
	if deps.Storage != nil && deps.Storage.BackendFor(path) != nil {
		return true
	}
	return isAdminRequest(deps, r)
}

func writeUnauthorizedJSON(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	w.Write([]byte(`{"error":"unauthorized"}`))
}

// requireAuthWhenPublic gates write branches that anonymous visitors could
// otherwise reach while AllowPublicAccess is on: the write methods of
// mixed-method RolePublicRead routes (whose outer middleware passes
// everyone when the flag is on), and the deps/onboarding write endpoints
// (open by design for Electron/the first-run wizard). With the flag off it
// is a pass-through — those routes are then already behind authMiddleware
// or intentionally open. Failures answer 401 JSON (no login redirect);
// these branches are only reachable from fetch(), never top-level
// navigation. Evaluated per request so toggling the flag needs no restart.
func requireAuthWhenPublic(deps *Dependencies, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if appconfig.Get().AllowPublicAccess && !isAdminRequest(deps, r) {
			writeUnauthorizedJSON(w)
			return
		}
		h(w, r)
	}
}
