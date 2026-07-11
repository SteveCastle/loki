package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

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

// mediaRowExists reports whether path is a row in the media table — i.e.
// curated library content an admin added, as opposed to an arbitrary path.
func mediaRowExists(deps *Dependencies, path string) bool {
	if deps == nil || deps.DB == nil {
		return false
	}
	var one int
	err := deps.DB.QueryRow("SELECT 1 FROM media WHERE path = ? LIMIT 1", path).Scan(&one)
	return err == nil
}

// mediaReadAllowed is the gate for anon-reachable handlers that turn a
// media path into a file read, subprocess, or server-side fetch. Admins are
// unrestricted (Electron/desktop serve arbitrary local files). Non-admins
// may only touch: paths inside a configured storage root (incl. s3://), or
// a curated library row — and NEVER an http(s):// URL, which would make the
// server fetch it (SSRF) or hand it to a subprocess that speaks network
// protocols (ffprobe/ffmpeg).
func mediaReadAllowed(deps *Dependencies, r *http.Request, path string) bool {
	if isAdminRequest(deps, r) {
		return true
	}
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return false
	}
	if deps.Storage != nil && deps.Storage.BackendFor(path) != nil {
		return true
	}
	return mediaRowExists(deps, path)
}

// ssrfSafeHTTPClient is an http.Client whose dialer refuses to connect to
// non-public IPs (loopback, private, link-local, cloud-metadata). It guards
// the remote-media proxy against SSRF even for admin-triggered fetches and
// DNS-rebinding, since the check runs on the RESOLVED address at connect
// time.
var ssrfSafeHTTPClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, err
			}
			ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
			if err != nil {
				return nil, err
			}
			for _, ip := range ips {
				if !isPublicIP(ip.IP) {
					return nil, fmt.Errorf("refusing to connect to non-public address %s", ip.IP)
				}
			}
			d := net.Dialer{Timeout: 10 * time.Second}
			// Dial the resolved IP directly so the connection can't race the
			// check via a second DNS lookup (rebinding).
			return d.DialContext(ctx, network, net.JoinHostPort(ips[0].IP.String(), port))
		},
	},
}

// isPublicIP reports whether ip is safe to connect to from the server: not
// loopback, private, link-local, unspecified, or multicast.
func isPublicIP(ip net.IP) bool {
	if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() ||
		ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() {
		return false
	}
	// Carrier-grade NAT 100.64.0.0/10 (covers the common cloud-metadata edge
	// cases some providers place there) — treat as non-public.
	if v4 := ip.To4(); v4 != nil && v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127 {
		return false
	}
	return true
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
