package main

// appendFieldArgs appends each non-empty key/value pair from fields onto args
// as two slice elements: "--<key>" then "<value>". The values are passed
// through verbatim (no quoting, no escaping) because ParseOptions consumes
// the resulting slice directly — never via the shell-style ParseCommand
// splitter. Callers can therefore safely include arbitrary text such as
// embedded quotes or newlines in field values.
//
// Scope: any key is accepted — this helper is a generic argument-injection
// mechanism, not a prompt-only path. The only consumer today (POST /create)
// is gated behind renderer.RoleAdmin, so the surface is restricted to
// trusted callers; the media server's deployment model is single-user-local
// (running on the user's own machine alongside the Electron desktop app).
// If a multi-user deployment ever ships, this helper should grow an
// allowlist of safe keys per task, or callers should pre-filter req.Fields.
func appendFieldArgs(args []string, fields map[string]string) []string {
	if len(fields) == 0 {
		return args
	}
	for k, v := range fields {
		if k == "" || v == "" {
			continue
		}
		args = append(args, "--"+k, v)
	}
	return args
}
