package main

// appendFieldArgs appends each non-empty key/value pair from fields onto args
// as two slice elements: "--<key>" then "<value>". The values are passed
// through verbatim (no quoting, no escaping) because ParseOptions consumes
// the resulting slice directly — never via the shell-style ParseCommand
// splitter. Callers can therefore safely include arbitrary text such as
// embedded quotes or newlines in field values.
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
