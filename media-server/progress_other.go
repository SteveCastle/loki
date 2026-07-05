//go:build !windows

package main

// enableVirtualTerminal is a no-op on non-Windows platforms, where terminals
// process ANSI escape sequences natively.
func enableVirtualTerminal() {}
