//go:build windows

package main

import (
	"os"

	"golang.org/x/sys/windows"
)

// enableVirtualTerminal turns on ANSI escape-sequence processing for the stderr
// console so the colored progress bar renders (legacy conhost needs this; modern
// terminals already support it, and this is a harmless no-op there).
func enableVirtualTerminal() {
	h := windows.Handle(os.Stderr.Fd())
	var mode uint32
	if err := windows.GetConsoleMode(h, &mode); err != nil {
		return
	}
	_ = windows.SetConsoleMode(h, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING)
}
