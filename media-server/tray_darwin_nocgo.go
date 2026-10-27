//go:build darwin && !cgo

package main

// Headless fallback for macOS builds without cgo (cross-compiles, dev
// checks from other platforms). No Cocoa means no menu-bar item; the
// server runs until an OS signal arrives. The released .app is built with
// cgo and gets the real tray (tray_darwin.go).

import (
	"log"
	"os"
)

func runDarwinUI(sigChan <-chan os.Signal) {
	openUIOnLaunch()
	sig := <-sigChan
	log.Printf("Received signal %v, shutting down...", sig)
	onExit()
}
