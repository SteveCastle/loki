//go:build darwin && !cgo

package main

// Headless fallback for macOS builds without cgo (cross-compiles, dev
// checks from other platforms). No Cocoa means no menu-bar item; the
// server runs until an OS signal arrives. The released .app is built with
// cgo and gets the real tray (tray_darwin.go).

import (
	"log"
	"os"

	"github.com/pkg/browser"
	"github.com/stevecastle/shrike/appconfig"
)

func runDarwinUI(sigChan <-chan os.Signal) {
	_ = browser.OpenURL(appconfig.Get().LocalBaseURL() + "/")
	sig := <-sigChan
	log.Printf("Received signal %v, shutting down...", sig)
	onExit()
}
