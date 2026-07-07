//go:build darwin && cgo

package main

// Menu-bar UI for macOS, mirroring the Windows tray. systray needs Cocoa
// (cgo), so this file only compiles in native macOS builds — the released
// .app is built with CGO_ENABLED=1 on macOS runners. Cross-compiled or
// CGO_ENABLED=0 builds get the headless fallback in tray_darwin_nocgo.go.

import (
	_ "embed"
	"log"
	"os"

	"github.com/getlantern/systray"
	"github.com/pkg/browser"
	"github.com/stevecastle/shrike/appconfig"
)

//go:embed assets/logo.png
var trayIconPNG []byte

// runDarwinUI blocks until quit: a menu-bar item with Open/Quit, with OS
// signals routed into the same clean-shutdown path. onExit runs exactly
// once, via systray's teardown callback.
func runDarwinUI(sigChan <-chan os.Signal) {
	go func() {
		sig := <-sigChan
		log.Printf("Received signal %v, shutting down...", sig)
		systray.Quit()
	}()

	systray.Run(func() {
		systray.SetTemplateIcon(trayIconPNG, trayIconPNG)
		systray.SetTooltip("Lowkey Media Server – click to open UI")

		openItem := systray.AddMenuItem("Open Web UI", "Launch the browser")
		systray.AddSeparator()
		quitItem := systray.AddMenuItem("Quit", "Shut down Lowkey Media Server")

		// open UI once at startup
		_ = browser.OpenURL(appconfig.Get().LocalBaseURL() + "/")

		go func() {
			for {
				select {
				case <-openItem.ClickedCh:
					_ = browser.OpenURL(appconfig.Get().LocalBaseURL() + "/")
				case <-quitItem.ClickedCh:
					systray.Quit()
					return
				}
			}
		}()
	}, onExit)
}
