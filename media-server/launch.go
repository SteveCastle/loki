package main

// Launch behaviour shared by the Windows, Linux and macOS mains.
//
// The server is started three ways: by hand (Start Menu, Finder, a
// terminal), at login (the installer's autostart shortcut), and by the
// Lowkey Media Viewer, which starts it in the background every time the
// viewer opens. Only the first should pop a browser tab; the other two pass
// --background (or set LOWKEY_NO_BROWSER=1) and stay quiet.
//
// A second launch while a server already answers on the configured port
// never starts another instance (it would fail to bind and its tray icon
// would flash up and vanish): a manual launch just opens the running
// server's UI, a background launch exits silently.

import (
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/pkg/browser"
	"github.com/stevecastle/shrike/appconfig"
)

// Version is the release this binary was built from, stamped by CI with
// -ldflags "-X main.Version=<version>" (release.yml). Reported by /health so
// the desktop viewer can offer to upgrade a server older than itself.
var Version = "dev"

// launchedInBackground reports whether this process was started by
// something other than the user (login item, the desktop viewer).
func launchedInBackground() bool {
	for _, a := range os.Args[1:] {
		switch strings.ToLower(a) {
		case "--background", "-background", "/background":
			return true
		}
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LOWKEY_NO_BROWSER"))) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// openUI opens the web UI in the default browser.
func openUI() {
	_ = browser.OpenURL(appconfig.Get().LocalBaseURL() + "/")
}

// openUIOnLaunch opens the web UI for a manual launch only.
func openUIOnLaunch() {
	if launchedInBackground() {
		log.Printf("Started in the background. Web UI: %s", appconfig.Get().LocalBaseURL())
		return
	}
	openUI()
}

// serverAlreadyRunning probes the configured port for a live server.
func serverAlreadyRunning() bool {
	client := &http.Client{Timeout: 1500 * time.Millisecond}
	resp, err := client.Get(appconfig.Get().LocalBaseURL() + "/health")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// exitIfAlreadyRunning hands off to an instance that is already serving.
// Called first thing in main, before the database is opened.
func exitIfAlreadyRunning() {
	if !serverAlreadyRunning() {
		return
	}
	log.Printf("Lowkey Media Server is already running at %s", appconfig.Get().LocalBaseURL())
	if !launchedInBackground() {
		openUI()
	}
	os.Exit(0)
}
