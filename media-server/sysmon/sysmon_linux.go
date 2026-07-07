//go:build linux

package sysmon

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// cpuTimes parses the aggregate "cpu" line of /proc/stat. Units are jiffies;
// only deltas matter so the unit is irrelevant.
func cpuTimes() (busy, total float64, ok bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	line, _, _ := strings.Cut(string(data), "\n")
	fields := strings.Fields(line)
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0, 0, false
	}
	var vals []float64
	for _, f := range fields[1:] {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return 0, 0, false
		}
		vals = append(vals, v)
	}
	for _, v := range vals {
		total += v
	}
	// busy = total - idle - iowait
	idle := vals[3]
	if len(vals) > 4 {
		idle += vals[4]
	}
	return total - idle, total, true
}

// sessionLocked asks systemd-logind for the LockedHint of graphical sessions.
// Headless/Docker deployments (no loginctl, no graphical session) report
// Unknown and the scheduler falls back to its other signals.
func sessionLocked() Tri {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "loginctl", "list-sessions", "--no-legend").Output()
	if err != nil {
		return Unknown
	}
	anyGraphical := false
	allLocked := true
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		seat := fields[3]
		if seat == "" || seat == "-" {
			continue // not a seated (graphical) session
		}
		anyGraphical = true
		hint, err := exec.CommandContext(ctx, "loginctl", "show-session", fields[0], "--property=LockedHint", "--value").Output()
		if err != nil {
			return Unknown
		}
		if strings.TrimSpace(string(hint)) != "yes" {
			allLocked = false
		}
	}
	if !anyGraphical {
		return Unknown
	}
	if allLocked {
		return Yes
	}
	return No
}

// fullscreenBusy has no portable answer on Linux (compositor-specific).
func fullscreenBusy() Tri { return Unknown }

// onBattery reads /sys/class/power_supply: on AC when any Mains supply is
// online; on battery when Mains supplies exist and none are online.
func onBattery() Tri {
	entries, err := os.ReadDir("/sys/class/power_supply")
	if err != nil || len(entries) == 0 {
		return Unknown
	}
	sawMains := false
	for _, e := range entries {
		base := filepath.Join("/sys/class/power_supply", e.Name())
		t, err := os.ReadFile(filepath.Join(base, "type"))
		if err != nil || strings.TrimSpace(string(t)) != "Mains" {
			continue
		}
		sawMains = true
		online, err := os.ReadFile(filepath.Join(base, "online"))
		if err == nil && strings.TrimSpace(string(online)) == "1" {
			return No // plugged in
		}
	}
	if sawMains {
		return Yes
	}
	return Unknown // desktop without a PSU sensor — effectively AC
}

func inputIdleSeconds() float64 { return -1 }
