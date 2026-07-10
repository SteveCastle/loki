// Package sysmon samples machine-level signals the auto-process scheduler
// uses to decide when the computer is free for background work: total CPU
// load, session lock state, fullscreen/game detection, battery state, and
// user input idleness.
//
// Every signal is best-effort tri-state: platforms (or deployments — Docker,
// headless) that can't answer report Unknown and the scheduler's policy falls
// back to the signals that ARE available. Windows has the full set; Linux has
// CPU + lock (systemd-logind) + battery; macOS is intentionally unsupported
// for now (no cgo).
package sysmon

import (
	"sync"
	"time"
)

// Tri is a three-valued boolean for best-effort signals.
type Tri int

const (
	Unknown Tri = iota
	No
	Yes
)

func (t Tri) String() string {
	switch t {
	case Yes:
		return "yes"
	case No:
		return "no"
	default:
		return "unknown"
	}
}

// MarshalJSON serializes a Tri as "yes"/"no"/"unknown".
func (t Tri) MarshalJSON() ([]byte, error) {
	return []byte(`"` + t.String() + `"`), nil
}

// State is one snapshot of the machine-level signals.
type State struct {
	// CPUPercent is total system CPU utilization (all processes, ours
	// included) averaged over the window since the previous Sample call.
	// -1 when unknown or on the first sample.
	CPUPercent float64 `json:"cpuPercent"`
	// SessionLocked reports whether the interactive session is locked.
	SessionLocked Tri `json:"sessionLocked"`
	// FullscreenBusy reports whether a fullscreen D3D app / presentation is
	// in the foreground (Windows only) — the "user is gaming" signal.
	FullscreenBusy Tri `json:"fullscreenBusy"`
	// OnBattery reports whether the machine is running on battery power.
	OnBattery Tri `json:"onBattery"`
	// InputIdleSeconds is the time since the last keyboard/mouse input in the
	// interactive session. -1 when unknown.
	InputIdleSeconds float64 `json:"inputIdleSeconds"`

	SampledAt time.Time `json:"sampledAt"`
}

// Monitor computes CPU utilization between successive Sample calls and
// gathers the point-in-time signals.
type Monitor struct {
	mu        sync.Mutex
	prevBusy  float64
	prevTotal float64
	havePrev  bool
	last      State
}

func NewMonitor() *Monitor {
	return &Monitor{last: State{CPUPercent: -1, InputIdleSeconds: -1}}
}

// Sample takes a fresh reading. CPU utilization is averaged over the window
// since the previous call, so callers should sample on a steady cadence.
func (m *Monitor) Sample() State {
	m.mu.Lock()
	defer m.mu.Unlock()

	s := State{
		CPUPercent:       -1,
		SessionLocked:    sessionLocked(),
		FullscreenBusy:   fullscreenBusy(),
		OnBattery:        onBattery(),
		InputIdleSeconds: inputIdleSeconds(),
		SampledAt:        time.Now(),
	}

	if busy, total, ok := cpuTimes(); ok {
		if m.havePrev && total > m.prevTotal {
			s.CPUPercent = 100 * (busy - m.prevBusy) / (total - m.prevTotal)
			if s.CPUPercent < 0 {
				s.CPUPercent = 0
			}
			if s.CPUPercent > 100 {
				s.CPUPercent = 100
			}
		}
		m.prevBusy, m.prevTotal, m.havePrev = busy, total, true
	}

	m.last = s
	return s
}

// Latest returns the most recent sample without taking a new one.
func (m *Monitor) Latest() State {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.last
}
