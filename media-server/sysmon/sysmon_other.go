//go:build !windows && !linux

package sysmon

// macOS (and anything else) is intentionally unsupported for now — lock and
// input state there need a cgo shim we don't want yet. Every signal reports
// Unknown and the scheduler policy falls back to app-level activity.

func cpuTimes() (busy, total float64, ok bool) { return 0, 0, false }
func sessionLocked() Tri                       { return Unknown }
func fullscreenBusy() Tri                      { return Unknown }
func onBattery() Tri                           { return Unknown }
func inputIdleSeconds() float64                { return -1 }
