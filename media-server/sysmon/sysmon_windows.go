//go:build windows

package sysmon

import (
	"syscall"
	"unsafe"
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	user32   = syscall.NewLazyDLL("user32.dll")
	shell32  = syscall.NewLazyDLL("shell32.dll")
	wtsapi32 = syscall.NewLazyDLL("wtsapi32.dll")

	procGetSystemTimes               = kernel32.NewProc("GetSystemTimes")
	procGetTickCount                 = kernel32.NewProc("GetTickCount")
	procGetSystemPowerStatus         = kernel32.NewProc("GetSystemPowerStatus")
	procGetLastInputInfo             = user32.NewProc("GetLastInputInfo")
	procSHQueryUserNotificationState = shell32.NewProc("SHQueryUserNotificationState")
	procWTSQuerySessionInformationW  = wtsapi32.NewProc("WTSQuerySessionInformationW")
	procWTSFreeMemory                = wtsapi32.NewProc("WTSFreeMemory")
)

type filetime struct {
	Low  uint32
	High uint32
}

func (f filetime) f64() float64 {
	return float64(uint64(f.High)<<32 | uint64(f.Low))
}

// cpuTimes returns cumulative busy and total CPU time (100ns units across
// all cores). GetSystemTimes' kernel time INCLUDES idle time.
func cpuTimes() (busy, total float64, ok bool) {
	var idle, kernel, user filetime
	r, _, _ := procGetSystemTimes.Call(
		uintptr(unsafe.Pointer(&idle)),
		uintptr(unsafe.Pointer(&kernel)),
		uintptr(unsafe.Pointer(&user)),
	)
	if r == 0 {
		return 0, 0, false
	}
	i, k, u := idle.f64(), kernel.f64(), user.f64()
	return (k - i) + u, k + u, true
}

const (
	wtsCurrentSession     = 0xFFFFFFFF
	wtsSessionInfoEx      = 25 // WTS_INFO_CLASS: WTSSessionInfoEx
	wtsSessionStateLock   = 0
	wtsSessionStateUnlock = 1
)

// sessionLocked reads the current session's lock flag via WTSSessionInfoEx.
// Win8+ semantics: SessionFlags 0 = locked, 1 = unlocked. (Windows 7 had
// these inverted; we don't support Win7.)
func sessionLocked() Tri {
	var buf *byte
	var n uint32
	// WTS_CURRENT_SERVER_HANDLE == 0
	r, _, _ := procWTSQuerySessionInformationW.Call(
		0,
		uintptr(wtsCurrentSession),
		uintptr(wtsSessionInfoEx),
		uintptr(unsafe.Pointer(&buf)),
		uintptr(unsafe.Pointer(&n)),
	)
	if r == 0 || buf == nil {
		return Unknown
	}
	defer procWTSFreeMemory.Call(uintptr(unsafe.Pointer(buf)))
	// WTSINFOEXW layout: DWORD Level; then WTSINFOEX_LEVEL1_W
	// { DWORD SessionId; DWORD SessionState; LONG SessionFlags; ... }.
	// SessionFlags therefore sits at byte offset 12.
	if n < 16 {
		return Unknown
	}
	flags := *(*int32)(unsafe.Add(unsafe.Pointer(buf), 12))
	switch flags {
	case wtsSessionStateLock:
		return Yes
	case wtsSessionStateUnlock:
		return No
	default:
		return Unknown
	}
}

// QUERY_USER_NOTIFICATION_STATE values that mean "the user's screen is
// owned by something important" (game, presentation, blocking app).
const (
	qunsBusy                 = 2
	qunsRunningD3DFullScreen = 3
	qunsPresentationMode     = 4
)

func fullscreenBusy() Tri {
	var state int32
	r, _, _ := procSHQueryUserNotificationState.Call(uintptr(unsafe.Pointer(&state)))
	if r != 0 { // non-S_OK HRESULT
		return Unknown
	}
	switch state {
	case qunsBusy, qunsRunningD3DFullScreen, qunsPresentationMode:
		return Yes
	default:
		return No
	}
}

type systemPowerStatus struct {
	ACLineStatus        byte
	BatteryFlag         byte
	BatteryLifePercent  byte
	SystemStatusFlag    byte
	BatteryLifeTime     uint32
	BatteryFullLifeTime uint32
}

func onBattery() Tri {
	var st systemPowerStatus
	r, _, _ := procGetSystemPowerStatus.Call(uintptr(unsafe.Pointer(&st)))
	if r == 0 {
		return Unknown
	}
	switch st.ACLineStatus {
	case 0:
		return Yes
	case 1:
		return No
	default:
		return Unknown
	}
}

type lastInputInfo struct {
	CbSize uint32
	DwTime uint32
}

// inputIdleSeconds only works in an interactive session (the tray build);
// running as a service reports Unknown via the failed call.
func inputIdleSeconds() float64 {
	li := lastInputInfo{CbSize: 8}
	r, _, _ := procGetLastInputInfo.Call(uintptr(unsafe.Pointer(&li)))
	if r == 0 {
		return -1
	}
	tick, _, _ := procGetTickCount.Call()
	// Both are 32-bit tick counters; unsigned subtraction handles wraparound.
	return float64(uint32(tick)-li.DwTime) / 1000.0
}
