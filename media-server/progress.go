package main

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/mattn/go-isatty"
)

// indexProgressFn returns a tasks.IndexProgress callback that renders a colorful
// in-place progress bar to stderr while the embedding index builds at startup.
// On a non-TTY (logs piped to a file/service) it falls back to plain percentage
// lines so the output stays readable.
func indexProgressFn() func(done, total int) {
	fd := os.Stderr.Fd()
	tty := isatty.IsTerminal(fd) || isatty.IsCygwinTerminal(fd)
	if tty {
		enableVirtualTerminal() // Windows: make ANSI escape codes render
	}

	var (
		mu       sync.Mutex
		started  time.Time
		lastDraw time.Time
		lastPct  = -1
	)
	const barWidth = 32

	return func(done, total int) {
		mu.Lock()
		defer mu.Unlock()
		if started.IsZero() {
			started = time.Now()
		}
		if total <= 0 {
			return // nothing to index
		}
		final := done >= total
		pct := done * 100 / total

		if !tty {
			// Plain output: one line per 10% bucket (and at completion).
			if pct/10 != lastPct/10 || final {
				lastPct = pct
				fmt.Fprintf(os.Stderr, "Building search index: %d%% (%s/%s)\n",
					pct, humanInt(done), humanInt(total))
			}
			return
		}

		now := time.Now()
		if !final && now.Sub(lastDraw) < 40*time.Millisecond {
			return // throttle redraws to ~25 fps
		}
		lastDraw = now
		drawIndexBar(done, total, barWidth, started)
		if final {
			fmt.Fprint(os.Stderr, "\n")
		}
	}
}

// drawIndexBar renders one frame of the bar in place (CR + clear line). The
// filled portion is painted with a bright neon gradient.
func drawIndexBar(done, total, width int, started time.Time) {
	frac := float64(done) / float64(total)
	if frac > 1 {
		frac = 1
	}
	filled := int(frac * float64(width))

	var b strings.Builder
	b.WriteString("\r\x1b[2K") // carriage return + erase line
	b.WriteString("\x1b[1m🔧 Building index\x1b[0m ")
	b.WriteString("\x1b[38;2;90;90;110m▕\x1b[0m")
	for i := 0; i < width; i++ {
		if i < filled {
			r, g, bl := neonGradient(float64(i) / float64(width-1))
			fmt.Fprintf(&b, "\x1b[38;2;%d;%d;%dm█", r, g, bl)
		} else {
			b.WriteString("\x1b[38;2;55;55;70m░")
		}
	}
	b.WriteString("\x1b[0m\x1b[38;2;90;90;110m▏\x1b[0m ")
	fmt.Fprintf(&b, "\x1b[1m\x1b[38;2;120;255;200m%3d%%\x1b[0m ", int(frac*100))
	fmt.Fprintf(&b, "\x1b[38;2;150;150;170m%s/%s\x1b[0m", humanInt(done), humanInt(total))

	if done > 0 && done < total {
		elapsed := time.Since(started)
		eta := time.Duration(float64(elapsed) / frac * (1 - frac))
		fmt.Fprintf(&b, " \x1b[38;2;150;150;170mETA %s\x1b[0m", eta.Round(time.Second))
	}
	fmt.Fprint(os.Stderr, b.String())
}

// neonGradient maps t∈[0,1] across a bright multi-stop palette
// (green → cyan → blue → violet → magenta) for a vivid, lively bar.
func neonGradient(t float64) (int, int, int) {
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	stops := [][3]float64{
		{57, 255, 136}, // neon green
		{0, 229, 255},  // cyan
		{64, 120, 255}, // blue
		{162, 89, 255}, // violet
		{255, 64, 200}, // magenta
	}
	seg := t * float64(len(stops)-1)
	i := int(seg)
	if i >= len(stops)-1 {
		c := stops[len(stops)-1]
		return int(c[0]), int(c[1]), int(c[2])
	}
	f := seg - float64(i)
	a, c := stops[i], stops[i+1]
	return int(a[0] + (c[0]-a[0])*f),
		int(a[1] + (c[1]-a[1])*f),
		int(a[2] + (c[2]-a[2])*f)
}

// humanInt formats n with thousands separators (e.g. 1234567 → "1,234,567").
func humanInt(n int) string {
	s := fmt.Sprintf("%d", n)
	if n < 0 {
		return "-" + humanInt(-n)
	}
	if len(s) <= 3 {
		return s
	}
	var out strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		out.WriteString(s[:pre])
		if len(s) > pre {
			out.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		out.WriteString(s[i : i+3])
		if i+3 < len(s) {
			out.WriteByte(',')
		}
	}
	return out.String()
}
