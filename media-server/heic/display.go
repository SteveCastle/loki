package heic

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

const (
	convertTimeout = 60 * time.Second
	pruneInterval  = 5 * time.Minute
	// A cache hit refreshes the file's mtime (its LRU recency) at most this
	// often, so the ETag /media/file derives from it stays stable.
	touchInterval = 24 * time.Hour
	// A meta box this big isn't a real photo's; don't read it into memory.
	maxMetaBytes = 64 << 20
)

// Cache converts HEIC files to JPEG on demand and keeps the results in Dir,
// keyed by source path + mtime + size, pruned least-recently-used first once
// they exceed MaxBytes.
type Cache struct {
	Dir      string
	MaxBytes int64
	// FFmpeg resolves the ffmpeg binary ("" when it isn't installed).
	FFmpeg func() string

	sem chan struct{}

	mu        sync.Mutex
	inflight  map[string]*call
	lastPrune time.Time
}

type call struct {
	done chan struct{}
	err  error
}

// NewCache returns a Cache that runs at most two conversions at once: each
// ffmpeg already decodes tiles across cores.
func NewCache(dir string, maxBytes int64, ffmpeg func() string) *Cache {
	return &Cache{
		Dir:      dir,
		MaxBytes: maxBytes,
		FFmpeg:   ffmpeg,
		sem:      make(chan struct{}, 2),
		inflight: map[string]*call{},
	}
}

// Path returns a JPEG rendering of the local HEIC file src, converting it on
// first use. info is src's FileInfo. A cancelled ctx stops the wait, not a
// conversion other requests may be sharing.
func (c *Cache) Path(ctx context.Context, src string, info os.FileInfo) (string, error) {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s|%d|%d", src, info.ModTime().UnixNano(), info.Size())))
	out := filepath.Join(c.Dir, hex.EncodeToString(sum[:])+".jpg")

	c.mu.Lock()
	cl, running := c.inflight[out]
	if !running {
		if st, err := os.Stat(out); err == nil && st.Size() > 0 {
			c.mu.Unlock()
			if time.Since(st.ModTime()) > touchInterval {
				now := time.Now()
				_ = os.Chtimes(out, now, now)
			}
			return out, nil
		}
		cl = &call{done: make(chan struct{})}
		c.inflight[out] = cl
		go func() {
			cl.err = c.convert(src, out)
			c.mu.Lock()
			delete(c.inflight, out)
			c.mu.Unlock()
			close(cl.done)
		}()
	}
	c.mu.Unlock()

	select {
	case <-cl.done:
		if cl.err != nil {
			return "", cl.err
		}
		return out, nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (c *Cache) convert(src, out string) error {
	c.sem <- struct{}{}
	defer func() { <-c.sem }()

	ffmpeg := ""
	if c.FFmpeg != nil {
		ffmpeg = c.FFmpeg()
	}
	jpeg, err := Render(ffmpeg, src)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(c.Dir, 0o755); err != nil {
		return err
	}
	var rnd [4]byte
	_, _ = rand.Read(rnd[:])
	tmp := fmt.Sprintf("%s.%d.%s.tmp", out, os.Getpid(), hex.EncodeToString(rnd[:]))
	if err := os.WriteFile(tmp, jpeg, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, out); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	c.maybePrune()
	return nil
}

// Render decodes the HEIC file src with ffmpeg and returns it as a JPEG with
// the primary image's ICC profile embedded. It doesn't cache: use a Cache for
// library files, Render for one-off temp copies.
func Render(ffmpeg, src string) ([]byte, error) {
	if ffmpeg == "" {
		return nil, fmt.Errorf("ffmpeg not found")
	}
	ctx, cancel := context.WithTimeout(context.Background(), convertTimeout)
	defer cancel()
	// No -vf: ffmpeg builds its own filtergraph to assemble a tile grid and
	// refuses a simple filter on top of it. -pix_fmt still applies at the output.
	cmd := exec.CommandContext(ctx, ffmpeg,
		"-hide_banner", "-loglevel", "error",
		"-i", src,
		"-frames:v", "1",
		"-pix_fmt", "yuvj420p",
		"-f", "image2pipe",
		"-c:v", "mjpeg",
		"-q:v", "2",
		"-",
	)
	platform.HideSubprocessWindow(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg HEIC conversion failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	jpeg := stdout.Bytes()
	if len(jpeg) == 0 {
		return nil, fmt.Errorf("ffmpeg HEIC conversion produced no image")
	}

	// ffmpeg before 8.0 can't assemble tile grids: it silently returns a
	// single 512×512 tile, or another image in the file entirely (often the
	// depth map). Refuse rather than show the wrong picture.
	info, ok := ReadInfo(src)
	if w, h, sized := JPEGSize(jpeg); ok && sized && info.Width > 0 && !info.Cropped &&
		(w != info.Width || h != info.Height) {
		return nil, fmt.Errorf("ffmpeg decoded %dx%d but the HEIC's primary image is %dx%d; ffmpeg 8.0+ is needed to assemble HEIC tile grids",
			w, h, info.Width, info.Height)
	}
	if ok && len(info.ICC) > 0 {
		jpeg = WithJPEGICC(jpeg, info.ICC)
	}
	return jpeg, nil
}

// ReadInfo parses the primary item's properties from src. The meta box comes
// before the image data in HEIC files, so only the top of the file is read.
func ReadInfo(src string) (Info, bool) {
	f, err := os.Open(src)
	if err != nil {
		return Info{}, false
	}
	defer f.Close()
	var hdr [16]byte
	var offset int64
	for i := 0; i < 16; i++ {
		n, err := f.ReadAt(hdr[:], offset)
		if n < 8 || (err != nil && err != io.EOF) {
			return Info{}, false
		}
		size := int64(binary.BigEndian.Uint32(hdr[:4]))
		if size == 1 && n == 16 {
			size = int64(binary.BigEndian.Uint64(hdr[8:]))
		}
		if string(hdr[4:8]) == "meta" {
			if size < 16 || size > maxMetaBytes {
				return Info{}, false
			}
			meta := make([]byte, size)
			if _, err := f.ReadAt(meta, offset); err != nil && err != io.EOF {
				return Info{}, false
			}
			return ParseMeta(meta)
		}
		if size < 8 {
			return Info{}, false
		}
		offset += size
	}
	return Info{}, false
}

func (c *Cache) maybePrune() {
	c.mu.Lock()
	if time.Since(c.lastPrune) < pruneInterval {
		c.mu.Unlock()
		return
	}
	c.lastPrune = time.Now()
	c.mu.Unlock()
	go c.prune()
}

func (c *Cache) prune() {
	entries, err := os.ReadDir(c.Dir)
	if err != nil {
		return
	}
	type file struct {
		path string
		size int64
		used time.Time
	}
	var files []file
	var total int64
	for _, e := range entries {
		info, err := e.Info()
		if err != nil || info.IsDir() {
			continue
		}
		// Stray .tmp files from a crash mid-write age out like everything else.
		files = append(files, file{filepath.Join(c.Dir, e.Name()), info.Size(), info.ModTime()})
		total += info.Size()
	}
	if total <= c.MaxBytes {
		return
	}
	sort.Slice(files, func(i, j int) bool { return files[i].used.Before(files[j].used) })
	// Prune to 75% so the next few conversions don't each trigger a scan.
	for _, f := range files {
		if total <= c.MaxBytes*3/4 {
			break
		}
		if os.Remove(f.path) == nil {
			total -= f.size
		}
	}
}
