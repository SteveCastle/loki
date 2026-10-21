package main

import (
	"context"
	"os"
	"path/filepath"
	"sync"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/heic"
	"github.com/stevecastle/shrike/platform"
)

// heicDisplayCacheBytes caps the JPEG renderings of HEIC files kept for
// browsers, which can't decode HEIC themselves. A 12 MP iPhone photo renders
// to ~1.5–3.5 MB.
const heicDisplayCacheBytes = 2 << 30

var heicDisplayCache = sync.OnceValue(func() *heic.Cache {
	return heic.NewCache(
		filepath.Join(platform.GetDataDir(), "heic-display"),
		heicDisplayCacheBytes,
		func() string { return depspkg.BundledOrEmpty("ffmpeg") },
	)
})

// heicDisplayFile returns the cached JPEG rendering of a local HEIC file and
// its FileInfo, converting it on first use. /media/file serves it in place of
// the original and image thumbnails are scaled from it.
func heicDisplayFile(ctx context.Context, filePath string, info os.FileInfo) (string, os.FileInfo, error) {
	out, err := heicDisplayCache().Path(ctx, filePath, info)
	if err != nil {
		return "", nil, err
	}
	outInfo, err := os.Stat(out)
	if err != nil {
		return "", nil, err
	}
	return out, outInfo, nil
}
