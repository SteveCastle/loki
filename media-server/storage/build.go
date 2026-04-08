package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

// BuildRegistry creates a Registry from the config's storage roots.
// Returns the registry and a slice of non-fatal errors (one per failed backend).
// Local backends never fail. Failed S3 backends are skipped with an error.
func BuildRegistry(roots []appconfig.StorageRoot) (*Registry, []error) {
	var backends []Backend
	var errs []error

	for _, root := range roots {
		switch root.Type {
		case "local", "":
			backends = append(backends, NewLocalBackend(root.Path, root.Label))
		case "s3":
			if root.Bucket == "" {
				errs = append(errs, fmt.Errorf("S3 root %q: bucket is required", root.Label))
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			b, err := NewS3Backend(ctx, S3Config{
				Label:           root.Label,
				Endpoint:        root.Endpoint,
				Region:          root.Region,
				Bucket:          root.Bucket,
				Prefix:          root.Prefix,
				AccessKey:       root.AccessKey,
				SecretKey:       root.SecretKey,
				ThumbnailPrefix: root.ThumbnailPrefix,
			})
			cancel()
			if err != nil {
				errs = append(errs, fmt.Errorf("S3 root %q: %w", root.Label, err))
				continue
			}
			backends = append(backends, b)
		default:
			errs = append(errs, fmt.Errorf("unknown storage type %q for root %q", root.Type, root.Label))
		}
	}

	return NewRegistry(backends), errs
}
