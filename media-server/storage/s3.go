package storage

import (
	"context"
	"fmt"
	"io"
	"path"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
)

// S3Config holds the configuration needed to connect to an S3-compatible store.
type S3Config struct {
	Label           string
	Endpoint        string
	Region          string
	Bucket          string
	Prefix          string
	AccessKey       string
	SecretKey       string
	ThumbnailPrefix string
}

// S3Backend serves files from an S3-compatible object store.
// Paths are represented as s3://{bucket}/{key}.
type S3Backend struct {
	client          *s3.Client
	presignClient   *s3.PresignClient
	bucket          string
	prefix          string
	label           string
	thumbnailPrefix string
}

// NewS3Backend creates an S3Backend using static credentials.
// UsePathStyle is enabled for MinIO compatibility.
// ThumbnailPrefix defaults to "_thumbnails" when not provided.
func NewS3Backend(ctx context.Context, cfg S3Config) (*S3Backend, error) {
	thumbnailPrefix := cfg.ThumbnailPrefix
	if thumbnailPrefix == "" {
		thumbnailPrefix = "_thumbnails"
	}

	opts := []func(*config.LoadOptions) error{
		config.WithRegion(cfg.Region),
		config.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(cfg.AccessKey, cfg.SecretKey, ""),
		),
	}

	awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("s3: load config: %w", err)
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		if cfg.Endpoint != "" {
			o.BaseEndpoint = aws.String(cfg.Endpoint)
		}
		o.UsePathStyle = true
	})

	return &S3Backend{
		client:          client,
		presignClient:   s3.NewPresignClient(client),
		bucket:          cfg.Bucket,
		prefix:          cfg.Prefix,
		label:           cfg.Label,
		thumbnailPrefix: thumbnailPrefix,
	}, nil
}

// pathToKey strips the s3://{bucket}/ prefix from a path to produce an S3 object key.
func (b *S3Backend) pathToKey(p string) string {
	prefix := "s3://" + b.bucket + "/"
	return strings.TrimPrefix(p, prefix)
}

// keyToPath converts an S3 object key to a full s3:// path.
func (b *S3Backend) keyToPath(key string) string {
	return "s3://" + b.bucket + "/" + key
}

// ThumbnailPath returns the s3:// path for a thumbnail file stored under thumbnailPrefix.
func (b *S3Backend) ThumbnailPath(filename string) string {
	key := b.thumbnailPrefix + "/" + filename
	return b.keyToPath(key)
}

// Root returns the Entry representing the top-level directory of this backend.
func (b *S3Backend) Root() Entry {
	rootPath := "s3://" + b.bucket + "/"
	if b.prefix != "" {
		rootPath = "s3://" + b.bucket + "/" + b.prefix
	}
	return Entry{
		Name:  b.label,
		Path:  rootPath,
		IsDir: true,
		Type:  "s3",
	}
}

// Contains reports whether p is rooted inside this backend's bucket and prefix.
func (b *S3Backend) Contains(p string) bool {
	prefix := "s3://" + b.bucket + "/"
	if b.prefix != "" {
		prefix = "s3://" + b.bucket + "/" + b.prefix
	}
	return strings.HasPrefix(p, prefix)
}

// List returns the immediate children (directories and media files) of dirPath.
// It uses ListObjectsV2 with a delimiter to get a single-level listing.
func (b *S3Backend) List(ctx context.Context, dirPath string) ([]Entry, error) {
	key := b.pathToKey(dirPath)
	// Ensure the key ends with "/" to list the directory contents.
	if key != "" && !strings.HasSuffix(key, "/") {
		key += "/"
	}

	var entries []Entry
	paginator := s3.NewListObjectsV2Paginator(b.client, &s3.ListObjectsV2Input{
		Bucket:    aws.String(b.bucket),
		Prefix:    aws.String(key),
		Delimiter: aws.String("/"),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("s3: list %q: %w", dirPath, err)
		}

		// Common prefixes are "subdirectories".
		for _, cp := range page.CommonPrefixes {
			if cp.Prefix == nil {
				continue
			}
			dirKey := strings.TrimSuffix(*cp.Prefix, "/")
			name := path.Base(dirKey)
			entries = append(entries, Entry{
				Name:  name,
				Path:  b.keyToPath(*cp.Prefix),
				IsDir: true,
				Type:  "s3",
			})
		}

		// Objects are files.
		for _, obj := range page.Contents {
			if obj.Key == nil {
				continue
			}
			// Skip the directory placeholder itself.
			if *obj.Key == key {
				continue
			}
			name := path.Base(*obj.Key)
			if !IsMediaFile(name) {
				continue
			}
			var mtimeMs float64
			if obj.LastModified != nil {
				mtimeMs = float64(obj.LastModified.UnixMilli())
			}
			entries = append(entries, Entry{
				Name:    name,
				Path:    b.keyToPath(*obj.Key),
				IsDir:   false,
				MtimeMs: mtimeMs,
				Type:    "s3",
			})
		}
	}

	return entries, nil
}

// Scan returns media files under dirPath.
// When recursive is true it lists all objects without a delimiter.
// When recursive is false it uses a delimiter to stay at a single level.
func (b *S3Backend) Scan(ctx context.Context, dirPath string, recursive bool) ([]FileInfo, error) {
	key := b.pathToKey(dirPath)
	if key != "" && !strings.HasSuffix(key, "/") {
		key += "/"
	}

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(b.bucket),
		Prefix: aws.String(key),
	}
	if !recursive {
		input.Delimiter = aws.String("/")
	}

	var files []FileInfo
	paginator := s3.NewListObjectsV2Paginator(b.client, input)

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("s3: scan %q: %w", dirPath, err)
		}

		for _, obj := range page.Contents {
			if obj.Key == nil {
				continue
			}
			if *obj.Key == key {
				continue
			}
			name := path.Base(*obj.Key)
			if !IsMediaFile(name) {
				continue
			}
			var mtimeMs float64
			if obj.LastModified != nil {
				mtimeMs = float64(obj.LastModified.UnixMilli())
			}
			files = append(files, FileInfo{
				Path:    b.keyToPath(*obj.Key),
				MtimeMs: mtimeMs,
			})
		}
	}

	return files, nil
}

// Download opens p for reading. The caller must close the returned reader.
func (b *S3Backend) Download(ctx context.Context, p string) (io.ReadCloser, error) {
	key := b.pathToKey(p)
	out, err := b.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("s3: download %q: %w", p, err)
	}
	return out.Body, nil
}

// Upload writes r to p with the given content type.
func (b *S3Backend) Upload(ctx context.Context, p string, r io.Reader, contentType string) error {
	key := b.pathToKey(p)
	_, err := b.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(b.bucket),
		Key:         aws.String(key),
		Body:        r,
		ContentType: aws.String(contentType),
	})
	if err != nil {
		return fmt.Errorf("s3: upload %q: %w", p, err)
	}
	return nil
}

// MediaURL returns a presigned URL for p with a 1-hour expiry.
func (b *S3Backend) MediaURL(p string) (string, error) {
	key := b.pathToKey(p)
	req, err := b.presignClient.PresignGetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	}, s3.WithPresignExpires(time.Hour))
	if err != nil {
		return "", fmt.Errorf("s3: presign %q: %w", p, err)
	}
	return req.URL, nil
}

// Exists reports whether p exists in the bucket. Returns false (not an error) for 404s.
func (b *S3Backend) Exists(ctx context.Context, p string) (bool, error) {
	key := b.pathToKey(p)
	_, err := b.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	})
	if err == nil {
		return true, nil
	}
	// Check for a NotFound API error.
	var apiErr smithy.APIError
	if ok := isAPIError(err, &apiErr); ok {
		code := apiErr.ErrorCode()
		if code == "NotFound" || code == "NoSuchKey" {
			return false, nil
		}
	}
	return false, fmt.Errorf("s3: exists %q: %w", p, err)
}

// isAPIError extracts a smithy.APIError from err if one is present.
func isAPIError(err error, target *smithy.APIError) bool {
	// Walk the error chain.
	type unwrapper interface{ Unwrap() error }
	for err != nil {
		if ae, ok := err.(smithy.APIError); ok {
			*target = ae
			return true
		}
		if u, ok := err.(unwrapper); ok {
			err = u.Unwrap()
		} else {
			break
		}
	}
	return false
}
