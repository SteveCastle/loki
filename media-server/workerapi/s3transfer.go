package workerapi

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// S3Config is an S3-compatible bucket used to hand files to serverless
// workers (AWS S3, Cloudflare R2, Backblaze B2, MinIO with a public
// endpoint...). The workers only ever see presigned URLs.
type S3Config struct {
	Endpoint  string `json:"endpoint,omitempty"`
	Region    string `json:"region,omitempty"`
	Bucket    string `json:"bucket"`
	Prefix    string `json:"prefix,omitempty"`
	AccessKey string `json:"accessKey,omitempty"`
	SecretKey string `json:"secretKey,omitempty"`
}

// S3Transfer is a Transfer over one bucket.
type S3Transfer struct {
	client  *s3.Client
	presign *s3.PresignClient
	bucket  string
	prefix  string
	// TTL of the presigned URLs: long enough for a queued serverless job to
	// start and finish.
	TTL time.Duration
}

// NewS3Transfer connects to the bucket (no network call yet).
func NewS3Transfer(ctx context.Context, c S3Config) (*S3Transfer, error) {
	if strings.TrimSpace(c.Bucket) == "" {
		return nil, fmt.Errorf("transfer bucket: bucket is required")
	}
	region := c.Region
	if region == "" {
		region = "us-east-1"
	}
	opts := []func(*config.LoadOptions) error{config.WithRegion(region)}
	if c.AccessKey != "" {
		opts = append(opts, config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(c.AccessKey, c.SecretKey, "")))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("transfer bucket: %w", err)
	}
	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		if c.Endpoint != "" {
			o.BaseEndpoint = aws.String(c.Endpoint)
			o.UsePathStyle = true
		}
	})
	return &S3Transfer{client: client, presign: s3.NewPresignClient(client), bucket: c.Bucket, prefix: strings.Trim(c.Prefix, "/"), TTL: 6 * time.Hour}, nil
}

func (t *S3Transfer) key(k string) string {
	if t.prefix == "" {
		return k
	}
	return path.Join(t.prefix, k)
}

// Upload implements Transfer.
func (t *S3Transfer) Upload(ctx context.Context, key, local string) (string, error) {
	f, err := os.Open(local)
	if err != nil {
		return "", err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return "", err
	}
	if _, err := t.client.PutObject(ctx, &s3.PutObjectInput{Bucket: aws.String(t.bucket), Key: aws.String(t.key(key)), Body: f, ContentLength: aws.Int64(st.Size())}); err != nil {
		return "", err
	}
	req, err := t.presign.PresignGetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(t.bucket), Key: aws.String(t.key(key))}, s3.WithPresignExpires(t.TTL))
	if err != nil {
		return "", err
	}
	return req.URL, nil
}

// PresignPut implements Transfer.
func (t *S3Transfer) PresignPut(ctx context.Context, key string) (string, error) {
	req, err := t.presign.PresignPutObject(ctx, &s3.PutObjectInput{Bucket: aws.String(t.bucket), Key: aws.String(t.key(key))}, s3.WithPresignExpires(t.TTL))
	if err != nil {
		return "", err
	}
	return req.URL, nil
}

// Fetch implements Transfer.
func (t *S3Transfer) Fetch(ctx context.Context, key string, fn func(io.Reader) error) error {
	out, err := t.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(t.bucket), Key: aws.String(t.key(key))})
	if err != nil {
		return err
	}
	defer out.Body.Close()
	return fn(out.Body)
}

// Delete implements Transfer.
func (t *S3Transfer) Delete(ctx context.Context, keys []string) {
	for len(keys) > 0 {
		n := min(len(keys), 1000)
		var objs []types.ObjectIdentifier
		for _, k := range keys[:n] {
			objs = append(objs, types.ObjectIdentifier{Key: aws.String(t.key(k))})
		}
		_, _ = t.client.DeleteObjects(ctx, &s3.DeleteObjectsInput{Bucket: aws.String(t.bucket), Delete: &types.Delete{Objects: objs, Quiet: aws.Bool(true)}})
		keys = keys[n:]
	}
}

// Check verifies the bucket accepts a write, a presigned read and a delete.
func (t *S3Transfer) Check(ctx context.Context) error {
	key := "loki-worker/.check-" + randHex(4)
	if _, err := t.client.PutObject(ctx, &s3.PutObjectInput{Bucket: aws.String(t.bucket), Key: aws.String(t.key(key)), Body: strings.NewReader("ok")}); err != nil {
		return fmt.Errorf("transfer bucket: writing: %w", err)
	}
	t.Delete(ctx, []string{key})
	return nil
}
