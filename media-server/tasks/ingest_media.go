package tasks

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

// Native media extraction — per-site extractors that download media in Go
// with no gallery-dl/yt-dlp dependency. The generic ingest engine here owns
// the whole pipeline (resolve → download → stage/upload → database insert →
// tags → follow-up tasks); an extractor only supplies the site-specific
// parts: which URLs it handles, how a listing page expands into item pages,
// and how an item page yields a downloadable file.
//
// Extractors are not compiled in — they load at runtime from a user-supplied
// definitions file (see extractors_dsl.go).

// mediaItem is one downloadable file resolved by an extractor.
type mediaItem struct {
	MediaURL string // direct URL of the file to download
	Title    string // display title for logs ("" if unknown)
	SubPath  string // slash-separated path under the extractor's folder, no extension (e.g. "user/slug")
	Ext      string // fallback extension including the dot, used when MediaURL's path has none
}

// mediaExtractor is a native extractor for one media hosting site.
type mediaExtractor interface {
	// Name labels log lines and is the top-level folder downloads land in.
	Name() string
	// Match reports whether this extractor handles the URL.
	Match(url string) bool
	// Resolve expands the input URL into individual item-page URLs: an
	// item page resolves to itself, a profile/listing page is scraped for
	// the items it links.
	Resolve(ctx context.Context, url string) ([]string, error)
	// Extract fetches one item page and returns the file to download.
	Extract(ctx context.Context, itemURL string) (mediaItem, error)
}

const mediaIngestUserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LowkeyMediaServer"

// mediaIngestClient has no overall timeout — media downloads can be long and
// are bounded by the job context instead.
var mediaIngestClient = &http.Client{}

// fetchExtractorPage fetches an HTML page with a bounded timeout and size,
// for extractors scraping item and listing pages.
func fetchExtractorPage(ctx context.Context, pageURL string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pageURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", mediaIngestUserAgent)

	resp, err := mediaIngestClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch %s: HTTP %d", pageURL, resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// ingestMediaTaskWithOptions downloads media natively in Go through the given
// extractor and adds it to the database. An item URL downloads one file; a
// listing URL downloads every item it resolves to.
func ingestMediaTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, _ *sync.Mutex, opts IngestOptions, ext mediaExtractor) error {
	ctx := j.Ctx
	input := strings.TrimSpace(j.Input)

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	itemURLs, err := ext.Resolve(ctx, input)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving %s URL: %v", ext.Name(), err))
		q.ErrorJob(j.ID)
		return err
	}
	if len(itemURLs) == 0 {
		q.PushJobStdout(j.ID, "No media found at URL")
		q.CompleteJob(j.ID)
		return nil
	}

	target, err := resolveIngestDir(j.ID, "downloads/")
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error resolving download directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	defer target.cleanup()

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting %s download: %s (%d item(s))", ext.Name(), input, len(itemURLs)))
	if target.direct {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download directory (direct): %s", target.dir))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Staging directory: %s", target.dir))
	}

	var downloadedFiles []string
	var lastErr error
	for _, iu := range itemURLs {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		localPath, skipped, err := downloadMediaItem(ctx, q, j.ID, ext, iu, target.dir)
		if err != nil {
			if ctx.Err() != nil {
				q.PushJobStdout(j.ID, "Task was canceled")
				_ = q.CancelJob(j.ID)
				return ctx.Err()
			}
			lastErr = err
			q.PushJobStdout(j.ID, fmt.Sprintf("Error downloading %s: %v", iu, err))
			continue
		}
		if skipped {
			q.PushJobStdout(j.ID, fmt.Sprintf("# %s", localPath))
			continue
		}
		downloadedFiles = append(downloadedFiles, localPath)
		q.PushJobStdout(j.ID, localPath)
	}

	// Like the gallery handler: on failure, still ingest whatever downloaded
	// before the error, then surface the failure by erroring the job.
	if lastErr != nil && len(downloadedFiles) == 0 {
		q.ErrorJob(j.ID)
		return lastErr
	}
	if lastErr != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Continuing to ingest %d file(s) downloaded before the failure", len(downloadedFiles)))
	}

	var finalFiles []storedFile
	if target.direct {
		finalFiles = localStoredFiles(downloadedFiles)
	} else {
		finalFiles = uploadStagedFiles(ctx, q, j.ID, downloadedFiles, target.dir, "downloads/")
	}

	var insertedFiles []string
	for _, f := range finalFiles {
		if err := insertMediaRecord(q.Db, f.Path, f.Size); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", f.Path, err))
			continue
		}
		insertedFiles = append(insertedFiles, f.Path)
		q.PushJobStdout(j.ID, fmt.Sprintf("Added to database: %s", f.Path))
		q.RegisterOutputFile(j.ID, f.Path)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Download completed: %d files added to database", len(insertedFiles)))

	if len(opts.Tags) > 0 {
		applyIngestTags(q.Db, j.ID, q, insertedFiles, opts.Tags)
	}

	queueFollowUpTasks(q, j.ID, insertedFiles, opts)

	if lastErr != nil {
		q.ErrorJob(j.ID)
		return lastErr
	}

	q.CompleteJob(j.ID)
	return nil
}

// downloadMediaItem extracts one item and downloads its file under baseDir as
// <extractor>/<subpath>.<ext>. Returns the local path, whether the file
// already existed (skipped), and any error.
func downloadMediaItem(ctx context.Context, q *jobqueue.Queue, jobID string, ext mediaExtractor, itemURL, baseDir string) (string, bool, error) {
	item, err := ext.Extract(ctx, itemURL)
	if err != nil {
		return "", false, fmt.Errorf("%s: %w", itemURL, err)
	}
	if item.Title != "" {
		q.PushJobStdout(jobID, fmt.Sprintf("Found media: %s", item.Title))
	}

	fileExt := item.Ext
	if u, err := url.Parse(item.MediaURL); err == nil {
		if e := path.Ext(u.Path); e != "" {
			fileExt = e
		}
	}

	localPath := filepath.Join(baseDir, ext.Name(), filepath.FromSlash(item.SubPath)+fileExt)
	if _, err := os.Stat(localPath); err == nil {
		return localPath, true, nil
	}
	if err := os.MkdirAll(filepath.Dir(localPath), 0755); err != nil {
		return "", false, fmt.Errorf("create download dir: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, item.MediaURL, nil)
	if err != nil {
		return "", false, err
	}
	req.Header.Set("User-Agent", mediaIngestUserAgent)
	req.Header.Set("Referer", itemURL)

	resp, err := mediaIngestClient.Do(req)
	if err != nil {
		return "", false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false, fmt.Errorf("download %s: HTTP %d", item.MediaURL, resp.StatusCode)
	}

	// Write to a .part file first so an interrupted download is never mistaken
	// for a complete file on a later run.
	partPath := localPath + ".part"
	f, err := os.Create(partPath)
	if err != nil {
		return "", false, err
	}
	_, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(partPath)
		if copyErr != nil {
			return "", false, fmt.Errorf("download %s: %w", item.MediaURL, copyErr)
		}
		return "", false, closeErr
	}
	if err := os.Rename(partPath, localPath); err != nil {
		_ = os.Remove(partPath)
		return "", false, err
	}

	return localPath, false, nil
}
