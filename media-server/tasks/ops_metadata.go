package tasks

// ops_metadata.go — the former "metadata" task's subtasks, each broken out as
// a standalone ItemOp: describe, transcribe, hash, dimensions, llm-autotag.
// Each runs as its own task and can be combined with any other op into a
// single per-file pass (see the "process" task and runItemOps).

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
)

var (
	imageExts = []string{".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic"}
	videoExts = []string{".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"}
)

// registerBuiltinItemOps installs the built-in per-item operations. Called at
// the top of the registry init so task registration can reference the ops.
func registerBuiltinItemOps() {
	RegisterItemOp(ItemOp{
		ID:   "describe",
		Name: "Description (LLM Vision)",
		Options: []TaskOption{
			{Name: "model", Label: "Vision Model", Type: "string", Description: "Override the configured vision model for descriptions"},
			{Name: "prompt", Label: "Custom Prompt", Type: "string", Description: "Override the configured describe prompt for this run"},
		},
		Applies: extAppliesFn(append(append([]string{}, imageExts...), videoExts...)...),
		Prepare: prepareDescribeOp,
	})

	RegisterItemOp(ItemOp{
		ID:      "transcribe",
		Name:    "Transcript",
		Applies: extAppliesFn(videoExts...),
		Prepare: prepareTranscribeOp,
	})

	RegisterItemOp(ItemOp{
		ID:          "hash",
		Name:        "Hash + Size",
		Concurrency: func() int { return 4 },
		Prepare:     prepareHashOp,
	})

	RegisterItemOp(ItemOp{
		ID:          "dimensions",
		Name:        "Dimensions",
		Concurrency: func() int { return 4 },
		Applies:     extAppliesFn(append(append([]string{}, imageExts...), ".mp4", ".mov", ".avi", ".mkv", ".webm")...),
		Prepare:     prepareDimensionsOp,
	})

	RegisterItemOp(ItemOp{
		ID:   "llm-autotag",
		Name: "Auto Tag (LLM Vision)",
		Options: []TaskOption{
			{Name: "model", Label: "Vision Model", Type: "string", Description: "Override the configured vision model for tag selection"},
		},
		Applies: extAppliesFn(imageExts...),
		Prepare: prepareLLMAutotagOp,
	})

	registerEmbedItemOp()
	registerAutotagItemOp()
	registerFacesItemOp()
}

func prepareDescribeOp(run *ItemRun) (*ItemProcessor, error) {
	q, jobID := run.Queue, run.Job.ID
	model, _ := run.Opts["model"].(string)
	if model == "" {
		model = appconfig.Get().OllamaModel
	}
	prompt, _ := run.Opts["prompt"].(string)
	db := q.Db

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingMetadata(db, path, "description") },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			description, err := describeFileWithOllama(ctx, q, jobID, path, model, prompt)
			if err != nil {
				return nil, err
			}
			return &ItemCommit{
				Commit: func() error {
					if err := updateMediaMetadata(db, path, "description", description); err != nil {
						return err
					}
					notifyProgress(ProgressDescription, 1)
					return nil
				},
				Detail: "description generated",
			}, nil
		},
	}, nil
}

func prepareTranscribeOp(run *ItemRun) (*ItemProcessor, error) {
	q, jobID := run.Queue, run.Job.ID
	db := q.Db

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingMetadata(db, path, "transcript") },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			transcript, err := generateTranscriptWithFasterWhisper(ctx, q, jobID, path)
			if err != nil {
				return nil, err
			}
			return &ItemCommit{
				Commit: func() error {
					if err := updateMediaMetadata(db, path, "transcript", transcript); err != nil {
						return err
					}
					notifyProgress(ProgressTranscript, 1)
					return nil
				},
				Detail: "transcript generated",
			}, nil
		},
	}, nil
}

func prepareHashOp(run *ItemRun) (*ItemProcessor, error) {
	const maxBytes = 3 * 1024 * 1024
	db := run.Queue.Db

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingMetadata(db, path, "hash") },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			fi, err := os.Stat(path)
			if err != nil {
				return nil, fmt.Errorf("stat: %w", err)
			}
			file, err := os.Open(path)
			if err != nil {
				return nil, fmt.Errorf("open: %w", err)
			}
			hashVal, err := hashFirstNBytes(file, maxBytes)
			file.Close()
			if err != nil {
				return nil, fmt.Errorf("hash: %w", err)
			}
			size := fi.Size()
			return &ItemCommit{
				Commit: func() error {
					if _, err := db.Exec(`UPDATE media SET hash = ?, size = ? WHERE path = ?`, hashVal, size, path); err != nil {
						return err
					}
					// One UPDATE sets both columns, so both coverage counters advance.
					notifyProgress(ProgressHash, 1)
					notifyProgress(ProgressSize, 1)
					return nil
				},
				Detail: "hash generated",
			}, nil
		},
	}, nil
}

func prepareDimensionsOp(run *ItemRun) (*ItemProcessor, error) {
	db := run.Queue.Db
	isImage := extAppliesFn(imageExts...)

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasExistingDimensions(db, path) },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			var width, height int
			var err error
			if isImage(path) {
				width, height, err = getImageDimensions(path)
			} else {
				width, height, err = getVideoDimensionsFFProbe(path)
			}
			if err != nil {
				return nil, fmt.Errorf("get dimensions: %w", err)
			}
			return &ItemCommit{
				Commit: func() error {
					if _, err := db.Exec(`UPDATE media SET width = ?, height = ? WHERE path = ?`, width, height, path); err != nil {
						return err
					}
					notifyProgress(ProgressDimensions, 1)
					return nil
				},
				Detail: fmt.Sprintf("dimensions %dx%d", width, height),
			}, nil
		},
	}, nil
}

func prepareLLMAutotagOp(run *ItemRun) (*ItemProcessor, error) {
	q := run.Queue
	db := q.Db
	model, _ := run.Opts["model"].(string)
	if model == "" {
		model = appconfig.Get().OllamaModel
	}
	overwrite := run.Overwrite

	availableTags, err := getAllAvailableTags(db)
	if err != nil {
		return nil, fmt.Errorf("fetch available tags: %w", err)
	}
	if len(availableTags) == 0 {
		return nil, fmt.Errorf("no tags exist in the library to select from - create tags first")
	}
	q.PushJobStdout(run.Job.ID, fmt.Sprintf("llm-autotag: selecting from %d available tags", len(availableTags)))

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) {
			existing, err := getExistingTagsForFile(db, path)
			if err != nil {
				return false, err
			}
			return len(existing) > 0, nil
		},
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			selected, err := generateAutoTagsWithVision(ctx, path, availableTags, model)
			if err != nil {
				return nil, fmt.Errorf("auto-tag: %w", err)
			}
			if len(selected) == 0 {
				return nil, nil // model picked nothing — not an error, nothing to write
			}
			labels := make([]string, len(selected))
			for i, t := range selected {
				labels[i] = t.Label
			}
			return &ItemCommit{
				Commit: func() error {
					if overwrite {
						if err := removeExistingTagsForFile(db, path); err != nil {
							return fmt.Errorf("remove existing tags: %w", err)
						}
					}
					return insertTagsForFile(db, path, selected)
				},
				Detail: "tags: " + strings.Join(labels, ", "),
			}, nil
		},
	}, nil
}
