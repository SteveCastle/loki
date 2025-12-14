package tasks

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

func autotagTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("autotag: using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "autotag: no image path provided in job input or query flag")
			q.CompleteJob(j.ID)
			return nil
		}
		paths = parseInputPaths(raw)
	}

	if err := EnsureCategoryExists(q.Db, "Suggested", 0); err != nil {
		q.PushJobStdout(j.ID, "autotag: failed to ensure category: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}

	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "autotag: no valid paths parsed from input")
		q.CompleteJob(j.ID)
		return nil
	}

	for idx, mediaPath := range paths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "autotag: task canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		// Handle video files by extracting a frame first
		imagePath := mediaPath
		var tempFramePath string
		ext := strings.ToLower(filepath.Ext(mediaPath))
		isVideo := false
		switch ext {
		case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".gif":
			isVideo = true
		}

		if isVideo {
			q.PushJobStdout(j.ID, fmt.Sprintf("autotag: [%d/%d] extracting frame from video %s", idx+1, len(paths), filepath.Base(mediaPath)))
			framePath, err := extractVideoFrame(ctx, mediaPath, "")
			if err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("autotag: failed to extract frame from %s: %v", filepath.Base(mediaPath), err))
				continue
			}
			tempFramePath = framePath
			imagePath = framePath
		}

		cfg := appconfig.Get()
		args := []string{}

		// Try to get paths from dependency system first, fall back to config
		labelsPath, err := deps.GetFilePath("onnx-bundle", "selected_tags.csv")
		if err == nil && labelsPath != "" {
			args = append(args, `--labels=`+labelsPath)
		} else if strings.TrimSpace(cfg.OnnxTagger.LabelsPath) != "" {
			args = append(args, `--labels=`+cfg.OnnxTagger.LabelsPath)
		}

		configPath, err := deps.GetFilePath("onnx-bundle", "config.json")
		if err == nil && configPath != "" {
			args = append(args, `--config=`+configPath)
		} else if strings.TrimSpace(cfg.OnnxTagger.ConfigPath) != "" {
			args = append(args, `--config=`+cfg.OnnxTagger.ConfigPath)
		}

		modelPath, err := deps.GetFilePath("onnx-bundle", "model.onnx")
		if err == nil && modelPath != "" {
			args = append(args, `--model=`+modelPath)
		} else if strings.TrimSpace(cfg.OnnxTagger.ModelPath) != "" {
			args = append(args, `--model=`+cfg.OnnxTagger.ModelPath)
		}

		ortPath, err := deps.GetFilePath("onnx-bundle", "onnxruntime.dll")
		if err == nil && ortPath != "" {
			args = append(args, `--ort=`+ortPath)
		} else if strings.TrimSpace(cfg.OnnxTagger.ORTSharedLibraryPath) != "" {
			args = append(args, `--ort=`+cfg.OnnxTagger.ORTSharedLibraryPath)
		}

		if cfg.OnnxTagger.GeneralThreshold > 0 {
			args = append(args, `--general-thresh=`+fmt.Sprintf("%g", cfg.OnnxTagger.GeneralThreshold))
		}
		if cfg.OnnxTagger.CharacterThreshold > 0 {
			args = append(args, `--character-thresh=`+fmt.Sprintf("%g", cfg.OnnxTagger.CharacterThreshold))
		}
		args = append(args, `--image=`+imagePath)

		q.PushJobStdout(j.ID, fmt.Sprintf("autotag: [%d/%d] tagging %s", idx+1, len(paths), filepath.Base(mediaPath)))

		cmd, cleanup, err := embedexec.GetExec(ctx, "onnxtag", args...)
		if err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to prepare onnxtag: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		if cleanup != nil {
			defer cleanup()
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to get stdout pipe: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		stderr, err := cmd.StderrPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to get stderr pipe: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		doneErr := make(chan struct{})
		go func() {
			s := bufio.NewScanner(stderr)
			for s.Scan() {
				_ = q.PushJobStdout(j.ID, "autotag stderr: "+s.Text())
			}
			close(doneErr)
		}()

		if err := cmd.Start(); err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to start onnxtag: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		var tags []string
		scan := bufio.NewScanner(stdout)
		for scan.Scan() {
			line := strings.TrimSpace(scan.Text())
			if line != "" {
				tags = append(tags, line)
				_ = q.PushJobStdout(j.ID, "autotag: "+line)
			}
		}
		_ = cmd.Wait()
		<-doneErr

		if len(tags) == 0 {
			q.PushJobStdout(j.ID, "autotag: no tags returned")
			if tempFramePath != "" {
				_ = os.Remove(tempFramePath)
			}
			continue
		}

		var tagInfos []TagInfo
		for _, t := range tags {
			name := t
			if pos := strings.LastIndex(t, ":"); pos > 0 {
				name = strings.TrimSpace(t[:pos])
			}
			if name == "" {
				continue
			}
			tagInfos = append(tagInfos, TagInfo{Label: name, Category: "Suggested"})
		}

		if err := insertTagsForFile(q.Db, mediaPath, tagInfos); err != nil {
			q.PushJobStdout(j.ID, "autotag: failed to insert tags: "+err.Error())
			if tempFramePath != "" {
				_ = os.Remove(tempFramePath)
			}
			q.ErrorJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("autotag: wrote %d Suggested tags for %s", len(tagInfos), filepath.Base(mediaPath)))

		// Clean up temporary frame if we extracted one
		if tempFramePath != "" {
			_ = os.Remove(tempFramePath)
		}
	}

	q.CompleteJob(j.ID)
	return nil
}
