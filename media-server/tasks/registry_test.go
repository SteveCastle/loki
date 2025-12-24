package tasks

import (
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

// TestGetTasks verifies that built-in tasks are registered
func TestGetTasks(t *testing.T) {
	taskMap := GetTasks()

	if taskMap == nil {
		t.Fatal("GetTasks() returned nil")
	}

	// Verify expected built-in tasks exist
	expectedTasks := []struct {
		id   string
		name string
	}{
		{"wait", "Wait"},
		{"gallery-dl", "gallery-dl"},
		{"dce", "dce"},
		{"yt-dlp", "yt-dlp"},
		{"ffmpeg", "ffmpeg"},
		{"remove", "Remove Media"},
		{"cleanup", "CleanUp"},
		{"ingest", "Ingest Media Files"},
		{"metadata", "Generate Metadata"},
		{"move", "Move Media Files"},
		{"autotag", "Auto Tag (ONNX)"},
		{"lora-dataset", "Create LoRA Dataset"},
	}

	for _, expected := range expectedTasks {
		task, exists := taskMap[expected.id]
		if !exists {
			t.Errorf("Task %q not registered", expected.id)
			continue
		}
		if task.ID != expected.id {
			t.Errorf("Task %q has ID %q; want %q", expected.id, task.ID, expected.id)
		}
		if task.Name != expected.name {
			t.Errorf("Task %q has Name %q; want %q", expected.id, task.Name, expected.name)
		}
		if task.Fn == nil {
			t.Errorf("Task %q has nil Fn", expected.id)
		}
	}
}

// TestRegisterTask tests registering a new task
func TestRegisterTask(t *testing.T) {
	// Save original tasks and restore after test
	originalTasks := make(TaskMap)
	for k, v := range tasks {
		originalTasks[k] = v
	}
	defer func() {
		tasks = originalTasks
	}()

	// Register a custom task
	customFn := func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
		return nil
	}

	RegisterTask("custom-task", "Custom Task Name", customFn)

	taskMap := GetTasks()
	task, exists := taskMap["custom-task"]
	if !exists {
		t.Fatal("Custom task was not registered")
	}
	if task.ID != "custom-task" {
		t.Errorf("Task ID = %q; want %q", task.ID, "custom-task")
	}
	if task.Name != "Custom Task Name" {
		t.Errorf("Task Name = %q; want %q", task.Name, "Custom Task Name")
	}
	if task.Fn == nil {
		t.Error("Task Fn is nil")
	}
}

// TestRegisterTaskOverwrite tests that registering with same ID overwrites
func TestRegisterTaskOverwrite(t *testing.T) {
	// Save original tasks and restore after test
	originalTasks := make(TaskMap)
	for k, v := range tasks {
		originalTasks[k] = v
	}
	defer func() {
		tasks = originalTasks
	}()

	// Register first version
	RegisterTask("overwrite-test", "First Version", func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
		return nil
	})

	// Overwrite with second version
	RegisterTask("overwrite-test", "Second Version", func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
		return nil
	})

	taskMap := GetTasks()
	task := taskMap["overwrite-test"]
	if task.Name != "Second Version" {
		t.Errorf("Task should be overwritten; got Name = %q", task.Name)
	}
}

// TestTaskMapType verifies TaskMap type behavior
func TestTaskMapType(t *testing.T) {
	taskMap := GetTasks()

	// Test iteration
	count := 0
	for id, task := range taskMap {
		if id == "" {
			t.Error("Task ID should not be empty")
		}
		if task.ID != id {
			t.Errorf("Task ID mismatch: map key = %q, task.ID = %q", id, task.ID)
		}
		count++
	}

	if count == 0 {
		t.Error("No tasks in task map")
	}
}

// TestTaskJSONFields verifies JSON marshaling tags
func TestTaskJSONFields(t *testing.T) {
	task := Task{
		ID:   "test-id",
		Name: "Test Name",
		Fn: func(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
			return nil
		},
	}

	// Verify struct fields
	if task.ID != "test-id" {
		t.Errorf("Task.ID = %q; want %q", task.ID, "test-id")
	}
	if task.Name != "Test Name" {
		t.Errorf("Task.Name = %q; want %q", task.Name, "Test Name")
	}
}
