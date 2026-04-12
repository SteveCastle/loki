package jobqueue

import (
	"database/sql"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
)

// SavedWorkflow represents a persisted workflow template with its full DAG.
type SavedWorkflow struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	DAG  []WorkflowTask `json:"dag"`
}

// SavedWorkflowSummary is a lightweight representation for listing workflows.
type SavedWorkflowSummary struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// createWorkflowsTable creates the workflows table if it doesn't exist.
func (q *Queue) createWorkflowsTable() error {
	query := `
	CREATE TABLE IF NOT EXISTS workflows (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL UNIQUE,
		dag TEXT NOT NULL
	)`
	_, err := q.Db.Exec(query)
	return err
}

// ListWorkflows returns all saved workflows ordered by name.
func (q *Queue) ListWorkflows() ([]SavedWorkflowSummary, error) {
	rows, err := q.Db.Query("SELECT id, name FROM workflows ORDER BY name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var workflows []SavedWorkflowSummary
	for rows.Next() {
		var w SavedWorkflowSummary
		if err := rows.Scan(&w.ID, &w.Name); err != nil {
			return nil, err
		}
		workflows = append(workflows, w)
	}
	return workflows, rows.Err()
}

// GetWorkflow retrieves a saved workflow by ID, including its full DAG.
func (q *Queue) GetWorkflow(id string) (*SavedWorkflow, error) {
	var w SavedWorkflow
	var dagJSON string
	err := q.Db.QueryRow("SELECT id, name, dag FROM workflows WHERE id = ?", id).Scan(&w.ID, &w.Name, &dagJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("workflow not found: %s", id)
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal([]byte(dagJSON), &w.DAG); err != nil {
		return nil, fmt.Errorf("failed to unmarshal DAG: %w", err)
	}
	return &w, nil
}

// CreateWorkflow validates and persists a new workflow template.
func (q *Queue) CreateWorkflow(name string, dag []WorkflowTask) (*SavedWorkflow, error) {
	if err := validateDAG(dag); err != nil {
		return nil, err
	}

	id := uuid.NewString()
	dagJSON, err := json.Marshal(dag)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal DAG: %w", err)
	}

	_, err = q.Db.Exec("INSERT INTO workflows (id, name, dag) VALUES (?, ?, ?)", id, name, string(dagJSON))
	if err != nil {
		return nil, err
	}

	return &SavedWorkflow{ID: id, Name: name, DAG: dag}, nil
}

// UpdateWorkflow updates an existing workflow's name and DAG.
func (q *Queue) UpdateWorkflow(id string, name string, dag []WorkflowTask) error {
	if err := validateDAG(dag); err != nil {
		return err
	}

	dagJSON, err := json.Marshal(dag)
	if err != nil {
		return fmt.Errorf("failed to marshal DAG: %w", err)
	}

	result, err := q.Db.Exec("UPDATE workflows SET name = ?, dag = ? WHERE id = ?", name, string(dagJSON), id)
	if err != nil {
		return err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows == 0 {
		return fmt.Errorf("workflow not found: %s", id)
	}
	return nil
}

// DeleteWorkflow removes a saved workflow by ID.
func (q *Queue) DeleteWorkflow(id string) error {
	result, err := q.Db.Exec("DELETE FROM workflows WHERE id = ?", id)
	if err != nil {
		return err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows == 0 {
		return fmt.Errorf("workflow not found: %s", id)
	}
	return nil
}

// validateDAG checks that a DAG is well-formed.
func validateDAG(dag []WorkflowTask) error {
	if len(dag) == 0 {
		return fmt.Errorf("DAG must not be empty")
	}

	ids := make(map[string]bool, len(dag))
	for _, task := range dag {
		if task.ID == "" {
			return fmt.Errorf("all tasks must have an id")
		}
		if task.Command == "" {
			return fmt.Errorf("all tasks must have a command")
		}
		if ids[task.ID] {
			return fmt.Errorf("duplicate task id: %s", task.ID)
		}
		ids[task.ID] = true
	}

	for _, task := range dag {
		for _, dep := range task.Dependencies {
			if !ids[dep] {
				return fmt.Errorf("dependency %s not found in DAG", dep)
			}
		}
	}

	return nil
}

// RunWorkflow loads a saved workflow, generates fresh UUIDs for all nodes,
// remaps dependency references, injects input into root nodes, and submits
// the workflow as live jobs via AddWorkflow.
func (q *Queue) RunWorkflow(id string, input string) ([]string, error) {
	saved, err := q.GetWorkflow(id)
	if err != nil {
		return nil, err
	}

	// Generate fresh UUIDs and build a mapping from template ID to live ID.
	idMap := make(map[string]string, len(saved.DAG))
	for _, task := range saved.DAG {
		idMap[task.ID] = uuid.NewString()
	}

	// Build live tasks with remapped IDs and dependencies.
	tasks := make([]WorkflowTask, len(saved.DAG))
	for i, task := range saved.DAG {
		liveDeps := make([]string, len(task.Dependencies))
		for j, dep := range task.Dependencies {
			liveDeps[j] = idMap[dep]
		}

		liveInput := task.Input
		// Inject runtime input into root nodes (those with no dependencies).
		if len(task.Dependencies) == 0 && input != "" {
			if liveInput != "" {
				liveInput = liveInput + " " + input
			} else {
				liveInput = input
			}
		}

		tasks[i] = WorkflowTask{
			ID:           idMap[task.ID],
			Command:      task.Command,
			Arguments:    task.Arguments,
			Input:        liveInput,
			Dependencies: liveDeps,
		}
	}

	return q.AddWorkflow(Workflow{Tasks: tasks})
}
