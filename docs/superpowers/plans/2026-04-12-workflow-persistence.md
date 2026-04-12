# Workflow Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ability to save, recall, and run reusable multi-step task workflows (DAGs) in the media-server, with context palette integration and Drawflow editor save/load.

**Architecture:** New `workflows` table in the existing job queue SQLite DB. CRUD + run endpoints on the server. The context palette fetches saved workflows and renders them as runnable actions. The Drawflow editor gains save/load/update/delete buttons.

**Tech Stack:** Go (net/http, database/sql, SQLite), TypeScript/React (context palette), HTML/JS (Drawflow editor)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `media-server/jobqueue/workflows.go` | **Create** — Workflow DB types, table creation, CRUD methods on Queue |
| `media-server/jobqueue/workflows_test.go` | **Create** — Tests for workflow CRUD and instantiation |
| `media-server/main.go` | **Modify** — Add 6 workflow HTTP handlers and route registrations |
| `src/renderer/components/controls/context-palette.tsx` | **Modify** — Fetch and render saved workflows as actions |
| `src/renderer/components/controls/context-palette.css` | **Modify** — Style the workflows section |
| `media-server/renderer/templates/editor.go.html` | **Modify** — Add save/load/update/delete UI |

---

### Task 1: Workflow DB layer — types, table, and CRUD

**Files:**
- Create: `media-server/jobqueue/workflows.go`
- Create: `media-server/jobqueue/workflows_test.go`

- [ ] **Step 1: Create the workflow types and table creation**

Create `media-server/jobqueue/workflows.go`:

```go
package jobqueue

import (
	"encoding/json"
	"errors"
	"log"

	"github.com/google/uuid"
)

// SavedWorkflow is a reusable workflow template stored in the database.
type SavedWorkflow struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	DAG  []WorkflowTask `json:"dag"`
}

// SavedWorkflowSummary is returned by list operations (no DAG blob).
type SavedWorkflowSummary struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

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
```

- [ ] **Step 2: Add CRUD methods**

Append to the same file:

```go
func (q *Queue) ListWorkflows() ([]SavedWorkflowSummary, error) {
	if q.Db == nil {
		return nil, errors.New("no database")
	}
	rows, err := q.Db.Query("SELECT id, name FROM workflows ORDER BY name")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []SavedWorkflowSummary
	for rows.Next() {
		var w SavedWorkflowSummary
		if err := rows.Scan(&w.ID, &w.Name); err != nil {
			return nil, err
		}
		list = append(list, w)
	}
	return list, rows.Err()
}

func (q *Queue) GetWorkflow(id string) (*SavedWorkflow, error) {
	if q.Db == nil {
		return nil, errors.New("no database")
	}
	var w SavedWorkflow
	var dagJSON string
	err := q.Db.QueryRow("SELECT id, name, dag FROM workflows WHERE id = ?", id).Scan(&w.ID, &w.Name, &dagJSON)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal([]byte(dagJSON), &w.DAG); err != nil {
		return nil, err
	}
	return &w, nil
}

func (q *Queue) CreateWorkflow(name string, dag []WorkflowTask) (*SavedWorkflow, error) {
	if q.Db == nil {
		return nil, errors.New("no database")
	}
	if name == "" {
		return nil, errors.New("name cannot be empty")
	}
	if err := validateDAG(dag); err != nil {
		return nil, err
	}

	id := uuid.NewString()
	dagJSON, err := json.Marshal(dag)
	if err != nil {
		return nil, err
	}
	_, err = q.Db.Exec("INSERT INTO workflows (id, name, dag) VALUES (?, ?, ?)", id, name, string(dagJSON))
	if err != nil {
		return nil, err
	}
	return &SavedWorkflow{ID: id, Name: name, DAG: dag}, nil
}

func (q *Queue) UpdateWorkflow(id, name string, dag []WorkflowTask) (*SavedWorkflow, error) {
	if q.Db == nil {
		return nil, errors.New("no database")
	}
	if name == "" {
		return nil, errors.New("name cannot be empty")
	}
	if err := validateDAG(dag); err != nil {
		return nil, err
	}

	dagJSON, err := json.Marshal(dag)
	if err != nil {
		return nil, err
	}
	res, err := q.Db.Exec("UPDATE workflows SET name = ?, dag = ? WHERE id = ?", name, string(dagJSON), id)
	if err != nil {
		return nil, err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return nil, errors.New("workflow not found")
	}
	return &SavedWorkflow{ID: id, Name: name, DAG: dag}, nil
}

func (q *Queue) DeleteWorkflow(id string) error {
	if q.Db == nil {
		return errors.New("no database")
	}
	res, err := q.Db.Exec("DELETE FROM workflows WHERE id = ?", id)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		return errors.New("workflow not found")
	}
	return nil
}
```

- [ ] **Step 3: Add DAG validation and run instantiation**

Append to the same file:

```go
// validateDAG checks that all dependency references resolve to nodes in the DAG
// and that all node IDs are non-empty and unique.
func validateDAG(dag []WorkflowTask) error {
	if len(dag) == 0 {
		return errors.New("dag cannot be empty")
	}
	ids := make(map[string]bool, len(dag))
	for _, t := range dag {
		if t.ID == "" {
			return errors.New("all dag nodes must have an id")
		}
		if t.Command == "" {
			return errors.New("all dag nodes must have a command")
		}
		if ids[t.ID] {
			return errors.New("duplicate node id: " + t.ID)
		}
		ids[t.ID] = true
	}
	for _, t := range dag {
		for _, dep := range t.Dependencies {
			if !ids[dep] {
				return errors.New("dependency references unknown node: " + dep)
			}
		}
	}
	return nil
}

// RunWorkflow loads a saved workflow, instantiates it with fresh UUIDs,
// injects input into root nodes, and submits it for execution.
func (q *Queue) RunWorkflow(id string, input string) ([]string, error) {
	w, err := q.GetWorkflow(id)
	if err != nil {
		return nil, err
	}

	// Generate fresh UUIDs and build mapping from template ID → live ID
	idMap := make(map[string]string, len(w.DAG))
	for _, t := range w.DAG {
		idMap[t.ID] = uuid.NewString()
	}

	// Build live workflow tasks
	var liveTasks []WorkflowTask
	for _, t := range w.DAG {
		deps := make([]string, len(t.Dependencies))
		for i, dep := range t.Dependencies {
			deps[i] = idMap[dep]
		}

		taskInput := t.Input
		// Root nodes (no dependencies) get the run-time input
		if len(t.Dependencies) == 0 && input != "" {
			if taskInput != "" {
				taskInput = taskInput + " " + input
			} else {
				taskInput = input
			}
		}

		liveTasks = append(liveTasks, WorkflowTask{
			ID:           idMap[t.ID],
			Command:      t.Command,
			Arguments:    t.Arguments,
			Input:        taskInput,
			Dependencies: deps,
		})
	}

	return q.AddWorkflow(Workflow{Tasks: liveTasks})
}
```

- [ ] **Step 4: Call createWorkflowsTable from NewQueueWithDB**

In `media-server/jobqueue/jobqueue.go`, add a call to `q.createWorkflowsTable()` in `NewQueueWithDB` after the `createJobsTable` call (after line 168):

```go
	// Create the workflows table if it doesn't exist
	if err := q.createWorkflowsTable(); err != nil {
		log.Printf("Failed to create workflows table: %v", err)
	}
```

- [ ] **Step 5: Write tests**

Create `media-server/jobqueue/workflows_test.go`:

```go
package jobqueue

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func newTestQueue(t *testing.T) *Queue {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return NewQueueWithDB(db)
}

func TestCreateAndGetWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{
		{ID: "a", Command: "autotag", Arguments: []string{"--apply", "all"}},
		{ID: "b", Command: "metadata", Arguments: []string{"--type", "transcript"}, Dependencies: []string{"a"}},
	}

	w, err := q.CreateWorkflow("Test Pipeline", dag)
	if err != nil {
		t.Fatal(err)
	}
	if w.Name != "Test Pipeline" {
		t.Errorf("name = %q, want %q", w.Name, "Test Pipeline")
	}

	got, err := q.GetWorkflow(w.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.DAG) != 2 {
		t.Fatalf("dag len = %d, want 2", len(got.DAG))
	}
	if got.DAG[1].Dependencies[0] != "a" {
		t.Errorf("dep = %q, want %q", got.DAG[1].Dependencies[0], "a")
	}
}

func TestListWorkflows(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "wait"}}
	q.CreateWorkflow("Alpha", dag)
	q.CreateWorkflow("Beta", dag)

	list, err := q.ListWorkflows()
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("len = %d, want 2", len(list))
	}
	// Ordered by name
	if list[0].Name != "Alpha" || list[1].Name != "Beta" {
		t.Errorf("order: %v", list)
	}
}

func TestUpdateWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "wait"}}
	w, _ := q.CreateWorkflow("Old", dag)

	newDAG := []WorkflowTask{{ID: "x", Command: "autotag"}}
	updated, err := q.UpdateWorkflow(w.ID, "New", newDAG)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Name != "New" {
		t.Errorf("name = %q", updated.Name)
	}

	got, _ := q.GetWorkflow(w.ID)
	if got.DAG[0].Command != "autotag" {
		t.Errorf("command = %q", got.DAG[0].Command)
	}
}

func TestDeleteWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "wait"}}
	w, _ := q.CreateWorkflow("ToDelete", dag)

	if err := q.DeleteWorkflow(w.ID); err != nil {
		t.Fatal(err)
	}

	_, err := q.GetWorkflow(w.ID)
	if err == nil {
		t.Error("expected error after delete")
	}
}

func TestDeleteWorkflowNotFound(t *testing.T) {
	q := newTestQueue(t)
	err := q.DeleteWorkflow("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent workflow")
	}
}

func TestCreateWorkflowDuplicateName(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "wait"}}
	q.CreateWorkflow("Dup", dag)

	_, err := q.CreateWorkflow("Dup", dag)
	if err == nil {
		t.Error("expected error for duplicate name")
	}
}

func TestValidateDAG(t *testing.T) {
	tests := []struct {
		name    string
		dag     []WorkflowTask
		wantErr bool
	}{
		{"empty", []WorkflowTask{}, true},
		{"missing id", []WorkflowTask{{Command: "wait"}}, true},
		{"missing command", []WorkflowTask{{ID: "a"}}, true},
		{"duplicate id", []WorkflowTask{{ID: "a", Command: "wait"}, {ID: "a", Command: "wait"}}, true},
		{"bad dep ref", []WorkflowTask{{ID: "a", Command: "wait", Dependencies: []string{"z"}}}, true},
		{"valid", []WorkflowTask{
			{ID: "a", Command: "wait"},
			{ID: "b", Command: "wait", Dependencies: []string{"a"}},
		}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDAG(tt.dag)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateDAG() err = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

func TestRunWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{
		{ID: "root", Command: "autotag"},
		{ID: "child", Command: "metadata", Arguments: []string{"--type", "transcript"}, Dependencies: []string{"root"}},
	}
	w, _ := q.CreateWorkflow("Pipeline", dag)

	ids, err := q.RunWorkflow(w.ID, "--query64=abc123")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Fatalf("got %d job IDs, want 2", len(ids))
	}

	// Verify root node got the input injected
	rootJob := q.GetJob(ids[0])
	if rootJob == nil {
		t.Fatal("root job not found")
	}
	if rootJob.Input != "--query64=abc123" {
		t.Errorf("root input = %q, want %q", rootJob.Input, "--query64=abc123")
	}

	// Verify child has dependency on root (using live UUIDs, not template IDs)
	childJob := q.GetJob(ids[1])
	if childJob == nil {
		t.Fatal("child job not found")
	}
	if len(childJob.Dependencies) != 1 || childJob.Dependencies[0] != ids[0] {
		t.Errorf("child deps = %v, want [%s]", childJob.Dependencies, ids[0])
	}

	// Verify template IDs are NOT used as live IDs
	if ids[0] == "root" || ids[1] == "child" {
		t.Error("live IDs should be UUIDs, not template IDs")
	}
}
```

- [ ] **Step 6: Run tests**

Run: `cd media-server && go test ./jobqueue/ -run TestWorkflow -v && go test ./jobqueue/ -run TestValidateDAG -v && go test ./jobqueue/ -run TestCreate -v && go test ./jobqueue/ -run TestList -v && go test ./jobqueue/ -run TestUpdate -v && go test ./jobqueue/ -run TestDelete -v && go test ./jobqueue/ -run TestRun -v`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add media-server/jobqueue/workflows.go media-server/jobqueue/workflows_test.go media-server/jobqueue/jobqueue.go
git commit -m "feat: add workflow persistence layer with CRUD and run instantiation"
```

---

### Task 2: Workflow HTTP handlers

**Files:**
- Modify: `media-server/main.go`

- [ ] **Step 1: Add the 6 workflow handler functions**

In `media-server/main.go`, add these handler functions near the existing `workflowHandler` (around line 1104):

```go
// --- Saved Workflow CRUD ---

func workflowsListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		list, err := deps.Queue.ListWorkflows()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if list == nil {
			list = []jobqueue.SavedWorkflowSummary{}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(list)
	}
}

func workflowDetailHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		switch r.Method {
		case http.MethodGet:
			wf, err := deps.Queue.GetWorkflow(id)
			if err != nil {
				http.Error(w, "workflow not found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(wf)

		case http.MethodPut:
			var req struct {
				Name string                 `json:"name"`
				DAG  []jobqueue.WorkflowTask `json:"dag"`
			}
			if err := readJSONBody(r, &req); err != nil {
				http.Error(w, "bad json", http.StatusBadRequest)
				return
			}
			wf, err := deps.Queue.UpdateWorkflow(id, req.Name, req.DAG)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(wf)

		case http.MethodDelete:
			if err := deps.Queue.DeleteWorkflow(id); err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusNoContent)

		default:
			http.Error(w, "Use GET, PUT, or DELETE", http.StatusMethodNotAllowed)
		}
	}
}

type createWorkflowRequest struct {
	Name string                 `json:"name"`
	DAG  []jobqueue.WorkflowTask `json:"dag"`
}

func workflowCreateHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		var req createWorkflowRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		wf, err := deps.Queue.CreateWorkflow(req.Name, req.DAG)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(wf)
	}
}

type runWorkflowRequest struct {
	Input string `json:"input"`
}

func workflowRunHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		id := r.PathValue("id")
		var req runWorkflowRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		ids, err := deps.Queue.RunWorkflow(id, req.Input)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"ids": ids})
	}
}
```

- [ ] **Step 2: Register routes**

In `media-server/main.go`, in the routes section (around line 2806, after the `/workflow` route), add:

```go
	mux.HandleFunc("/workflows", renderer.ApplyMiddlewares(workflowsListHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/create", renderer.ApplyMiddlewares(workflowCreateHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/{id}", renderer.ApplyMiddlewares(workflowDetailHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/{id}/run", renderer.ApplyMiddlewares(workflowRunHandler(deps), renderer.RoleAdmin))
```

Note: Using `/workflows/create` for POST instead of relying on method-based dispatch on `/workflows` because Go's `http.ServeMux` matches the most specific pattern. The list handler on `/workflows` handles GET, the create handler on `/workflows/create` handles POST.

- [ ] **Step 3: Verify compilation**

Run: `cd media-server && go build ./...`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add media-server/main.go
git commit -m "feat: add workflow CRUD and run HTTP endpoints"
```

---

### Task 3: Context palette — fetch and render saved workflows

**Files:**
- Modify: `src/renderer/components/controls/context-palette.tsx`
- Modify: `src/renderer/components/controls/context-palette.css`

- [ ] **Step 1: Add a hook to fetch saved workflows**

In `src/renderer/components/controls/context-palette.tsx`, after the `useActiveJobs` hook definition (around line 240), add:

```typescript
interface SavedWorkflow {
  id: string;
  name: string;
}

function useSavedWorkflows(
  isOpen: boolean,
  authToken: string | null
): SavedWorkflow[] {
  const [workflows, setWorkflows] = useState<SavedWorkflow[]>([]);

  useEffect(() => {
    if (!isOpen || !authToken) {
      setWorkflows([]);
      return;
    }
    const fetchWorkflows = async () => {
      try {
        const headers: HeadersInit = {
          Authorization: `Bearer ${authToken}`,
        };
        const res = await fetch('http://localhost:8090/workflows', {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          const data = await res.json();
          setWorkflows(data as SavedWorkflow[]);
        }
      } catch {
        // Ignore
      }
    };
    fetchWorkflows();
  }, [isOpen, authToken]);

  return workflows;
}
```

- [ ] **Step 2: Wire up the hook and add a run handler in the component**

Inside the `ContextPalette` component, after the `useActiveJobs` call, add:

```typescript
  const savedWorkflows = useSavedWorkflows(display, authToken);
```

Then, after the `handleAction` function, add:

```typescript
  const handleRunWorkflow = async (workflow: SavedWorkflow) => {
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
      const res = await fetch(
        `http://localhost:8090/workflows/${workflow.id}/run`,
        {
          method: 'POST',
          headers,
          body: JSON.stringify({ input: queryString ? `--query64=${query64}` : '' }),
          signal: AbortSignal.timeout(10000),
          redirect: 'error',
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      libraryService.send('HIDE_CONTEXT_PALETTE');
    } catch {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Run Workflow',
          message: 'Could not communicate with job service',
        },
      });
      libraryService.send('HIDE_CONTEXT_PALETTE');
    }
  };
```

- [ ] **Step 3: Render the workflows section in JSX**

After the existing `{serverAvailable && authToken && (...actions...)}` block and before the jobs footer, add:

```tsx
      {serverAvailable && authToken && savedWorkflows.length > 0 && (
        <div className="context-palette-workflows">
          <div className="action-group">
            <span className="action-group-title">Workflows</span>
            <div className="action-buttons">
              {savedWorkflows.map((wf) => (
                <button
                  key={wf.id}
                  className="action-btn"
                  onClick={() => handleRunWorkflow(wf)}
                >
                  {wf.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
```

- [ ] **Step 4: Add CSS for the workflows section**

In `src/renderer/components/controls/context-palette.css`, after the `.action-btn:active` rule, add:

```css
.context-palette-workflows {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 6px;
  padding-top: 6px;
}

.context-palette-workflows .action-buttons {
  flex-direction: column;
  border: none;
}

.context-palette-workflows .action-btn {
  border-left: none;
  text-align: left;
  padding: 4px 8px;
}

.context-palette-workflows .action-btn + .action-btn {
  border-left: none;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
```

- [ ] **Step 5: Verify compilation**

Run: `npx tsc --noEmit 2>&1 | grep context-palette`
Expected: No errors from context-palette.

- [ ] **Step 6: Commit**

```bash
git add src/renderer/components/controls/context-palette.tsx src/renderer/components/controls/context-palette.css
git commit -m "feat: render saved workflows as actions in context palette"
```

---

### Task 4: Drawflow editor — save/load/update/delete

**Files:**
- Modify: `media-server/renderer/templates/editor.go.html`

- [ ] **Step 1: Add save/load UI buttons to the toolbar**

In `media-server/renderer/templates/editor.go.html`, replace the button toolbar (around lines 301-306):

```html
        <div style="display: flex; gap: var(--space-2)">
          <button class="btn" onclick="editor.clear()">Clear</button>
          <button class="btn btn-primary" onclick="runWorkflow()">
            Run Workflow
          </button>
        </div>
```

With:

```html
        <div style="display: flex; gap: var(--space-2); align-items: center">
          <select id="workflow-select" onchange="loadWorkflow(this.value)" style="background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border-subtle); padding: 6px 8px; border-radius: 4px; font-size: 13px;">
            <option value="">Load workflow...</option>
          </select>
          <button class="btn" onclick="saveWorkflow()">Save</button>
          <button class="btn" id="btn-update" onclick="updateWorkflow()" style="display:none">Update</button>
          <button class="btn" id="btn-delete" onclick="deleteWorkflow()" style="display:none">Delete</button>
          <button class="btn" onclick="editor.clear(); clearEditState();">Clear</button>
          <button class="btn btn-primary" onclick="runWorkflow()">Run</button>
        </div>
```

- [ ] **Step 2: Add the workflow management JavaScript**

In the same file, before the closing `</script>` tag (line 484), add:

```javascript
      // --- Workflow persistence ---
      let currentWorkflowId = null;

      function clearEditState() {
        currentWorkflowId = null;
        document.getElementById('btn-update').style.display = 'none';
        document.getElementById('btn-delete').style.display = 'none';
        document.getElementById('workflow-select').value = '';
      }

      function refreshWorkflowList() {
        fetch('/workflows')
          .then(r => r.json())
          .then(list => {
            const sel = document.getElementById('workflow-select');
            // Keep the first "Load workflow..." option
            while (sel.options.length > 1) sel.remove(1);
            (list || []).forEach(w => {
              const opt = document.createElement('option');
              opt.value = w.id;
              opt.textContent = w.name;
              sel.appendChild(opt);
            });
          })
          .catch(() => {});
      }

      // Export current canvas to DAG format
      function exportDAG() {
        const data = editor.export();
        const nodes = data.drawflow.Home.data;
        const tasks = [];
        const nodeMap = {};

        Object.keys(nodes).forEach(key => {
          const node = nodes[key];
          const id = 'node-' + key;
          nodeMap[key] = id;

          const command = node.data.command;
          const argsStr = node.data.args || '';
          const args = argsStr.match(/(?:[^\s"]+|"[^"]*")+/g)?.map(s => s.replace(/^"|"$/g, '')) || [];
          const input = node.data.input || '';

          tasks.push({
            id: id,
            drawflowId: key,
            command: command,
            arguments: args,
            input: input,
            dependencies: [],
          });
        });

        tasks.forEach(task => {
          const node = nodes[task.drawflowId];
          Object.keys(node.inputs).forEach(inputKey => {
            node.inputs[inputKey].connections.forEach(conn => {
              const parentId = nodeMap[conn.node];
              if (parentId) task.dependencies.push(parentId);
            });
          });
          delete task.drawflowId;
        });

        return tasks;
      }

      function saveWorkflow() {
        const dag = exportDAG();
        if (dag.length === 0) {
          alert('Add at least one node before saving.');
          return;
        }
        const name = prompt('Workflow name:');
        if (!name) return;

        fetch('/workflows/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, dag }),
        })
          .then(r => {
            if (!r.ok) return r.text().then(t => { throw new Error(t); });
            return r.json();
          })
          .then(wf => {
            currentWorkflowId = wf.id;
            document.getElementById('btn-update').style.display = '';
            document.getElementById('btn-delete').style.display = '';
            refreshWorkflowList();
            alert('Saved: ' + wf.name);
          })
          .catch(err => alert('Error: ' + err.message));
      }

      function updateWorkflow() {
        if (!currentWorkflowId) return;
        const dag = exportDAG();
        if (dag.length === 0) {
          alert('Add at least one node before saving.');
          return;
        }
        const name = prompt('Workflow name:', document.getElementById('workflow-select').selectedOptions[0]?.text || '');
        if (!name) return;

        fetch('/workflows/' + currentWorkflowId, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, dag }),
        })
          .then(r => {
            if (!r.ok) return r.text().then(t => { throw new Error(t); });
            return r.json();
          })
          .then(() => {
            refreshWorkflowList();
            alert('Updated');
          })
          .catch(err => alert('Error: ' + err.message));
      }

      function deleteWorkflow() {
        if (!currentWorkflowId) return;
        if (!confirm('Delete this workflow?')) return;

        fetch('/workflows/' + currentWorkflowId, { method: 'DELETE' })
          .then(r => {
            if (!r.ok) throw new Error('Failed to delete');
            editor.clear();
            clearEditState();
            refreshWorkflowList();
          })
          .catch(err => alert('Error: ' + err.message));
      }

      function loadWorkflow(id) {
        if (!id) {
          clearEditState();
          return;
        }

        fetch('/workflows/' + id)
          .then(r => {
            if (!r.ok) throw new Error('Failed to load');
            return r.json();
          })
          .then(wf => {
            editor.clear();
            currentWorkflowId = wf.id;
            document.getElementById('btn-update').style.display = '';
            document.getElementById('btn-delete').style.display = '';

            // Import DAG nodes into Drawflow
            const dag = wf.dag || [];
            const idToDrawflowId = {};
            let x = 100, y = 100;

            // Create nodes
            dag.forEach((task, idx) => {
              const argsStr = (task.arguments || []).join(' ');
              const html = `
                <div class="node-content">
                  <div class="node-input-group">
                    <label>Command</label>
                    <input type="text" df-command value="${task.command}" readonly>
                  </div>
                  <div class="node-input-group">
                    <label>Arguments</label>
                    <input type="text" df-args value="${argsStr}">
                  </div>
                  <div class="node-input-group">
                    <label>Initial Input</label>
                    <input type="text" df-input value="${task.input || ''}">
                  </div>
                </div>`;

              const drawflowId = editor.addNode(
                task.command,
                1, 1,
                x + (idx % 3) * 300,
                y + Math.floor(idx / 3) * 200,
                'node',
                { command: task.command, args: argsStr, input: task.input || '' },
                html
              );
              idToDrawflowId[task.id] = drawflowId;
            });

            // Create connections
            dag.forEach(task => {
              const targetId = idToDrawflowId[task.id];
              (task.dependencies || []).forEach(dep => {
                const sourceId = idToDrawflowId[dep];
                if (sourceId && targetId) {
                  editor.addConnection(sourceId, targetId, 'output_1', 'input_1');
                }
              });
            });
          })
          .catch(err => alert('Error: ' + err.message));
      }

      // Load workflow list on page load
      refreshWorkflowList();
```

- [ ] **Step 3: Update the existing runWorkflow function to use exportDAG**

Replace the existing `runWorkflow` function (lines 418-483) with:

```javascript
      function runWorkflow() {
        const tasks = exportDAG();
        if (tasks.length === 0) {
          alert('Add at least one node before running.');
          return;
        }

        // Replace stable IDs with UUIDs for live execution
        const idMap = {};
        tasks.forEach(t => {
          const uuid = crypto.randomUUID();
          idMap[t.id] = uuid;
          t.id = uuid;
        });
        tasks.forEach(t => {
          t.dependencies = t.dependencies.map(d => idMap[d] || d);
        });

        fetch('/workflow', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tasks }),
        })
          .then(r => {
            if (!r.ok) throw new Error('Failed to submit workflow');
            return r.json();
          })
          .then(res => {
            alert('Workflow started! Job IDs: ' + res.ids.join(', '));
          })
          .catch(err => alert('Error: ' + err.message));
      }
```

- [ ] **Step 4: Commit**

```bash
git add media-server/renderer/templates/editor.go.html
git commit -m "feat: add save/load/update/delete to Drawflow workflow editor"
```

---

### Task 5: End-to-end manual testing

- [ ] **Step 1: Test workflow CRUD via API**

```bash
# Create
curl -X POST http://localhost:8090/workflows/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"name":"Test Pipeline","dag":[{"id":"a","command":"autotag","arguments":[],"input":"","dependencies":[]},{"id":"b","command":"metadata","arguments":["--type","transcript","--apply","all"],"input":"","dependencies":["a"]}]}'

# List
curl http://localhost:8090/workflows -H "Authorization: Bearer <token>"

# Get
curl http://localhost:8090/workflows/<id> -H "Authorization: Bearer <token>"

# Run
curl -X POST http://localhost:8090/workflows/<id>/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"input":"--query64=dGFnOmxhbmRzY2FwZQ=="}'

# Delete
curl -X DELETE http://localhost:8090/workflows/<id> -H "Authorization: Bearer <token>"
```

- [ ] **Step 2: Test context palette integration**

1. Create a workflow via API or editor
2. Open context palette (shift+right-click)
3. Verify the "Workflows" section appears with the saved workflow name
4. Click it — verify jobs are created and toasts appear
5. Delete all workflows — verify the section disappears

- [ ] **Step 3: Test Drawflow editor save/load**

1. Open `/editor` in browser
2. Drag nodes, connect them, click Save, enter a name
3. Clear the canvas, select the workflow from the Load dropdown
4. Verify nodes and connections are restored
5. Modify the workflow, click Update
6. Reload, load again — verify changes persisted
7. Click Delete — verify workflow removed from dropdown

- [ ] **Step 4: Test output chaining**

1. Create a workflow where step 1 produces output and step 2 depends on it
2. Run it and verify step 2's input includes step 1's stdout

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: workflow polish from manual testing"
```
