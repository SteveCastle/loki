# Context Palette Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shift+right-click contextual palette for bulk server task operations (transcripts, auto-tagging, descriptions) on the current library selection.

**Architecture:** New `ContextPalette` component with its own XState parallel state node, positioned at click coordinates. Builds a query string from current library state and sends it to the media-server's `/create` endpoint. Shows active jobs from the server's `/jobs/list` endpoint and SSE stream.

**Tech Stack:** React, XState, CSS, media-server HTTP API + SSE

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/renderer/components/controls/context-palette.tsx` | **Create** — Context palette component with actions, job footer, positioning |
| `src/renderer/components/controls/context-palette.css` | **Create** — Styling matching command palette dark theme |
| `src/renderer/state.tsx` | **Modify** — Add `contextPalette` to state type, initial context, and parallel state node |
| `src/renderer/components/layout/panels.tsx` | **Modify** — Add shift+right-click handler |
| `src/renderer/components/detail/detail.tsx` | **Modify** — Add shift+right-click handler |
| `src/renderer/components/list/list-item.tsx` | **Modify** — Add shift+right-click handler |

---

### Task 1: Add contextPalette to XState machine

**Files:**
- Modify: `src/renderer/state.tsx:93-96` (LibraryState type, after `commandPalette`)
- Modify: `src/renderer/state.tsx:549-552` (initial context, after `commandPalette` init)
- Modify: `src/renderer/state.tsx:779-802` (parallel state node, after `commandPalette` node)

- [ ] **Step 1: Add `contextPalette` to the `LibraryState` type**

In `src/renderer/state.tsx`, after line 96 (`};` closing `commandPalette` type), add:

```typescript
  contextPalette: {
    display: boolean;
    position: { x: number; y: number };
  };
```

- [ ] **Step 2: Add `contextPalette` to initial context**

In `src/renderer/state.tsx`, after line 552 (`},` closing `commandPalette` initial value), add:

```typescript
    contextPalette: {
      display: false,
      position: { x: 0, y: 0 },
    },
```

- [ ] **Step 3: Add parallel state node for contextPalette**

In `src/renderer/state.tsx`, after line 802 (`},` closing `commandPalette` parallel state node), add a new parallel state node:

```typescript
      contextPalette: {
        on: {
          SHOW_CONTEXT_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context, event) => {
                return {
                  display: true,
                  position: event.position,
                };
              },
              commandPalette: (context) => {
                return {
                  display: false,
                  position: context.commandPalette.position,
                };
              },
            }),
          },
          HIDE_CONTEXT_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context, event) => {
                return {
                  display: false,
                  position: context.contextPalette.position,
                };
              },
            }),
          },
          SHOW_COMMAND_PALETTE: {
            actions: assign<LibraryState, AnyEventObject>({
              contextPalette: (context) => {
                return {
                  display: false,
                  position: context.contextPalette.position,
                };
              },
            }),
          },
        },
      },
```

Note: The `SHOW_COMMAND_PALETTE` handler here ensures the context palette closes when the regular command palette opens. The `SHOW_CONTEXT_PALETTE` handler also closes the command palette via the `commandPalette` assign.

- [ ] **Step 4: Verify the app compiles**

Run: `npx webpack --mode development 2>&1 | head -20`
Expected: No new type errors related to `contextPalette`

- [ ] **Step 5: Commit**

```bash
git add src/renderer/state.tsx
git commit -m "feat: add contextPalette state to XState machine"
```

---

### Task 2: Update right-click handlers for shift detection

**Files:**
- Modify: `src/renderer/components/layout/panels.tsx:16-21`
- Modify: `src/renderer/components/detail/detail.tsx:322-327`
- Modify: `src/renderer/components/list/list-item.tsx:209-217`

- [ ] **Step 1: Update panels.tsx onContextMenu**

In `src/renderer/components/layout/panels.tsx`, replace lines 16-21:

```typescript
        onContextMenu={(e) => {
          e.preventDefault();
          libraryService.send('SHOW_COMMAND_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
```

With:

```typescript
        onContextMenu={(e) => {
          e.preventDefault();
          const event = e.shiftKey
            ? 'SHOW_CONTEXT_PALETTE'
            : 'SHOW_COMMAND_PALETTE';
          libraryService.send(event, {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
```

- [ ] **Step 2: Update detail.tsx onContextMenu**

In `src/renderer/components/detail/detail.tsx`, replace lines 322-327:

```typescript
        onContextMenu={(e) => {
          e.preventDefault();
          libraryService.send('SHOW_COMMAND_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
```

With:

```typescript
        onContextMenu={(e) => {
          e.preventDefault();
          const event = e.shiftKey
            ? 'SHOW_CONTEXT_PALETTE'
            : 'SHOW_COMMAND_PALETTE';
          libraryService.send(event, {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
```

- [ ] **Step 3: Update list-item.tsx handleContextMenu**

In `src/renderer/components/list/list-item.tsx`, replace lines 209-216:

```typescript
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      libraryService.send('SHOW_COMMAND_PALETTE', {
        position: { x: e.clientX, y: e.clientY },
      });
    },
    [libraryService]
  );
```

With:

```typescript
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const event = e.shiftKey
        ? 'SHOW_CONTEXT_PALETTE'
        : 'SHOW_COMMAND_PALETTE';
      libraryService.send(event, {
        position: { x: e.clientX, y: e.clientY },
      });
    },
    [libraryService]
  );
```

- [ ] **Step 4: Verify the app compiles**

Run: `npx webpack --mode development 2>&1 | head -20`
Expected: No new errors

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/layout/panels.tsx src/renderer/components/detail/detail.tsx src/renderer/components/list/list-item.tsx
git commit -m "feat: dispatch SHOW_CONTEXT_PALETTE on shift+right-click"
```

---

### Task 3: Create the ContextPalette component

**Files:**
- Create: `src/renderer/components/controls/context-palette.tsx`

This is the main component. It handles:
- Reading library state to build a query and show context info
- Rendering actions grouped by type (each with generate/regenerate)
- Positioning at click coordinates (same pattern as command palette)
- Server health check on open
- Creating tasks via POST to `/create`
- Showing active jobs via GET `/jobs/list` + SSE `/stream`
- Closing on click outside, Escape, and library change

- [ ] **Step 1: Create the query builder utility**

Create `src/renderer/components/controls/context-palette.tsx` with the initial query builder and types:

```typescript
import React, {
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { useSelector } from '@xstate/react';
import useComponentSize from '@rehooks/component-size';
import { GlobalStateContext, LibraryState } from '../../state';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import filter from '../../filter';
import './context-palette.css';

type ActionDef = {
  label: string;
  command: (query64: string) => string;
};

type ActionGroup = {
  title: string;
  actions: ActionDef[];
};

const ACTION_GROUPS: ActionGroup[] = [
  {
    title: 'Transcripts',
    actions: [
      {
        label: 'Generate Transcripts',
        command: (q) => `metadata --type transcript --apply all --query64=${q}`,
      },
      {
        label: 'Regenerate Transcripts',
        command: (q) =>
          `metadata --type transcript --apply all --overwrite --query64=${q}`,
      },
    ],
  },
  {
    title: 'Tags',
    actions: [
      {
        label: 'Generate Tags',
        command: (q) => `autotag --query64=${q}`,
      },
      {
        label: 'Regenerate Tags',
        command: (q) => `autotag --overwrite --query64=${q}`,
      },
    ],
  },
  {
    title: 'Descriptions',
    actions: [
      {
        label: 'Generate Descriptions',
        command: (q) =>
          `metadata --type description --apply all --query64=${q}`,
      },
      {
        label: 'Regenerate Descriptions',
        command: (q) =>
          `metadata --type description --apply all --overwrite --query64=${q}`,
      },
    ],
  },
];

function buildQueryFromState(context: {
  currentStateType: 'fs' | 'db' | 'search';
  dbQuery: { tags: string[] };
  textFilter: string;
  initialFile: string;
  settings: { filteringMode: string };
}): string {
  const { currentStateType, dbQuery, textFilter, initialFile, settings } =
    context;

  if (currentStateType === 'db' && dbQuery.tags.length > 0) {
    const joiner = settings.filteringMode === 'EXCLUSIVE' ? ' AND ' : ' OR ';
    return dbQuery.tags.map((t) => `tag:${t}`).join(joiner);
  }

  if (currentStateType === 'search' && textFilter) {
    const parts: string[] = [`description:${textFilter}`];
    if (dbQuery.tags.length > 0) {
      const joiner = settings.filteringMode === 'EXCLUSIVE' ? ' AND ' : ' OR ';
      const tagPart = dbQuery.tags.map((t) => `tag:${t}`).join(joiner);
      parts.push(tagPart);
    }
    return parts.join(' AND ');
  }

  // FS mode — use directory path
  return `pathdir:${initialFile}`;
}

function buildContextLabel(context: {
  currentStateType: 'fs' | 'db' | 'search';
  dbQuery: { tags: string[] };
  textFilter: string;
  initialFile: string;
}): string {
  const { currentStateType, dbQuery, textFilter, initialFile } = context;

  if (currentStateType === 'db' && dbQuery.tags.length > 0) {
    return `${dbQuery.tags.length} tag${dbQuery.tags.length !== 1 ? 's' : ''} selected`;
  }

  if (currentStateType === 'search' && textFilter) {
    return `Search: ${textFilter}`;
  }

  // FS mode — show directory name
  const dirName = initialFile.split(/[/\\]/).filter(Boolean).pop() || initialFile;
  return `Directory: ${dirName}`;
}
```

- [ ] **Step 2: Add the job status types and fetching logic**

Append to the same file, below the utility functions:

```typescript
type JobState = 'pending' | 'in_progress' | 'completed' | 'cancelled' | 'error';

interface JobInfo {
  id: string;
  command: string;
  state: JobState;
  input: string;
}

const JOB_TITLES: Record<string, string> = {
  metadata: 'Metadata',
  autotag: 'Auto-tag',
};

function useActiveJobs(
  isOpen: boolean,
  authToken: string | null
): JobInfo[] {
  const [jobs, setJobs] = useState<JobInfo[]>([]);

  useEffect(() => {
    if (!isOpen) {
      setJobs([]);
      return;
    }

    // Fetch current jobs on open
    const fetchJobs = async () => {
      try {
        const headers: HeadersInit = {};
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
        const res = await fetch('http://localhost:8090/jobs/list', {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          const data = await res.json();
          const active = (data as JobInfo[]).filter(
            (j) => j.state === 'pending' || j.state === 'in_progress'
          );
          setJobs(active);
        }
      } catch {
        // Ignore — server may be unavailable
      }
    };
    fetchJobs();

    // Subscribe to SSE for live updates
    const es = new EventSource('http://localhost:8090/stream');

    const handleEvent = (event: Event) => {
      try {
        const parsed = JSON.parse((event as MessageEvent).data);
        const job = parsed?.job as JobInfo | undefined;
        if (!job) return;
        setJobs((prev) => {
          const without = prev.filter((j) => j.id !== job.id);
          if (job.state === 'pending' || job.state === 'in_progress') {
            return [...without, job];
          }
          return without;
        });
      } catch {
        // Ignore malformed events
      }
    };

    es.addEventListener('create', handleEvent);
    es.addEventListener('update', handleEvent);
    es.addEventListener('delete', handleEvent);

    return () => {
      es.close();
    };
  }, [isOpen, authToken]);

  return jobs;
}
```

- [ ] **Step 3: Add the main ContextPalette component**

Append the component to the same file:

```typescript
export default function ContextPalette() {
  const { libraryService } = useContext(GlobalStateContext);

  const display = useSelector(
    libraryService,
    (state) => state.context.contextPalette.display
  );
  const position = useSelector(
    libraryService,
    (state) => state.context.contextPalette.position
  );
  const currentStateType = useSelector(
    libraryService,
    (state) => state.context.currentStateType
  );
  const dbQuery = useSelector(
    libraryService,
    (state) => state.context.dbQuery
  );
  const textFilter = useSelector(
    libraryService,
    (state) => state.context.textFilter
  );
  const initialFile = useSelector(
    libraryService,
    (state) => state.context.initialFile
  );
  const filteringMode = useSelector(
    libraryService,
    (state) => state.context.settings.filteringMode
  );
  const library = useSelector(
    libraryService,
    (state) => state.context.library
  );
  const libraryLoadId = useSelector(
    libraryService,
    (state) => state.context.libraryLoadId
  );
  const filters = useSelector(
    libraryService,
    (state) => state.context.settings.filters
  );
  const sortBy = useSelector(
    libraryService,
    (state) => state.context.settings.sortBy
  );
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );

  const paletteRef = useRef<HTMLDivElement>(null);
  const { width, height } = useComponentSize(paletteRef);

  const activeJobs = useActiveJobs(display, authToken);

  // Server health
  const [serverAvailable, setServerAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    if (!display) {
      setServerAvailable(null);
      return;
    }
    const check = async () => {
      try {
        const headers: HeadersInit = {};
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
        const res = await fetch('http://localhost:8090/health', {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(3000),
        });
        setServerAvailable(res.ok);
      } catch {
        setServerAvailable(false);
      }
    };
    check();
  }, [display, authToken]);

  // Positioning (same pattern as CommandPalette)
  const [computedPosition, setComputedPosition] = useState<{
    left: number;
    top: number;
  }>({ left: -9999, top: -9999 });
  const [positionReady, setPositionReady] = useState(false);

  const getMenuPosition = useCallback(
    (x: number, y: number, w: number, h: number) => {
      const margin = 8;
      const maxLeft = Math.max(margin, window.innerWidth - w - margin);
      const maxTop = Math.max(margin, window.innerHeight - h - margin);
      return {
        left: Math.min(Math.max(x, margin), maxLeft),
        top: Math.min(Math.max(y, margin), maxTop),
      };
    },
    []
  );

  const recomputePosition = useCallback(() => {
    const el = paletteRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pw = Math.max(rect.width || 0, width || 0, 280);
    const ph = Math.max(rect.height || 0, height || 0, 100);
    const px = typeof position?.x === 'number' ? position.x : 0;
    const py = typeof position?.y === 'number' ? position.y : 0;
    const next = getMenuPosition(px, py, pw, ph);
    setComputedPosition(next);
    setPositionReady(true);
  }, [getMenuPosition, position?.x, position?.y, width, height]);

  useLayoutEffect(() => {
    if (!display) return;
    recomputePosition();
  }, [display, recomputePosition]);

  useEffect(() => {
    if (!display) return;
    const onResize = () => recomputePosition();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [display, recomputePosition]);

  useEffect(() => {
    if (!display) {
      setPositionReady(false);
      setComputedPosition({ left: -9999, top: -9999 });
    }
  }, [display]);

  // Close on click outside
  useOnClickOutside(paletteRef, () => {
    if (display) libraryService.send('HIDE_CONTEXT_PALETTE');
  });

  // Close on Escape
  useEffect(() => {
    if (!display) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        libraryService.send('HIDE_CONTEXT_PALETTE');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [display, libraryService]);

  // Derived data
  const items = filter(libraryLoadId, textFilter, library, filters, sortBy);
  const itemCount = items.length;
  const contextLabel = buildContextLabel({
    currentStateType,
    dbQuery,
    textFilter,
    initialFile,
  });
  const queryString = buildQueryFromState({
    currentStateType,
    dbQuery,
    textFilter,
    initialFile,
    settings: { filteringMode },
  });
  const query64 = btoa(queryString);

  // Action handler
  const handleAction = async (action: ActionDef) => {
    const input = action.command(query64);
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
      const res = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers,
        body: JSON.stringify({ input }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    }
    libraryService.send('HIDE_CONTEXT_PALETTE');
  };

  if (!display) return null;

  const style: React.CSSProperties = positionReady
    ? { left: computedPosition.left, top: computedPosition.top, visibility: 'visible' }
    : { left: -9999, top: -9999, visibility: 'hidden' };

  return (
    <div className="ContextPalette" ref={paletteRef} style={style}>
      <div className="context-palette-header">
        <span className="context-label">{contextLabel}</span>
        <span className="context-count">{itemCount} items</span>
      </div>

      {serverAvailable === false && (
        <div className="context-palette-unavailable">
          Job service unavailable at localhost:8090
        </div>
      )}

      {serverAvailable === null && (
        <div className="context-palette-checking">Checking job service...</div>
      )}

      {serverAvailable && (
        <div className="context-palette-actions">
          {ACTION_GROUPS.map((group) => (
            <div key={group.title} className="action-group">
              <div className="action-group-title">{group.title}</div>
              {group.actions.map((action) => (
                <button
                  key={action.label}
                  className="action-row"
                  onClick={() => handleAction(action)}
                >
                  {action.label}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}

      {serverAvailable && activeJobs.length > 0 && (
        <div className="context-palette-footer">
          <div className="footer-title">Active Jobs</div>
          {activeJobs.map((job) => (
            <div key={job.id} className="footer-job">
              <span
                className={`job-indicator ${job.state}`}
              />
              <span className="job-label">
                {JOB_TITLES[job.command] || job.command}
              </span>
              <span className="job-state">{job.state.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Verify the app compiles**

Run: `npx webpack --mode development 2>&1 | head -20`
Expected: Compiles (component not yet mounted, but no import/type errors)

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/controls/context-palette.tsx
git commit -m "feat: create ContextPalette component with actions and job footer"
```

---

### Task 4: Create context-palette.css

**Files:**
- Create: `src/renderer/components/controls/context-palette.css`

- [ ] **Step 1: Create the stylesheet**

Create `src/renderer/components/controls/context-palette.css`:

```css
.ContextPalette {
  color: white;
  background-color: var(--controls-background-color);
  border-radius: 2px;
  display: flex;
  flex-direction: column;
  z-index: 9999;
  padding: 12px 16px;
  position: fixed;
  user-select: none;
  opacity: 0.9;
  min-width: 280px;
  max-width: 400px;
  font-size: 13px;
}

.context-palette-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
  margin-bottom: 8px;
}

.context-label {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  margin-right: 12px;
}

.context-count {
  color: #00c896;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.context-palette-unavailable,
.context-palette-checking {
  padding: 12px 0;
  color: rgba(255, 255, 255, 0.5);
  text-align: center;
}

.context-palette-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.action-group {
  display: flex;
  flex-direction: column;
}

.action-group-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.4);
  padding: 6px 0 2px;
}

.action-row {
  background: none;
  border: none;
  color: white;
  text-align: left;
  padding: 6px 8px;
  border-radius: 2px;
  cursor: pointer;
  font-size: 13px;
  font-family: inherit;
}

.action-row:hover {
  background-color: rgba(0, 200, 150, 0.15);
}

.action-row:active {
  background-color: rgba(0, 200, 150, 0.25);
}

.context-palette-footer {
  border-top: 1px solid rgba(255, 255, 255, 0.15);
  margin-top: 8px;
  padding-top: 8px;
}

.footer-title {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.4);
  margin-bottom: 4px;
}

.footer-job {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
  font-size: 12px;
}

.job-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.job-indicator.pending {
  background-color: rgb(223, 226, 31);
}

.job-indicator.in_progress {
  background-color: #00c896;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.job-label {
  flex: 1;
}

.job-state {
  color: rgba(255, 255, 255, 0.5);
  text-transform: capitalize;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/renderer/components/controls/context-palette.css
git commit -m "feat: add context palette styles"
```

---

### Task 5: Mount ContextPalette in the app

**Files:**
- Modify: `src/renderer/components/layout/layout.tsx:20,200`
- Modify: `src/renderer/components/layout/detail-only.tsx:7,12`

`CommandPalette` is mounted in two layout files. Add `ContextPalette` in both.

- [ ] **Step 1: Add ContextPalette to layout.tsx**

In `src/renderer/components/layout/layout.tsx`, add the import after the existing CommandPalette import (line 20):

```typescript
import ContextPalette from '../controls/context-palette';
```

Then render it right after `<CommandPalette />` at line 200:

```tsx
      <CommandPalette />
      <ContextPalette />
```

- [ ] **Step 2: Add ContextPalette to detail-only.tsx**

In `src/renderer/components/layout/detail-only.tsx`, add the import after line 7:

```typescript
import ContextPalette from '../controls/context-palette';
```

Then render it right after `<CommandPalette />` at line 12:

```tsx
      <CommandPalette />
      <ContextPalette />
```

- [ ] **Step 3: Verify the app compiles and runs**

Run: `npx webpack --mode development 2>&1 | head -20`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add src/renderer/components/layout/layout.tsx src/renderer/components/layout/detail-only.tsx
git commit -m "feat: mount ContextPalette component in app"
```

---

### Task 6: Manual testing and polish

- [ ] **Step 1: Test shift+right-click opens context palette**

1. Start the dev server
2. Load a library (any mode — FS, DB tags, or search)
3. Shift+right-click anywhere in the panels/detail/list area
4. Verify the context palette appears at click position with correct header info (mode + item count)
5. Verify regular right-click still opens the settings palette
6. Verify only one palette can be open at a time

- [ ] **Step 2: Test actions create server tasks**

1. With the media-server running at localhost:8090
2. Open context palette on a library
3. Click "Generate Transcripts"
4. Verify palette closes and a job toast appears
5. Verify the job was created with correct `--query64` parameter (check server logs or `/jobs/list`)

- [ ] **Step 3: Test job footer shows active jobs**

1. Create a long-running task (e.g. transcripts on many items)
2. Re-open the context palette while the job is running
3. Verify the footer shows the active job with correct status
4. Verify the status updates in real-time via SSE

- [ ] **Step 4: Test close behaviors**

1. Open palette, click outside — should close
2. Open palette, press Escape — should close
3. Open palette, change library (click a tag) — should close
4. Open context palette, then right-click (no shift) — context palette closes, settings palette opens

- [ ] **Step 5: Test server unavailable state**

1. Stop the media-server
2. Open context palette
3. Verify it shows "Job service unavailable" message instead of actions

- [ ] **Step 6: Test all three library modes**

1. FS mode: load a directory, shift+right-click, verify header shows "Directory: <name>"
2. DB mode: select tags, shift+right-click, verify header shows "N tags selected"
3. Search mode: search text, shift+right-click, verify header shows "Search: <text>"

- [ ] **Step 7: Commit any fixes**

```bash
git add -A
git commit -m "fix: context palette polish from manual testing"
```
