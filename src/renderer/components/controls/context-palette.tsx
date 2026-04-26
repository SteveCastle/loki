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
import { GlobalStateContext } from '../../state';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import filter from '../../filter';
import LoginWidget from './login-widget';
import './context-palette.css';

type ActionDef = {
  label: string;
  command: (query64: string) => string;
  // When true and the current context is a folder (pathdir query), append
  // `tagcount:<3` to the query before base64 encoding so already-tagged
  // media are skipped.
  skipTaggedInFolder?: boolean;
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
        label: 'Generate',
        command: (q) => `metadata --type transcript --apply all --query64=${q}`,
      },
      {
        label: 'Regenerate',
        command: (q) =>
          `metadata --type transcript --apply all --overwrite --query64=${q}`,
      },
    ],
  },
  {
    title: 'Tags',
    actions: [
      {
        label: 'Generate',
        command: (q) => `autotag --query64=${q}`,
        skipTaggedInFolder: true,
      },
      {
        label: 'Regenerate',
        command: (q) => `autotag --overwrite --query64=${q}`,
      },
    ],
  },
  {
    title: 'Descriptions',
    actions: [
      {
        label: 'Generate',
        command: (q) =>
          `metadata --type description --apply all --query64=${q}`,
      },
      {
        label: 'Regenerate',
        command: (q) =>
          `metadata --type description --apply all --overwrite --query64=${q}`,
      },
    ],
  },
];

type ContextTarget =
  | { type: 'library' }
  | { type: 'file'; path: string }
  | { type: 'tag'; tag: string }
  | { type: 'category'; category: string };

/** If initialFile is a media file path, return its parent directory; otherwise return as-is. */
function getDirFromInitialFile(initialFile: string): string {
  const lastSegment = initialFile.split(/[/\\]/).pop() || '';
  if (lastSegment.includes('.')) {
    const sep = initialFile.includes('\\') ? '\\' : '/';
    const idx = initialFile.lastIndexOf(sep);
    let dir = idx > 0 ? initialFile.slice(0, idx) : initialFile;
    // Ensure Windows drive-letter roots keep their trailing separator (e.g. "D:" → "D:\")
    if (/^[A-Za-z]:$/.test(dir)) {
      dir += sep;
    }
    return dir;
  }
  return initialFile;
}

function buildQuery(
  target: ContextTarget,
  libraryContext: {
    currentStateType: 'fs' | 'db' | 'search';
    dbQuery: { tags: string[] };
    textFilter: string;
    initialFile: string;
    settings: { filteringMode: string };
  }
): string {
  switch (target.type) {
    case 'file':
      return `path:"${target.path}"`;
    case 'tag':
      return `tag:${target.tag}`;
    case 'category':
      return `category:${target.category}`;
    case 'library': {
      const { currentStateType, dbQuery, textFilter, initialFile, settings } =
        libraryContext;
      if (currentStateType === 'db' && dbQuery.tags.length > 0) {
        const joiner =
          settings.filteringMode === 'EXCLUSIVE' ? ' AND ' : ' OR ';
        return dbQuery.tags.map((t) => `tag:${t}`).join(joiner);
      }
      if (currentStateType === 'search' && textFilter) {
        const parts: string[] = [`description:${textFilter}`];
        if (dbQuery.tags.length > 0) {
          const joiner =
            settings.filteringMode === 'EXCLUSIVE' ? ' AND ' : ' OR ';
          parts.push(dbQuery.tags.map((t) => `tag:${t}`).join(joiner));
        }
        return parts.join(' AND ');
      }
      return `pathdir:"${getDirFromInitialFile(initialFile)}"`;
    }
  }
}

function buildLabel(
  target: ContextTarget,
  libraryContext: {
    currentStateType: 'fs' | 'db' | 'search';
    dbQuery: { tags: string[] };
    textFilter: string;
    initialFile: string;
  }
): string {
  switch (target.type) {
    case 'file': {
      const name = target.path.split(/[/\\]/).pop() || target.path;
      return name;
    }
    case 'tag':
      return `Tag: ${target.tag}`;
    case 'category':
      return `Category: ${target.category}`;
    case 'library': {
      const { currentStateType, dbQuery, textFilter, initialFile } =
        libraryContext;
      if (currentStateType === 'db' && dbQuery.tags.length > 0) {
        return `${dbQuery.tags.length} tag${dbQuery.tags.length !== 1 ? 's' : ''} selected`;
      }
      if (currentStateType === 'search' && textFilter) {
        return `Search: ${textFilter}`;
      }
      const dir = getDirFromInitialFile(initialFile);
      return `Directory: ${dir.split(/[/\\]/).filter(Boolean).pop() || dir}`;
    }
  }
}

type JobState =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'cancelled'
  | 'error';

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

function useActiveJobs(isOpen: boolean, authToken: string | null): JobInfo[] {
  const [jobs, setJobs] = useState<JobInfo[]>([]);

  useEffect(() => {
    if (!isOpen) {
      setJobs([]);
      return;
    }

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

function WorkflowPicker({
  workflows,
  onRun,
}: {
  workflows: SavedWorkflow[];
  onRun: (wf: SavedWorkflow) => void;
}) {
  const [search, setSearch] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = search
    ? workflows.filter((wf) =>
        wf.name.toLowerCase().includes(search.toLowerCase())
      )
    : workflows;

  // Reset selection when filter changes
  useEffect(() => {
    setSelectedIdx(0);
  }, [search]);

  // Keyboard navigation inside the picker
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && filtered[selectedIdx]) {
      e.preventDefault();
      e.stopPropagation();
      onRun(filtered[selectedIdx]);
    }
  };

  // Scroll selected item into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const item = list.children[selectedIdx] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }, [selectedIdx]);

  return (
    <div className="context-palette-workflows" onKeyDown={handleKeyDown}>
      <div className="workflow-picker-header">
        <span className="action-group-title">Workflows</span>
        <span className="workflow-count">{filtered.length}</span>
      </div>
      <input
        ref={inputRef}
        className="workflow-search"
        type="text"
        placeholder="Filter..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        autoFocus={false}
      />
      <div className="workflow-list" ref={listRef}>
        {filtered.map((wf, i) => (
          <div
            key={wf.id}
            className={`workflow-item${i === selectedIdx ? ' selected' : ''}`}
            onClick={() => onRun(wf)}
            onMouseEnter={() => setSelectedIdx(i)}
          >
            {wf.name}
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="workflow-empty">No matches</div>
        )}
      </div>
    </div>
  );
}

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
  const target = useSelector(
    libraryService,
    (state) => state.context.contextPalette.target
  ) as ContextTarget;
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
  const savedWorkflows = useSavedWorkflows(display, authToken);

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

  // Close on library change
  const prevLibraryLoadId = useRef(libraryLoadId);
  useEffect(() => {
    if (
      display &&
      prevLibraryLoadId.current &&
      prevLibraryLoadId.current !== libraryLoadId
    ) {
      libraryService.send('HIDE_CONTEXT_PALETTE');
    }
    prevLibraryLoadId.current = libraryLoadId;
  }, [display, libraryLoadId, libraryService]);

  // Derived data
  const libraryCtx = {
    currentStateType,
    dbQuery,
    textFilter,
    initialFile,
    settings: { filteringMode },
  };
  const items = filter(libraryLoadId, textFilter, library, filters, sortBy);
  const itemCount = target.type === 'file' ? 1 : items.length;
  const contextLabel = buildLabel(target, libraryCtx);
  const queryString = buildQuery(target, libraryCtx);
  const isFolderContext =
    target.type === 'library' &&
    !(currentStateType === 'db' && dbQuery.tags.length > 0) &&
    !(currentStateType === 'search' && textFilter);
  const encodeQuery64 = (q: string) =>
    btoa(
      new TextEncoder()
        .encode(q)
        .reduce((s, b) => s + String.fromCharCode(b), ''),
    );
  const query64 = encodeQuery64(queryString);

  // Action handler
  const handleAction = async (action: ActionDef) => {
    const effectiveQuery =
      action.skipTaggedInFolder && isFolderContext
        ? `${queryString} tagcount:<3`
        : queryString;
    const effectiveQuery64 =
      effectiveQuery === queryString ? query64 : encodeQuery64(effectiveQuery);
    const input = action.command(effectiveQuery64);
    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
      const res = await fetch('http://localhost:8090/create', {
        method: 'POST',
        headers,
        body: JSON.stringify({ input }),
        signal: AbortSignal.timeout(10000),
        redirect: 'error',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Job toast appears automatically via SSE stream in ToastSystem
      libraryService.send('HIDE_CONTEXT_PALETTE');
    } catch {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
      libraryService.send('HIDE_CONTEXT_PALETTE');
    }
  };

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

  if (!display) return null;

  const style: React.CSSProperties = positionReady
    ? {
        left: computedPosition.left,
        top: computedPosition.top,
        visibility: 'visible',
      }
    : { left: -9999, top: -9999, visibility: 'hidden' };

  return (
    <div className="ContextPalette" ref={paletteRef} style={style}>
      <div className="context-palette-header">
        <span className="context-label">{contextLabel}</span>
        {target.type === 'library' && (
          <span className="context-count">{itemCount} items</span>
        )}
        {target.type === 'file' && (
          <span className="context-count">1 file</span>
        )}
      </div>

      {serverAvailable === false && (
        <div className="context-palette-unavailable">
          Job service unavailable at localhost:8090
        </div>
      )}

      {serverAvailable === null && (
        <div className="context-palette-checking">
          Checking job service...
        </div>
      )}

      {serverAvailable && !authToken && (
        <div className="context-palette-login">
          <LoginWidget />
        </div>
      )}

      {serverAvailable && authToken && (
        <div className="context-palette-actions">
          {ACTION_GROUPS.map((group) => (
            <div key={group.title} className="action-group">
              <span className="action-group-title">{group.title}</span>
              <div className="action-buttons">
                {group.actions.map((action) => (
                  <button
                    key={action.label}
                    className="action-btn"
                    onClick={() => handleAction(action)}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {serverAvailable && authToken && savedWorkflows.length > 0 && (
        <WorkflowPicker
          workflows={savedWorkflows}
          onRun={handleRunWorkflow}
        />
      )}

      {serverAvailable && authToken && activeJobs.length > 0 && (
        <div className="context-palette-footer">
          <div className="footer-title">Active Jobs</div>
          <div className="footer-jobs-scroll">
            {activeJobs.map((job) => (
              <div key={job.id} className="footer-job">
                <span className={`job-indicator ${job.state}`} />
                <span className="job-label">
                  {JOB_TITLES[job.command] || job.command}
                </span>
                <span className="job-state">
                  {job.state.replace('_', ' ')}
                </span>
                <button
                  className="job-cancel"
                  onClick={async () => {
                    try {
                      const headers: HeadersInit = {};
                      if (authToken)
                        headers['Authorization'] = `Bearer ${authToken}`;
                      await fetch(
                        `http://localhost:8090/job/${job.id}/cancel`,
                        { method: 'POST', headers }
                      );
                    } catch {
                      // ignore
                    }
                  }}
                  title="Cancel job"
                >
                  &times;
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
