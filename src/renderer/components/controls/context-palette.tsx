import React, {
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import useComponentSize from '@rehooks/component-size';
import { GlobalStateContext } from '../../state';
import { capabilities, mediaServerBase } from '../../platform';
import { subscribeStream, streamConnected } from '../../stream-bus';
import { displayTagLabel } from '../../tag-display';
import type { Predicate } from '../../query/types';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import { absorbNextClick } from '../../absorb-next-click';
import filter from '../../filter';
import LoginWidget from './login-widget';
import {
  getDirFromInitialFile,
  buildLibraryPathQuery,
  buildLegacyQuery,
  quoteValue,
  LEGACY_PREFIX,
} from './context-query';
import {
  fetchStatus,
  startModelDownload,
  isDownloadableState,
  isDownloadingState,
  type DepStatus,
} from '../../onboarding/api';
import {
  TASK_REQUIREMENTS,
  depsApiBase,
  fmtSize,
} from '../../onboarding/requirements';
import './context-palette.css';

// Generation mode for the metadata chips. `missing` only fills gaps; `all`
// replaces existing metadata (passes `--overwrite`). The mode is chosen once
// via the panel-wide toggle and applied to whichever chip is clicked.
type GenMode = 'missing' | 'all';

type MetadataType = {
  label: string;
  // The per-item operation(s) this chip contributes. Chips multi-select:
  // every selected op joins ONE `process` job — a single pass that applies
  // all of them to each file together, with unified overwrite (`all` mode
  // adds `--overwrite`), live progress, and pause/resume.
  ops: string[];
  // When true and the current context is a folder (pathdir query), the `missing`
  // mode appends `tagcount:<3` to the query before base64 encoding so already-
  // tagged media are skipped. Only applied when this chip runs ALONE — a
  // combined job shares one query and must not restrict the other ops.
  skipTaggedInFolder?: boolean;
};

// (--query64 must stay the LAST token — the server treats the final token as
// the job input.)
const METADATA_TYPES: MetadataType[] = [
  { label: 'Tags', ops: ['autotag'], skipTaggedInFolder: true },
  { label: 'Descriptions', ops: ['describe'] },
  { label: 'Transcripts', ops: ['transcribe'] },
  { label: 'File info', ops: ['hash', 'dimensions'] },
  { label: 'Embeddings', ops: ['embed'] },
  { label: 'Faces', ops: ['faces'] },
];

// The chip selection persists across palette opens and app restarts.
const SELECTED_TYPES_STORAGE_KEY = 'loki.contextPalette.selectedTypes';

// loadSelectedTypes restores the persisted chip selection, dropping any label
// that no longer exists (chips get renamed/removed across versions).
function loadSelectedTypes(): string[] {
  try {
    const raw = localStorage.getItem(SELECTED_TYPES_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const known = new Set(METADATA_TYPES.map((m) => m.label));
    return parsed.filter(
      (l): l is string => typeof l === 'string' && known.has(l)
    );
  } catch {
    return [];
  }
}

type ContextTarget =
  | { type: 'library' }
  | { type: 'file'; path: string }
  | { type: 'tag'; tag: string }
  | { type: 'category'; category: string };

function buildQuery(
  target: ContextTarget,
  libraryContext: {
    currentStateType: 'fs' | 'db';
    predicates: Predicate[];
    textFilter: string;
    initialFile: string;
    settings: { filteringMode: string; recursive: boolean };
  }
): string {
  switch (target.type) {
    case 'file':
      return `path:"${target.path}"`;
    case 'tag':
      return `tag:${quoteValue(target.tag)}`;
    case 'category':
      return `category:${quoteValue(target.category)}`;
    case 'library': {
      const { predicates, initialFile, settings } = libraryContext;
      // Match the FULL unified query the search input shows — every predicate
      // type, excludes (NOT), and per-predicate AND/OR joins — not just tags.
      const legacy = buildLegacyQuery(predicates, settings.filteringMode);
      if (legacy) return legacy;
      // No representable predicates (filesystem browsing): match the current
      // list view. When recursive browsing is on the list spans subdirectories,
      // so match every path under the directory; otherwise its immediate children.
      return buildLibraryPathQuery(initialFile, settings.recursive);
    }
  }
}

function buildLabel(
  target: ContextTarget,
  libraryContext: {
    currentStateType: 'fs' | 'db';
    predicates: Predicate[];
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
      const { predicates, initialFile } = libraryContext;
      const n = predicates.filter((p) => LEGACY_PREFIX[p.type] && p.value).length;
      if (n > 0) {
        return `${n} filter${n !== 1 ? 's' : ''} selected`;
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
  autotag: 'Auto-Tagging',
  embed: 'Visual Embedding',
  faces: 'Face Scan',
  'faces-cluster': 'Face Clustering',
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
        const res = await fetch(`${mediaServerBase}/jobs/list`, {
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

    // Shared /stream bus — the palette must never cost an extra socket
    // (Chromium caps connections per origin at 6; see stream-bus.ts).
    return subscribeStream((type, event) => {
      if (type !== 'create' && type !== 'update' && type !== 'delete') return;
      try {
        const parsed = JSON.parse(event.data);
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
    });
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
        const res = await fetch(`${mediaServerBase}/workflows`, {
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

// Dependency state for the Generate chips — polls only while the palette is
// open, so a closed palette costs nothing. Progress rides along in `detail`
// (no EventSource: palette sessions are short and connection slots are scarce).
function useDeps(isOpen: boolean): {
  deps: Map<string, DepStatus>;
  refreshDeps: () => void;
} {
  const [deps, setDeps] = useState<Map<string, DepStatus>>(new Map());

  const refreshDeps = useCallback(async () => {
    try {
      const items = await fetchStatus(depsApiBase);
      setDeps(new Map(items.map((d) => [d.id, d])));
    } catch {
      // Deps API unavailable — chips stay ungated and jobs fail with their
      // own (polite) in-log errors, same as before this feature existed.
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return undefined;
    refreshDeps();
    const t = window.setInterval(refreshDeps, 3000);
    return () => window.clearInterval(t);
  }, [isOpen, refreshDeps]);

  return { deps, refreshDeps };
}

// One row per Generate chip whose dependency needs attention: a Download
// button for missing models/tools, live progress while installing, and a
// non-blocking hint for external tools like Ollama.
function DepRequirementRows({
  deps,
  onChange,
}: {
  deps: Map<string, DepStatus>;
  onChange: () => void;
}) {
  const rows = Object.entries(TASK_REQUIREMENTS)
    .map(([label, req]) => ({ label, req, dep: deps.get(req.depId) }))
    .filter(({ req, dep }) => {
      if (!dep) return false;
      if (req.kind === 'external') return dep.state === 'not_installed';
      return isDownloadableState(dep.state) || isDownloadingState(dep.state);
    });
  if (rows.length === 0) return null;

  return (
    <div className="dep-rows">
      {rows.map(({ label, req, dep }) => {
        const d = dep!;
        if (req.kind === 'external') {
          return (
            <div key={label} className="dep-row hint">
              <span>
                {req.feature} uses your configured AI provider — Ollama not
                detected.{' '}
                <a href="https://ollama.com/download" target="_blank" rel="noreferrer">
                  Get Ollama
                </a>
              </span>
            </div>
          );
        }
        if (isDownloadingState(d.state)) {
          const inst = d.detail || {};
          const done: number = inst.bytes_done ?? 0;
          const total: number = inst.bytes_total ?? d.size_bytes ?? 0;
          const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
          return (
            <div key={label} className="dep-row">
              <span>
                {req.feature}: downloading… {fmtSize(done)} / {fmtSize(total)} ({pct}%)
              </span>
              <div className="dep-progress">
                <div className="dep-progress-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        }
        return (
          <div key={label} className="dep-row">
            <span>
              {req.feature} needs a one-time download ({fmtSize(d.size_bytes)}).
              {d.state === 'failed' && d.error ? ` Last attempt failed: ${d.error}` : ''}
            </span>
            <button
              type="button"
              className="dep-download-btn"
              onClick={async () => {
                try {
                  await startModelDownload(req.depId, depsApiBase);
                } catch {
                  /* row will show failed state on next poll */
                }
                onChange();
              }}
            >
              {d.state === 'failed' ? 'Retry download' : 'Download'}
            </button>
          </div>
        );
      })}
    </div>
  );
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
  const queryClient = useQueryClient();

  const display = useSelector(
    libraryService,
    (state) => state.context.contextPalette.display
  );
  // The palette is primarily a job/AI launcher — hidden entirely for
  // view-only public visitors (see render guard below the hooks).
  const canWrite = useSelector(
    libraryService,
    (state) => state.context.canWrite
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
  const predicates = useSelector(
    libraryService,
    (state) => state.context.query.predicates
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
  const recursive = useSelector(
    libraryService,
    (state) => state.context.settings.recursive
  );
  const library = useSelector(
    libraryService,
    (state) => state.context.library
  );
  const libraryLoadId = useSelector(
    libraryService,
    (state) => state.context.libraryLoadId
  );
  const streaming = useSelector(
    libraryService,
    (state) => state.context.streaming
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

  // Narrow the right-clicked file path; empty string when the target is not a file.
  const similarTargetPath = target.type === 'file' ? target.path : '';

  const paletteRef = useRef<HTMLDivElement>(null);
  const { width, height } = useComponentSize(paletteRef);

  const activeJobs = useActiveJobs(display, authToken);
  const savedWorkflows = useSavedWorkflows(display, authToken);
  const { deps, refreshDeps } = useDeps(display);

  // Generation mode for the metadata chips. Resets to the non-destructive
  // `missing` on every open so a previous `all` (replace) choice is never sticky.
  const [genMode, setGenMode] = useState<GenMode>('missing');
  // Multi-select chip state. Persisted across opens AND app restarts —
  // the picked set is a preference ("these are the ops I generate"), and
  // nothing launches without an explicit Run click, so stickiness is safe
  // (unlike the mode, which deliberately snaps back to `missing`).
  const [selectedTypes, setSelectedTypes] = useState<string[]>(
    loadSelectedTypes
  );
  useEffect(() => {
    if (display) setGenMode('missing');
  }, [display]);
  useEffect(() => {
    try {
      localStorage.setItem(
        SELECTED_TYPES_STORAGE_KEY,
        JSON.stringify(selectedTypes)
      );
    } catch {
      // Storage unavailable (private mode, quota) — selection just won't persist.
    }
  }, [selectedTypes]);

  // Server health. A live SSE connection on the shared stream bus is proof
  // the server is reachable without spending a socket; the /health probe is
  // the fallback when the bus isn't connected. Never latch: while the palette
  // is open and the server looks down, keep re-probing so a recovered server
  // (or a freed-up socket pool) is noticed without closing the palette.
  const [serverAvailable, setServerAvailable] = useState<boolean | null>(null);
  useEffect(() => {
    if (!display) {
      setServerAvailable(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const check = async () => {
      if (cancelled) return;
      if (streamConnected()) {
        setServerAvailable(true);
        return;
      }
      let ok = false;
      try {
        const headers: HeadersInit = {};
        if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
        const res = await fetch(`${mediaServerBase}/health`, {
          method: 'GET',
          headers,
          signal: AbortSignal.timeout(5000),
        });
        ok = res.ok;
      } catch {
        ok = false;
      }
      if (cancelled) return;
      setServerAvailable(ok);
      if (!ok) timer = setTimeout(check, 4000); // retry while open
    };
    check();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
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

  // Close on click outside.
  //
  // In trackpad/touchpad mode the detail view binds its own onClick (cursor
  // advance/decrement). Without intervention the click that dismisses the
  // palette also lands on the detail handler and jumps to a different media
  // item — absorb the trailing `click` in the capture phase so closing never
  // doubles as navigation. (Same treatment as the command palette.)
  useOnClickOutside(paletteRef, (event) => {
    if (!display) return;
    // Defensive: walk up from the click target like the command palette does
    // — its ref-based containment check has been observed to misfire on
    // clicks inside the palette (pill × buttons re-render under the pointer).
    const target = event.target as HTMLElement | null;
    if (
      target &&
      typeof target.closest === 'function' &&
      target.closest('.ContextPalette')
    ) {
      return;
    }
    absorbNextClick();
    libraryService.send('HIDE_CONTEXT_PALETTE');
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

  // Close on library change — but NOT on streaming growth. While a directory
  // load streams in, EVERY LOAD_FILES_BATCH (and the final stream-complete
  // assign) mints a new libraryLoadId for the SAME logical load — that id
  // exists so list views re-key, not to signal a library switch. Closing on
  // those made the palette collapse the instant the next batch arrived. A
  // loadId change counts as a switch only when it didn't originate from an
  // in-flight stream; a genuinely new load STARTED mid-stream is caught by
  // its initialFile change instead.
  const prevLibraryLoadId = useRef(libraryLoadId);
  const prevStreaming = useRef(streaming);
  const prevInitialFile = useRef(initialFile);
  useEffect(() => {
    const loadIdSwitched =
      prevLibraryLoadId.current &&
      prevLibraryLoadId.current !== libraryLoadId &&
      !prevStreaming.current;
    const loadTargetSwitched = prevInitialFile.current !== initialFile;
    if (display && (loadIdSwitched || loadTargetSwitched)) {
      libraryService.send('HIDE_CONTEXT_PALETTE');
    }
    prevLibraryLoadId.current = libraryLoadId;
    prevStreaming.current = streaming;
    prevInitialFile.current = initialFile;
  }, [display, libraryLoadId, streaming, initialFile, libraryService]);

  // Derived data
  const libraryCtx = {
    currentStateType,
    predicates,
    textFilter,
    initialFile,
    settings: { filteringMode, recursive },
  };
  const items = filter(libraryLoadId, textFilter, library, filters, sortBy);
  const itemCount = target.type === 'file' ? 1 : items.length;
  const contextLabel = buildLabel(target, libraryCtx);
  const queryString = buildQuery(target, libraryCtx);
  const isFolderContext =
    target.type === 'library' &&
    !(currentStateType === 'db' && dbQuery.tags.length > 0);
  const encodeQuery64 = (q: string) =>
    btoa(
      new TextEncoder()
        .encode(q)
        .reduce((s, b) => s + String.fromCharCode(b), ''),
    );
  const query64 = encodeQuery64(queryString);

  // Visual/vector similarity search for the right-clicked file. Adds a `similar`
  // predicate to the unified query (no server job) and closes the palette.
  const handleFindSimilar = () => {
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'similar',
          value: similarTargetPath,
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
      },
    });
    libraryService.send('HIDE_CONTEXT_PALETTE');
  };

  // Face-identity search for the right-clicked file: "show me more of this
  // person". Adds a `face` predicate (the server scans the file on the fly if
  // needed, takes its largest face, and matches against the face index).
  const handleFindPerson = () => {
    libraryService.send({
      type: 'ADD_PREDICATE',
      data: {
        predicate: {
          type: 'face',
          value: similarTargetPath,
          exclude: false,
          join: filteringMode === 'OR' ? 'OR' : 'AND',
        },
      },
    });
    libraryService.send('HIDE_CONTEXT_PALETTE');
  };

  // Person context: when the right-clicked tag is a person (its name exists
  // in /api/people — i.e. a People-category tag), the palette offers person
  // actions (rename) that keep the person table and its taxonomy tag in sync.
  // Renaming through the normal tag editor would desync them.
  const [personTarget, setPersonTarget] = useState<{
    id: number;
    name: string;
  } | null>(null);
  const [personRename, setPersonRename] = useState('');
  useEffect(() => {
    setPersonTarget(null);
    setPersonRename('');
    if (!display || target.type !== 'tag' || !authToken) return;
    const controller = new AbortController();
    (async () => {
      try {
        const res = await fetch(`${mediaServerBase}/api/people`, {
          headers: { Authorization: `Bearer ${authToken}` },
          signal: controller.signal,
        });
        if (!res.ok) return;
        const people = (await res.json()) as Array<{ id: number; name: string }>;
        const match = people.find((p) => p.name === (target as { tag: string }).tag);
        if (match) {
          setPersonTarget(match);
          setPersonRename(displayTagLabel(match.name));
        }
      } catch {
        // server unavailable — no person section
      }
    })();
    return () => controller.abort();
  }, [display, target, authToken]);

  const handleRenamePerson = async () => {
    if (
      !personTarget ||
      !personRename.trim() ||
      personRename === displayTagLabel(personTarget.name)
    ) {
      return;
    }
    try {
      const res = await fetch(
        `${mediaServerBase}/api/people/${personTarget.id}/rename`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify({ name: personRename.trim() }),
        }
      );
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }
      // The rename cascades to the person's People-category tag rows on the
      // server; refresh everything that renders those tags client-side.
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      queryClient.invalidateQueries({ queryKey: ['tags-by-path'] });
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'Person renamed',
          message: `${displayTagLabel(personTarget.name)} → ${personRename.trim()}`,
        },
      });
      libraryService.send('HIDE_CONTEXT_PALETTE');
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Rename failed',
          message: e instanceof Error ? e.message : 'Could not rename person',
        },
      });
    }
  };


  // Chip selection: chips toggle; the Run button launches everything the
  // user picked as ONE combined job (plus a parallel faces job when chosen).
  const toggleType = (meta: MetadataType) => {
    // Point-of-use gate: a job whose model/tool isn't installed would only
    // fail later inside its log where casual users never look. Downloadable
    // deps show a Download button right under the chip instead.
    const alreadySelected = selectedTypes.includes(meta.label);
    if (!alreadySelected) {
      const req = TASK_REQUIREMENTS[meta.label];
      const dep = req ? deps.get(req.depId) : undefined;
      if (req && req.kind === 'downloadable' && dep && isDownloadableState(dep.state)) {
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'info',
            title: `${req.feature} needs a one-time download`,
            message: `Use the Download button under the ${meta.label} chip (${fmtSize(dep.size_bytes)}).`,
          },
        });
        return;
      }
      if (req && dep && isDownloadingState(dep.state)) {
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'info',
            title: `${req.feature} is still downloading`,
            message: 'Select this again once the download finishes.',
          },
        });
        return;
      }
    }
    setSelectedTypes((prev) =>
      alreadySelected
        ? prev.filter((l) => l !== meta.label)
        : [...prev, meta.label]
    );
  };

  // Launches the selected chips in the current mode. `missing` fills gaps;
  // `all` replaces (adds `--overwrite`). Everything becomes ONE job: a
  // single op runs as its own task, several run as a `process` job — one
  // pass applying each op per file.
  const handleRunSelected = async () => {
    const selected = METADATA_TYPES.filter((m) =>
      selectedTypes.includes(m.label)
    );
    if (selected.length === 0) return;
    const mode = genMode;
    const overwrite = mode === 'all' ? ' --overwrite' : '';

    // tagcount:<3 only makes sense when filling tag gaps over a folder, and
    // only when Tags runs alone — a combined job's query targets every op.
    const tagsAlone = selected.length === 1 && selected[0].skipTaggedInFolder;
    const effectiveQuery =
      tagsAlone && isFolderContext && mode === 'missing'
        ? `${queryString} tagcount:<3`
        : queryString;
    const effectiveQuery64 =
      effectiveQuery === queryString ? query64 : encodeQuery64(effectiveQuery);

    const ops = selected.flatMap((m) => m.ops);
    const input =
      ops.length === 1
        ? `${ops[0]}${overwrite} --query64=${effectiveQuery64}`
        : `process --ops=${ops.join(',')}${overwrite} --query64=${effectiveQuery64}`;

    try {
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
      const res = await fetch(`${mediaServerBase}/create`, {
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
        `${mediaServerBase}/workflows/${workflow.id}/run`,
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

  if (!display || !canWrite) return null;

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
        <div className="context-header-right">
          {target.type === 'library' && (
            <span className="context-count">{itemCount} items</span>
          )}
          {target.type === 'file' && (
            <span className="context-count">1 file</span>
          )}
          {(capabilities.visualSearch ||
            (serverAvailable && authToken)) &&
            similarTargetPath && (
              <>
                <button
                  className="find-similar-btn"
                  onClick={handleFindSimilar}
                  title="Find visually similar"
                  aria-label="Find visually similar"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <circle cx="11" cy="11" r="7" />
                    <line x1="21" y1="21" x2="16.5" y2="16.5" />
                    <path d="M11 8l.9 1.8 1.9.3-1.4 1.4.3 1.9-1.7-.9-1.7.9.3-1.9L8.2 10.1l1.9-.3z" />
                  </svg>
                </button>
                <button
                  className="find-similar-btn"
                  onClick={handleFindPerson}
                  title="Find this person (face match)"
                  aria-label="Find this person"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="8" r="4" />
                    <path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />
                  </svg>
                </button>
              </>
            )}
        </div>
      </div>

      {personTarget && serverAvailable && authToken && (
        <div className="context-palette-person">
          <span className="action-group-title">Person</span>
          <div className="person-rename-row">
            <input
              className="person-rename-input"
              type="text"
              value={personRename}
              onChange={(e) => setPersonRename(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleRenamePerson();
                e.stopPropagation();
              }}
              placeholder="Person name"
              aria-label="Rename person"
            />
            <button
              type="button"
              className="person-rename-btn"
              onClick={handleRenamePerson}
              disabled={
                !personRename.trim() || personRename === personTarget.name
              }
            >
              Rename
            </button>
          </div>
        </div>
      )}

      {serverAvailable === false && (
        <div className="context-palette-unavailable">
          Job service unavailable at {mediaServerBase || 'this server'}
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
        <div
          className={`generate-block${genMode === 'all' ? ' caution' : ''}`}
        >
          <div className="generate-mode-row">
            <span className="generate-label">Generate</span>
            <div className="mode-toggle" role="radiogroup" aria-label="Generation mode">
              <button
                type="button"
                role="radio"
                aria-checked={genMode === 'missing'}
                className={`mode-opt${genMode === 'missing' ? ' active' : ''}`}
                onClick={() => setGenMode('missing')}
              >
                Missing
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={genMode === 'all'}
                className={`mode-opt${genMode === 'all' ? ' active' : ''}`}
                onClick={() => setGenMode('all')}
              >
                All
              </button>
            </div>
          </div>
          <div className="type-chips">
            {METADATA_TYPES.map((meta) => {
              const req = TASK_REQUIREMENTS[meta.label];
              const dep = req ? deps.get(req.depId) : undefined;
              const needsDownload =
                !!req && req.kind === 'downloadable' && !!dep && isDownloadableState(dep.state);
              const downloading = !!req && !!dep && isDownloadingState(dep.state);
              const isSelected = selectedTypes.includes(meta.label);
              return (
                <button
                  key={meta.label}
                  role="checkbox"
                  aria-checked={isSelected}
                  className={`type-chip${needsDownload ? ' needs-dep' : ''}${
                    isSelected ? ' selected' : ''
                  }`}
                  onClick={() => toggleType(meta)}
                  title={
                    needsDownload
                      ? `${req!.feature} needs a one-time download first`
                      : isSelected
                      ? `Remove ${meta.label.toLowerCase()} from this run`
                      : `Add ${meta.label.toLowerCase()} to this run`
                  }
                >
                  {isSelected && <span className="chip-check">✓</span>}
                  {meta.label}
                  {needsDownload && <span className="dep-badge">setup</span>}
                  {downloading && <span className="dep-badge downloading">…</span>}
                </button>
              );
            })}
          </div>
          <div className="generate-run-row">
            <button
              type="button"
              className="generate-run-btn"
              disabled={selectedTypes.length === 0}
              onClick={handleRunSelected}
              title={
                selectedTypes.length > 1
                  ? `Run ${selectedTypes.length} operations together — one pass per file`
                  : undefined
              }
            >
              Run
              {selectedTypes.length > 1 && (
                <span className="generate-run-count">
                  {selectedTypes.length}
                </span>
              )}
            </button>
            <span className="generate-run-hint">
              {selectedTypes.length === 0
                ? 'Pick one or more operations'
                : selectedTypes.length > 1
                ? 'Together, one pass per file'
                : genMode === 'all'
                ? 'Replaces existing output'
                : 'Fills what’s missing'}
            </span>
          </div>
          <DepRequirementRows deps={deps} onChange={refreshDeps} />
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
                        `${mediaServerBase}/job/${job.id}/cancel`,
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
