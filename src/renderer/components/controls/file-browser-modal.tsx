import { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './file-browser-modal.css';

type FsEntry = {
  name: string;
  path: string;
  isDir: boolean;
  mtimeMs: number;
  type?: 'local' | 's3';
};

type FsListResponse = {
  entries: FsEntry[];
  parent: string | null;
  roots: string[];
};

type BrowseMode = 'directory' | 'file';

let resolveModal: ((path: string) => void) | null = null;
let rejectModal: (() => void) | null = null;
let mountContainer: HTMLDivElement | null = null;
let root: ReturnType<typeof createRoot> | null = null;
let currentMode: BrowseMode = 'directory';

function FileBrowserModal() {
  const mode = currentMode;
  // Selected entry (file in file mode, directory in directory mode) —
  // standard picker semantics: single-click selects, double-click opens.
  const [selected, setSelected] = useState<FsEntry | null>(null);
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [parent, setParent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchEntries = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const res = await fetch('/api/fs/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ path }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Error ${res.status}`);
      }
      const data: FsListResponse = await res.json();
      setEntries(data.entries || []);
      setParent(data.parent);
      setCurrentPath(path);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEntries('');
  }, [fetchEntries]);

  const handleEntryClick = (entry: FsEntry) => {
    if (entry.isDir) {
      // Single-click SELECTS the folder (so "Open" loads it); double-click
      // navigates into it. In file mode a folder can't be the final answer,
      // so a single click just navigates like before.
      if (mode === 'directory') {
        setSelected(entry);
      } else {
        setSelected(null);
        fetchEntries(entry.path);
      }
    } else if (mode === 'file') {
      setSelected(entry);
    }
  };

  const handleEntryDoubleClick = (entry: FsEntry) => {
    if (entry.isDir) {
      setSelected(null);
      fetchEntries(entry.path);
      return;
    }
    if (mode === 'file' && resolveModal) {
      resolveModal(entry.path);
      cleanup();
    }
  };

  // What "Open" resolves to: the selected entry, or (directory mode) the
  // directory currently being browsed.
  const openTarget =
    mode === 'file'
      ? selected && !selected.isDir
        ? selected.path
        : null
      : selected?.isDir
        ? selected.path
        : currentPath || null;

  const handleOpen = () => {
    if (!resolveModal || !openTarget) return;
    resolveModal(openTarget);
    cleanup();
  };

  const handleCancel = () => {
    if (rejectModal) {
      rejectModal();
    }
    cleanup();
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      handleCancel();
    }
  };

  // Build breadcrumb segments from currentPath. S3 paths keep their scheme
  // out of the segments and reconstruct as s3://bucket/prefix/.
  const isS3 = currentPath.startsWith('s3://');
  const breadcrumbs = currentPath
    ? (isS3 ? currentPath.slice('s3://'.length) : currentPath)
        .replace(/\\/g, '/')
        .split('/')
        .filter(Boolean)
    : [];

  const breadcrumbPath = (index: number) => {
    const segments = breadcrumbs.slice(0, index + 1);
    if (isS3) {
      return 's3://' + segments.join('/') + '/';
    }
    const isWindows = currentPath.includes('\\') || /^[A-Z]:/.test(currentPath);
    const sep = isWindows ? '\\' : '/';
    if (isWindows) {
      const p = segments.join(sep);
      return /^[A-Z]:$/.test(p) ? p + sep : p;
    }
    return sep + segments.join(sep);
  };

  return (
    <div className="file-browser-overlay" onClick={handleOverlayClick}>
      <div className="file-browser-modal">
        <div className="file-browser-header">
          <h3>{mode === 'file' ? 'Select File' : 'Browse Directory'}</h3>
        </div>

        {currentPath && (
          <div className="file-browser-breadcrumb">
            <span onClick={() => fetchEntries('')}>Root</span>
            {breadcrumbs.map((seg, i) => (
              <span key={i}>
                <span className="separator"> / </span>
                <span onClick={() => fetchEntries(breadcrumbPath(i))}>{seg}</span>
              </span>
            ))}
          </div>
        )}

        <div className="file-browser-entries">
          {loading && <div className="file-browser-loading">Loading...</div>}
          {error && <div className="file-browser-empty">{error}</div>}
          {!loading && !error && entries.length === 0 && (
            <div className="file-browser-empty">Empty directory</div>
          )}
          {!loading && !error && parent !== null && (
            <div className="file-browser-entry" onClick={() => fetchEntries(parent!)}>
              <span className="icon">..</span>
              <span className="name">(parent directory)</span>
            </div>
          )}
          {!loading &&
            !error &&
            entries.map((entry) => (
              <div
                key={entry.path}
                className={`file-browser-entry${selected?.path === entry.path ? ' selected' : ''}`}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => handleEntryDoubleClick(entry)}
              >
                <span className="icon">{entry.isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4'}</span>
                <span className="name">{entry.name}</span>
                {entry.type === 's3' && <span className="badge-s3">S3</span>}
              </div>
            ))}
        </div>

        <div className="file-browser-footer">
          {mode === 'directory' && openTarget && (
            <span className="file-browser-open-target" title={openTarget}>
              {openTarget}
            </span>
          )}
          <button onClick={handleCancel}>Cancel</button>
          <button className="primary" onClick={handleOpen} disabled={!openTarget}>
            Open
          </button>
        </div>
      </div>
    </div>
  );
}

function cleanup() {
  resolveModal = null;
  rejectModal = null;
  if (root) {
    root.unmount();
    root = null;
  }
  if (mountContainer) {
    mountContainer.remove();
    mountContainer = null;
  }
}

/**
 * Opens the file browser modal and returns a promise that resolves
 * with the selected path (bare string) or rejects on cancel.
 */
export function openFileBrowser(mode: BrowseMode = 'directory'): Promise<string> {
  cleanup();
  currentMode = mode;

  return new Promise<string>((resolve, reject) => {
    resolveModal = resolve;
    rejectModal = reject;

    mountContainer = document.createElement('div');
    mountContainer.id = 'file-browser-mount';
    document.body.appendChild(mountContainer);

    root = createRoot(mountContainer);
    root.render(<FileBrowserModal />);
  });
}
