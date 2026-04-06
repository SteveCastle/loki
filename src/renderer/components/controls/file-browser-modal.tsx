import { useState, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './file-browser-modal.css';

type FsEntry = {
  name: string;
  path: string;
  isDir: boolean;
  mtimeMs: number;
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
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [parent, setParent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchEntries = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
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
      setSelectedFile(null);
      fetchEntries(entry.path);
    } else if (mode === 'file') {
      setSelectedFile(entry.path);
    }
  };

  const handleEntryDoubleClick = (entry: FsEntry) => {
    if (!entry.isDir && mode === 'file' && resolveModal) {
      resolveModal(entry.path);
      cleanup();
    }
  };

  const handleOpen = () => {
    if (!resolveModal) return;
    if (mode === 'file' && selectedFile) {
      resolveModal(selectedFile);
      cleanup();
    } else if (mode === 'directory' && currentPath) {
      resolveModal(currentPath);
      cleanup();
    }
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

  // Build breadcrumb segments from currentPath
  const breadcrumbs = currentPath
    ? currentPath.replace(/\\/g, '/').split('/').filter(Boolean)
    : [];

  const breadcrumbPath = (index: number) => {
    const isWindows = currentPath.includes('\\') || /^[A-Z]:/.test(currentPath);
    const sep = isWindows ? '\\' : '/';
    const segments = breadcrumbs.slice(0, index + 1);
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
                className={`file-browser-entry${!entry.isDir && selectedFile === entry.path ? ' selected' : ''}`}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => handleEntryDoubleClick(entry)}
              >
                <span className="icon">{entry.isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4'}</span>
                <span className="name">{entry.name}</span>
              </div>
            ))}
        </div>

        <div className="file-browser-footer">
          <button onClick={handleCancel}>Cancel</button>
          <button
            className="primary"
            onClick={handleOpen}
            disabled={mode === 'file' ? !selectedFile : !currentPath}
          >
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
