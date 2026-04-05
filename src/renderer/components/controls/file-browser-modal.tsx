import { useState, useEffect, useCallback, useRef } from 'react';
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

let resolveModal: ((path: string) => void) | null = null;
let rejectModal: (() => void) | null = null;
let mountContainer: HTMLDivElement | null = null;
let root: ReturnType<typeof createRoot> | null = null;

function FileBrowserModal() {
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
      fetchEntries(entry.path);
    }
  };

  const handleOpen = () => {
    if (resolveModal && currentPath) {
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
          <h3>Browse Directory</h3>
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
                className="file-browser-entry"
                onClick={() => handleEntryClick(entry)}
              >
                <span className="icon">{entry.isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4'}</span>
                <span className="name">{entry.name}</span>
              </div>
            ))}
        </div>

        <div className="file-browser-footer">
          <button onClick={handleCancel}>Cancel</button>
          <button className="primary" onClick={handleOpen} disabled={!currentPath}>
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
 * with the selected directory path (bare string) or rejects on cancel.
 */
export function openFileBrowser(): Promise<string> {
  cleanup();

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
