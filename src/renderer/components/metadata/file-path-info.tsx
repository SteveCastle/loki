import React, { useCallback } from 'react';
import './file-path-info.css';

interface FilePathInfoProps {
  path: string;
  // Navigates the library to the file's folder.
  onNavigate: () => void;
}

// Keep the press away from the item/detail handlers underneath: a click
// selects or advances, a double click swaps list and detail, a pointer
// press can start a drag.
const stop = (e: React.SyntheticEvent) => e.stopPropagation();

// The path strip shown along the bottom of list items and the detail view.
// Only the folder button navigates; the path text itself is inert.
export default function FilePathInfo({ path, onNavigate }: FilePathInfoProps) {
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onNavigate();
    },
    [onNavigate]
  );
  return (
    <div className="item-info">
      <button
        type="button"
        className="file-path-folder"
        title="Go to this folder in the library"
        aria-label="Go to this folder in the library"
        onClick={handleClick}
        onDoubleClick={stop}
        onPointerDown={stop}
        onMouseDown={stop}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M3 7a2 2 0 0 1 2-2h3.5l2 2H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      <span className="file-path">{path}</span>
    </div>
  );
}
