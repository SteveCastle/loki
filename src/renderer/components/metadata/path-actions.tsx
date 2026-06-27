import { FC, useEffect, useRef, useState } from 'react';
import copyIcon from '../../../../assets/copy.svg';
import { isElectron, send } from '../../platform';
import './path-actions.css';

interface PathActionsProps {
  path: string;
}

const COPIED_RESET_MS = 1200;

const PathActions: FC<PathActionsProps> = ({ path }) => {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    },
    []
  );

  const handleCopy = async (e: React.MouseEvent) => {
    // The Path heading sits in a section; don't let clicks bubble up.
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => setCopied(false), COPIED_RESET_MS);
    } catch (err) {
      console.error('Failed to copy path: ', err);
    }
  };

  const handleReveal = (e: React.MouseEvent) => {
    e.stopPropagation();
    send('show-item-in-folder', [path]);
  };

  return (
    <div className="path-actions">
      {isElectron && (
        <button
          type="button"
          className="path-action"
          onClick={handleReveal}
          title="Show in file explorer"
          aria-label="Show in file explorer"
        >
          <svg
            className="path-action-icon"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M3 7a2 2 0 0 1 2-2h3.5l2 2H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"
              stroke="#ffffff"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
      <button
        type="button"
        className={`path-action${copied ? ' copied' : ''}`}
        onClick={handleCopy}
        title="Copy file path"
        aria-label="Copy file path"
      >
        {copied ? (
          <svg
            className="path-action-icon check"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M5 13l4 4L19 7"
              stroke="#ffffff"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <img className="path-action-icon" src={copyIcon} alt="" />
        )}
      </button>
    </div>
  );
};

export default PathActions;
