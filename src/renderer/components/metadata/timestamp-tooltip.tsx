import React, { useState, useRef, useEffect } from 'react';
import { Tooltip } from 'react-tooltip';
import './timestamp-tooltip.css';

function formatTimestamp(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds - hours * 3600) / 60);
  const secs = Math.floor(seconds - hours * 3600 - minutes * 60);

  const hoursString = hours < 10 ? `0${hours}` : `${hours}`;
  const minutesString = minutes < 10 ? `0${minutes}` : `${minutes}`;
  const secondsString = secs < 10 ? `0${secs}` : `${secs}`;

  return `${hoursString}:${minutesString}:${secondsString}`;
}

function parseTimestamp(timeString: string): number {
  const parts = timeString.split(':');
  if (parts.length !== 3) return 0;
  
  const hours = parseInt(parts[0], 10) || 0;
  const minutes = parseInt(parts[1], 10) || 0;
  const seconds = parseInt(parts[2], 10) || 0;
  
  return hours * 3600 + minutes * 60 + seconds;
}

interface TimestampTooltipProps {
  id: string;
  timestamp: number;
  onEdit: (newTimestamp: number) => void;
  onRemove: () => void;
}

export default function TimestampTooltip({ 
  id, 
  timestamp, 
  onEdit, 
  onRemove 
}: TimestampTooltipProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(formatTimestamp(timestamp));
  const [isValid, setIsValid] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleEdit = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setIsEditing(true);
    setEditValue(formatTimestamp(timestamp));
  };

  const handleSave = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    const newTimestamp = parseTimestamp(editValue);
    if (newTimestamp >= 0) {
      onEdit(newTimestamp);
      setIsEditing(false);
      setIsValid(true);
    } else {
      setIsValid(false);
    }
  };

  const handleCancel = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setIsEditing(false);
    setEditValue(formatTimestamp(timestamp));
    setIsValid(true);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    setEditValue(e.target.value);
    const parsed = parseTimestamp(e.target.value);
    setIsValid(parsed >= 0);
  };

  const handleRemove = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    onRemove();
  };

  const handleContentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  const tooltipContent = (
    <div className="timestamp-tooltip-content" onClick={handleContentClick}>
      {isEditing ? (
        <div className="timestamp-editor">
          <input
            ref={inputRef}
            type="text"
            value={editValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            className={`timestamp-input ${!isValid ? 'invalid' : ''}`}
            placeholder="HH:MM:SS"
            pattern="^\d{2}:\d{2}:\d{2}$"
            onClick={handleContentClick}
          />
          <div className="timestamp-buttons">
            <button 
              className="save-btn" 
              onClick={handleSave}
              disabled={!isValid}
              title="Save changes"
            >
              ✓
            </button>
            <button 
              className="cancel-btn" 
              onClick={handleCancel}
              title="Cancel editing"
            >
              ✕
            </button>
          </div>
          {!isValid && (
            <div className="error-message">Invalid time format (use HH:MM:SS)</div>
          )}
        </div>
      ) : (
        <div className="timestamp-display">
          <div className="timestamp-value" onClick={handleContentClick}>
            {formatTimestamp(timestamp)}
          </div>
          <div className="timestamp-controls">
            <button 
              className="edit-btn"
              onClick={handleEdit}
              title="Edit timestamp"
            >
              ✏️
            </button>
            <button 
              className="remove-btn"
              onClick={handleRemove}
              title="Remove timestamp"
            >
              🗑️
            </button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <Tooltip
      id={id}
      clickable
      place="top"
      className="timestamp-tooltip"
      openOnClick={false}
      closeOnClick={false}
      events={['hover']}
      float={true}
      positionStrategy="fixed"
      offset={10}
    >
      {tooltipContent}
    </Tooltip>
  );
}