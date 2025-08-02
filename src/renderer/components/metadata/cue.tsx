import { useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { useQueryClient } from '@tanstack/react-query';
import { GlobalStateContext } from '../../state';
import { VttCue } from 'main/parse-vtt';
import { uniqueId } from 'xstate/lib/utils';

type Props = {
  cue: VttCue;
  cueIndex: number;
  mediaPath: string;
  setScrollTop: (scrollTop: number) => void;
  followVideoTime?: boolean;
};

function convertVTTTimestampToSeconds(timestamp: string) {
  const [minutes, seconds] = timestamp.split(':');
  return parseInt(minutes) * 60 + parseFloat(seconds);
}

function usePrevious<T>(value: T): T {
  // The ref object is a generic container whose current property is mutable ...
  // ... and can hold any value, similar to an instance property on a class
  const ref: any = useRef<T>();
  // Store current value in ref
  useEffect(() => {
    ref.current = value;
  }, [value]); // Only re-run if value changes
  // Return previous value (happens before update in useEffect above)
  return ref.current;
}

export function Cue({ cue, cueIndex, mediaPath, setScrollTop, followVideoTime = false }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const queryClient = useQueryClient();
  const { actualVideoTime } = useSelector(
    libraryService,
    (state) => state.context.videoPlayer
  );

  const [isEditing, setIsEditing] = useState(false);
  const [editStartTime, setEditStartTime] = useState(cue.startTime);
  const [editEndTime, setEditEndTime] = useState(cue.endTime);
  const [editText, setEditText] = useState(cue.text);
  const [isSaving, setIsSaving] = useState(false);

  const startTimeInSeconds = convertVTTTimestampToSeconds(cue.startTime);
  const endTimeInSeconds = convertVTTTimestampToSeconds(cue.endTime);
  const isActive =
    actualVideoTime >= startTimeInSeconds && actualVideoTime < endTimeInSeconds;
  const previousIsActive = usePrevious(isActive);
  const ref = useRef<HTMLLIElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  
  useEffect(() => {
    if (isActive && !previousIsActive && ref.current && followVideoTime) {
      setScrollTop(ref.current.offsetTop);
    }
  }, [actualVideoTime, followVideoTime]);

  useEffect(() => {
    if (isEditing && textAreaRef.current) {
      textAreaRef.current.focus();
      textAreaRef.current.setSelectionRange(textAreaRef.current.value.length, textAreaRef.current.value.length);
    }
  }, [isEditing]);

  const handleSave = async () => {
    if (isSaving) return;
    
    setIsSaving(true);
    try {
      await window.electron.modifyTranscript({
        mediaPath,
        cueIndex,
        startTime: editStartTime !== cue.startTime ? editStartTime : undefined,
        endTime: editEndTime !== cue.endTime ? editEndTime : undefined,
        text: editText !== cue.text ? editText : undefined,
      });
      
      // Invalidate the transcript query to refresh the data
      queryClient.invalidateQueries({ queryKey: ['transcript', mediaPath] });
      
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save transcript changes:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditStartTime(cue.startTime);
    setEditEndTime(cue.endTime);
    setEditText(cue.text);
    setIsEditing(false);
  };

  const handleDoubleClick = () => {
    setIsEditing(true);
  };

  const handleTextKeyDown = (e: React.KeyboardEvent) => {
    // Stop event bubbling to prevent hotkey system from capturing
    e.stopPropagation();
    
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  };

  const handleTimeKeyDown = (e: React.KeyboardEvent) => {
    // Stop event bubbling to prevent hotkey system from capturing
    e.stopPropagation();
    
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancel();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    // Stop event bubbling to prevent any interference
    e.stopPropagation();
  };

  const handleCueClick = () => {
    if (!isEditing) {
      const timeStamp = convertVTTTimestampToSeconds(cue.startTime);
      libraryService.send('SET_VIDEO_TIME', {
        timeStamp,
        eventId: uniqueId(),
      });
    }
  };

  return (
    <li
      ref={ref}
      className={['cue-container', isActive ? 'active' : '', isEditing ? 'editing' : ''].join(' ')}
    >
      <div className="cue" onClick={handleCueClick} onDoubleClick={handleDoubleClick}>
        <div className="cue-header">
          <div className="timestamps">
            {isEditing ? (
              <>
                <input
                  type="text"
                  className="time-input start-time"
                  value={editStartTime}
                  onChange={(e) => {
                    handleInputChange(e);
                    setEditStartTime(e.target.value);
                  }}
                  onKeyDown={handleTimeKeyDown}
                  placeholder="00:00:00.000"
                />
                <span className="time-separator">→</span>
                <input
                  type="text"
                  className="time-input end-time"
                  value={editEndTime}
                  onChange={(e) => {
                    handleInputChange(e);
                    setEditEndTime(e.target.value);
                  }}
                  onKeyDown={handleTimeKeyDown}
                  placeholder="00:00:00.000"
                />
              </>
            ) : (
              <>
                <span className="start-time">{cue.startTime}</span>
                <span className="time-separator">→</span>
                <span className="end-time">{cue.endTime}</span>
              </>
            )}
          </div>
          {isEditing && (
            <div className="edit-actions">
              <button 
                className="save-btn" 
                onClick={handleSave} 
                disabled={isSaving}
                title="Save (Ctrl+Enter)"
              >
                {isSaving ? '⟳' : '✓'}
              </button>
              <button 
                className="cancel-btn" 
                onClick={handleCancel}
                title="Cancel (Escape)"
              >
                ✕
              </button>
            </div>
          )}
        </div>
        <div className="cue-content">
          {isEditing ? (
            <textarea
              ref={textAreaRef}
              className="text-input"
              value={editText}
              onChange={(e) => {
                handleInputChange(e);
                setEditText(e.target.value);
              }}
              onKeyDown={handleTextKeyDown}
              placeholder="Transcript text..."
              rows={Math.max(2, editText.split('\n').length)}
            />
          ) : (
            <div className="text-display">{cue.text}</div>
          )}
        </div>
      </div>
      {!isEditing && (
        <div className="edit-hint">Double-click to edit</div>
      )}
    </li>
  );
}
