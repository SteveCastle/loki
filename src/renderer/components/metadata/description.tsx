import { Metadata } from 'main/metadata';
import { useEffect, useRef, useState } from 'react';
import { invoke } from '../../platform';
import './description.css';
import { debounce } from 'lodash';
import { useQueryClient } from '@tanstack/react-query';
import GenerateDescription from './generate-description';
import { useCanWrite } from '../../hooks/useCanWrite';

export function Description({ path, data }: { path: string; data: Metadata }) {
  const canWrite = useCanWrite();
  const [editing, setEditing] = useState(false);
  const [description, setDescription] = useState(data?.description || '');
  const client = useQueryClient();

  // Update local description state when data changes
  useEffect(() => {
    setDescription(data?.description || '');
  }, [data?.description]);

  const updateDescription = async (newValue: string) => {
    console.log('updateDescription', path, newValue);
    await invoke('update-description', [
      path,
      newValue,
    ]);
    client.invalidateQueries(['file-metadata', path]);
  };
  const debouncedUpdateDescription = useRef(
    debounce((newValue: string) => {
      updateDescription(newValue);
    }, 500)
  );

  useEffect(() => {
    if (description !== data?.description) {
      debouncedUpdateDescription.current(description);
    }
  }, [description]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // Reset height to recalculate
      textareaRef.current.style.height =
        textareaRef.current.scrollHeight + 'px'; // Set new height
    }
  }, [description, editing]); // Runs whenever text changes

  return (
    <div className="Description">
      {editing ? (
        <textarea
          className="description-editor"
          ref={textareaRef}
          autoFocus
          value={description}
          onKeyDown={(e) => {
            e.stopPropagation();
          }}
          onKeyUp={(e) => {
            e.stopPropagation();
          }}
          onChange={(e) => {
            e.stopPropagation();
            setDescription(e.target.value);
          }}
          onBlur={() => {
            setEditing(false);
          }}
        />
      ) : (
        <div
          title={canWrite ? 'Double-click to enter text manually' : undefined}
          onDoubleClick={() => canWrite && setEditing(true)}
          className={`description-display ${
            !description ? 'empty-description' : ''
          }`}
        >
          {description ? (
            description
          ) : (
            <div className="empty-content">
              {canWrite && (
                <span className="placeholder-text">
                  Double-click to enter text or use Generate
                </span>
              )}
            </div>
          )}
        </div>
      )}
      {/* The generate/regenerate action lives in the section's top-right corner
          (not in the text flow), so the empty, read and edit views all share
          identical layout — no shift. Same affordance for the initial generate
          and subsequent regenerate, just a different label. */}
      <GenerateDescription
        path={path}
        label={description ? 'Regenerate' : 'Generate'}
        variant="corner"
      />
    </div>
  );
}
