import { Metadata } from 'main/metadata';
import { useEffect, useRef, useState } from 'react';
import './description.css';
import { debounce } from 'lodash';
import { useQueryClient } from '@tanstack/react-query';

export function Description({ path, data }: { path: string; data: Metadata }) {
  const [editing, setEditing] = useState(false);
  const [description, setDescription] = useState(data?.description || '');
  const client = useQueryClient();

  const updateDescription = async (newValue: string) => {
    console.log('updateDescription', path, newValue);
    await window.electron.ipcRenderer.invoke('update-description', [
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
        <div title="Click to edit" onClick={() => setEditing(true)}>
          {description}
        </div>
      )}
    </div>
  );
}
