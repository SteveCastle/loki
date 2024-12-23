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

  return (
    <div className="Description">
      {editing ? (
        <textarea
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
