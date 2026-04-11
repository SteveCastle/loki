import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';

import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

import './new-modal.css';

type Props = {
  handleClose: () => void;
  setCategory: (category: string) => void;
  currentValue?: string;
  currentDescription?: string;
};

export default function NewCategoryModal({
  handleClose,
  setCategory,
  currentValue = '',
  currentDescription = '',
}: Props) {
  const [newLabel, setNewLabel] = useState<string>(currentValue);
  const [description, setDescription] = useState<string>(currentDescription);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });

  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const isEditing = Boolean(currentValue);

  function handleSubmit() {
    async function submit() {
      if (isEditing) {
        if (newLabel !== currentValue) {
          await invoke('rename-category', [currentValue, newLabel]);
        }
        if (description !== currentDescription) {
          await invoke('update-category-description', [
            newLabel,
            description,
          ]);
        }
      } else {
        await invoke('create-category', [newLabel, 0]);
        if (description) {
          await invoke('update-category-description', [
            newLabel,
            description,
          ]);
        }
      }
      setNewLabel('');
      setCategory(newLabel);
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      handleClose();
    }
    submit();
  }

  async function handleResetOrdering() {
    try {
      await invoke('order-tags', [currentValue]);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: 'Tag order reset to alphabetical',
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to reset tag order',
          message: String(e),
        },
      });
    }
  }

  async function handleConsolidateFiles() {
    try {
      const targetDir = await invoke('select-directory', [undefined]);
      if (!targetDir) return;

      const result = await invoke('consolidate-category-files', [
        currentValue,
        targetDir,
      ]);
      const { copied, errors } = result as any;
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: errors > 0 ? 'error' : 'success',
          title: `Copied ${copied} files to ${targetDir}`,
          message: errors > 0 ? `${errors} files failed to copy` : undefined,
        },
      });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to consolidate files',
          message: String(e),
        },
      });
    }
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {isEditing ? 'Edit Category' : 'New Category'}
          </div>
          <div
            className="input-modal-close"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            <img src={cancel} />
          </div>
        </div>

        <div className="input-modal-properties">
          <label>Name</label>
          <input
            autoFocus
            type="text"
            onChange={(e) => {
              e.stopPropagation();
              setNewLabel(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
            value={newLabel}
          />
          <label>Description</label>
          <textarea
            value={description}
            onChange={(e) => {
              e.stopPropagation();
              setDescription(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
            }}
            placeholder="Optional notes about this category..."
          />
        </div>

        {isEditing && (
          <>
            <div className="input-modal-divider" />
            <div className="input-modal-actions">
              <div className="input-modal-actions-label">Actions</div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">Reset tag order</div>
                  <div className="action-row-description">
                    Alphabetically sort all tags in this category
                  </div>
                </div>
                <button onClick={handleResetOrdering}>Reset</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Consolidate files to directory
                  </div>
                  <div className="action-row-description">
                    Copy all files in this category into a chosen folder
                  </div>
                </div>
                <button onClick={handleConsolidateFiles}>Choose...</button>
              </div>
            </div>
          </>
        )}

        <div className="input-modal-divider" />
        <div className="input-modal-footer">
          <button
            className="btn-cancel"
            onClick={() => {
              setNewLabel('');
              handleClose();
            }}
          >
            Cancel
          </button>
          <button className="btn-save" onClick={handleSubmit}>
            {isEditing ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
