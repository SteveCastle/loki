import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

import './new-modal.css';

type CachedConcept = {
  label: string;
  category: string;
  weight: number;
  description?: string;
};
type CachedCategory = {
  label: string;
  tags: CachedConcept[];
  description: string;
};
type TaxonomyCache = { [key: string]: CachedCategory };

type Props = {
  categoryLabel: string;
  handleClose: () => void;
  currentValue?: string;
  currentDescription?: string;
};

export default function NewTagModal({
  categoryLabel,
  handleClose,
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
        try {
          if (newLabel !== currentValue) {
            await invoke('rename-tag', [currentValue, newLabel]);
          }
          if (description !== currentDescription) {
            await invoke('update-tag-description', [newLabel, description]);
          }
          setNewLabel('');
          handleClose();
          queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
          queryClient.invalidateQueries({ queryKey: ['metadata'] });
        } catch (err) {
          console.error('[new-tag-modal] edit failed', err);
          libraryService.send({
            type: 'ADD_TOAST',
            data: {
              type: 'error',
              title: 'Failed to update tag',
              message: String(err),
            },
          });
        }
        return;
      }

      // Optimistic create: snapshot, mutate cache, close modal, then fire IPC.
      const snapshot = queryClient.getQueriesData<TaxonomyCache>({
        queryKey: ['taxonomy'],
      });
      queryClient.setQueriesData<TaxonomyCache>(
        { queryKey: ['taxonomy'] },
        (old) => {
          if (!old) return old;
          const existing = old[categoryLabel];
          if (!existing) return old;
          const maxWeight = existing.tags.reduce(
            (m, t) => Math.max(m, t.weight ?? 0),
            0
          );
          const newTag = {
            label: newLabel,
            category: categoryLabel,
            weight: maxWeight + 1,
            description,
          };
          return {
            ...old,
            [categoryLabel]: {
              ...existing,
              tags: [...existing.tags, newTag],
            },
          };
        }
      );
      const submittedLabel = newLabel;
      setNewLabel('');
      handleClose();

      try {
        await invoke('create-tag', [submittedLabel, categoryLabel, 0]);
        if (description) {
          await invoke('update-tag-description', [submittedLabel, description]);
        }
        queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        queryClient.invalidateQueries({ queryKey: ['metadata'] });
      } catch (err) {
        console.error('[new-tag-modal] create failed', err);
        for (const [key, value] of snapshot) {
          queryClient.setQueryData(key, value);
        }
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'error',
            title: 'Failed to create tag',
            message: String(err),
          },
        });
      }
    }
    submit();
  }

  async function handleApplyElo() {
    try {
      const result = await invoke('apply-elo-ordering', [currentValue]);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'success',
          title: `Reordered ${(result as any).count} items by ELO ranking`,
        },
      });
      libraryService.send({ type: 'SORTED_WEIGHTS' });
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    } catch (e) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to apply ELO ordering',
          message: String(e),
        },
      });
    }
  }

  async function handleConsolidateFiles() {
    try {
      const targetDir = await invoke('select-directory', [undefined]);
      if (!targetDir) return;

      const result = await invoke('consolidate-tag-files', [
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
            {isEditing ? 'Edit Tag' : 'New Tag'}
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
            value={newLabel}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
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
            placeholder="Optional notes about this tag..."
          />
        </div>

        {isEditing && (
          <>
            <div className="input-modal-divider" />
            <div className="input-modal-actions">
              <div className="input-modal-actions-label">Actions</div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Apply ELO as custom order
                  </div>
                  <div className="action-row-description">
                    Seed custom sort weights from Battle Mode ELO rankings
                  </div>
                </div>
                <button onClick={handleApplyElo}>Apply</button>
              </div>
              <div className="action-row">
                <div className="action-row-text">
                  <div className="action-row-title">
                    Consolidate files to directory
                  </div>
                  <div className="action-row-description">
                    Copy tagged files into a chosen folder and update references
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
