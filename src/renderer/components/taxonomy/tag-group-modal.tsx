import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';
import { TagGroup, renameCollapsedGroup } from './tag-groups';

import './new-modal.css';

type Props = {
  categoryLabel: string;
  // The category's existing groups, to refuse a duplicate name.
  groups: TagGroup[];
  handleClose: () => void;
  // Set when renaming; absent when creating.
  currentValue?: string;
};

// Create or rename a tag group within one category.
export default function TagGroupModal({
  categoryLabel,
  groups,
  handleClose,
  currentValue = '',
}: Props) {
  const [label, setLabel] = useState<string>(currentValue);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  const isEditing = Boolean(currentValue);

  const trimmed = label.trim();
  const duplicate =
    trimmed !== currentValue && groups.some((g) => g.label === trimmed);
  const canSubmit = trimmed.length > 0 && !duplicate;

  async function handleSubmit() {
    if (!canSubmit) return;
    try {
      if (isEditing) {
        if (trimmed !== currentValue) {
          await invoke('rename-tag-group', [
            categoryLabel,
            currentValue,
            trimmed,
          ]);
          renameCollapsedGroup(categoryLabel, currentValue, trimmed);
        }
      } else {
        await invoke('create-tag-group', [categoryLabel, trimmed]);
      }
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      handleClose();
    } catch (err) {
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: isEditing ? 'Failed to rename group' : 'Failed to create group',
          message: String(err),
        },
      });
    }
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {isEditing ? 'Rename Group' : `New Group in ${categoryLabel}`}
          </div>
          <div className="input-modal-close" onClick={handleClose}>
            <img src={cancel} />
          </div>
        </div>

        <div className="input-modal-properties">
          <label>Name</label>
          <input
            autoFocus
            type="text"
            value={label}
            onChange={(e) => {
              e.stopPropagation();
              setLabel(e.currentTarget.value);
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') handleSubmit();
              if (e.key === 'Escape') handleClose();
            }}
          />
          {duplicate && (
            <div className="input-modal-hint">
              {categoryLabel} already has a group with that name.
            </div>
          )}
        </div>

        <div className="input-modal-divider" />
        <div className="input-modal-footer">
          <button className="btn-cancel" onClick={handleClose}>
            Cancel
          </button>
          <button
            className="btn-save"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {isEditing ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
