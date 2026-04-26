import { useContext, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import useOnClickOutside from '../../hooks/useOnClickOutside';
import { invoke } from '../../platform';
import { GlobalStateContext } from '../../state';

import './confirm-delete-tag.css';

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
  handleClose: () => void;
  currentValue?: string;
};
export default function ConfirmDeleteTag({ handleClose, currentValue }: Props) {
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);
  function handleSubmit(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.stopPropagation();
    if (!currentValue) {
      handleClose();
      return;
    }
    const targetLabel = currentValue;
    async function submit() {
      // Optimistic delete: snapshot, remove from cache, close modal, then fire IPC.
      const snapshot = queryClient.getQueriesData<TaxonomyCache>({
        queryKey: ['taxonomy'],
      });
      queryClient.setQueriesData<TaxonomyCache>(
        { queryKey: ['taxonomy'] },
        (old) => {
          if (!old) return old;
          let mutated = false;
          const next: TaxonomyCache = {};
          for (const [catLabel, cat] of Object.entries(old)) {
            if (cat.tags.some((t) => t.label === targetLabel)) {
              next[catLabel] = {
                ...cat,
                tags: cat.tags.filter((t) => t.label !== targetLabel),
              };
              mutated = true;
            } else {
              next[catLabel] = cat;
            }
          }
          return mutated ? next : old;
        }
      );
      handleClose();

      try {
        await invoke('delete-tag', [targetLabel]);
        queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        queryClient.invalidateQueries({ queryKey: ['metadata'] });
        queryClient.invalidateQueries(['tags-by-path']);
      } catch (err) {
        console.error('[confirm-delete-tag] delete failed', err);
        for (const [key, value] of snapshot) {
          queryClient.setQueryData(key, value);
        }
        libraryService.send({
          type: 'ADD_TOAST',
          data: {
            type: 'error',
            title: 'Failed to delete tag',
            message: String(err),
          },
        });
      }
    }
    submit();
  }

  return (
    <div className="ConfirmDeleteTag" ref={ref}>
      <div className="buttons">
        <button onClick={handleSubmit} className="confirm">
          Delete
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            handleClose();
          }}
          className="cancel"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
