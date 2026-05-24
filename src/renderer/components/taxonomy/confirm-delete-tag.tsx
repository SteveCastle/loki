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
  thumbnail_path_600?: string | null;
};

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
      // Optimistic delete. The tag could live in any loaded per-category
      // cache; we don't know its category from props, so scan every
      // ['taxonomy', 'category-tags', *] cache (plus the all-tags cache)
      // and remove the tag wherever it shows up.
      const tagsSnapshot = queryClient.getQueriesData<CachedConcept[]>({
        queryKey: ['taxonomy', 'category-tags'],
      });
      const allSnapshot = queryClient.getQueriesData<CachedConcept[]>({
        queryKey: ['taxonomy', 'all-tags'],
      });

      const removeFromList = (
        old: CachedConcept[] | undefined
      ): CachedConcept[] | undefined => {
        if (!old) return old;
        const next = old.filter((t) => t.label !== targetLabel);
        return next.length === old.length ? old : next;
      };

      queryClient.setQueriesData<CachedConcept[]>(
        { queryKey: ['taxonomy', 'category-tags'] },
        removeFromList
      );
      queryClient.setQueriesData<CachedConcept[]>(
        { queryKey: ['taxonomy', 'all-tags'] },
        removeFromList
      );
      handleClose();

      try {
        await invoke('delete-tag', [targetLabel]);
        queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
        queryClient.invalidateQueries({ queryKey: ['metadata'] });
        queryClient.invalidateQueries(['tags-by-path']);
      } catch (err) {
        console.error('[confirm-delete-tag] delete failed', err);
        for (const [key, value] of tagsSnapshot) {
          queryClient.setQueryData(key, value);
        }
        for (const [key, value] of allSnapshot) {
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
