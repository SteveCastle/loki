import { useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import useOnClickOutside from '../../hooks/useOnClickOutside';

import './confirm-delete-category.css';

type Props = {
  handleClose: () => void;
  currentValue?: string;
};
export default function ConfirmDeleteCategory({
  handleClose,
  currentValue,
}: Props) {
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });
  const queryClient = useQueryClient();
  function handleSubmit(e: React.MouseEvent<HTMLButtonElement, MouseEvent>) {
    e.stopPropagation();
    async function submit() {
      await window.electron.ipcRenderer.invoke('delete-category', [
        currentValue,
      ]);
      handleClose();
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      queryClient.invalidateQueries(['tags-by-path']);
    }
    submit();
  }

  return (
    <div className="ConfirmDeleteCategory" ref={ref}>
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
