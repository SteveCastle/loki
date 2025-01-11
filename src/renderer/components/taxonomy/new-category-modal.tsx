import { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';

import useOnClickOutside from '../../hooks/useOnClickOutside';

import './new-modal.css';

type Props = {
  handleClose: () => void;
  // React setState function
  setCategory: (category: string) => void;
  currentValue?: string;
};
export default function NewCategoryModal({
  handleClose,
  setCategory,
  currentValue = '',
}: Props) {
  const [newLabel, setNewLabel] = useState<string>(currentValue);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });

  const queryClient = useQueryClient();

  function handleSubmit() {
    async function submit() {
      if (currentValue) {
        await window.electron.ipcRenderer.invoke('rename-category', [
          currentValue,
          newLabel,
        ]);
      } else {
        await window.electron.ipcRenderer.invoke('create-category', [
          newLabel,
          0,
        ]);
      }
      setNewLabel('');
      setCategory(newLabel);
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      handleClose();
    }
    submit();
  }

  function handleResetOrdering() {
    async function resetOrdering() {
      await window.electron.ipcRenderer.invoke('order-tags', [currentValue]);
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
      handleClose();
    }
    resetOrdering();
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {currentValue ? `Edit Category` : `New Category`}
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
        <div className="input-modal-body">
          <input
            autoFocus
            type="text"
            className="input"
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
          {currentValue ? (
            <button onClick={handleResetOrdering}>Reset Tag Order</button>
          ) : null}
          <button onClick={handleSubmit}>
            {currentValue ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
