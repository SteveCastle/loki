import { useContext, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import cancel from '../../../../assets/cancel.svg';
import useOnClickOutside from '../../hooks/useOnClickOutside';

import './new-modal.css';
import { GlobalStateContext } from 'renderer/state';

type Props = {
  categoryLabel: string;
  handleClose: () => void;
  currentValue?: string;
};

export default function NewTagModal({
  categoryLabel,
  handleClose,
  currentValue = '',
}: Props) {
  const [newLabel, setNewLabel] = useState<string>(currentValue);
  const ref = useRef(null);
  useOnClickOutside(ref, () => {
    handleClose();
  });
  const queryClient = useQueryClient();
  const { libraryService } = useContext(GlobalStateContext);

  function handleApplyELO() {
    async function applyELO() {
      await window.electron.ipcRenderer.invoke('apply-elo', [currentValue]);
      libraryService.send('REFRESH');
      handleClose();
    }
    applyELO();
  }

  function handleApplyWeight() {
    async function applyWeight() {
      await window.electron.ipcRenderer.invoke('apply-weight', [currentValue]);
      libraryService.send('REFRESH');
      handleClose();
    }
    applyWeight();
  }

  function handleSubmit() {
    async function submit() {
      if (currentValue) {
        await window.electron.ipcRenderer.invoke('rename-tag', [
          currentValue,
          newLabel,
        ]);
      } else {
        await window.electron.ipcRenderer.invoke('create-tag', [
          newLabel,
          categoryLabel,
          0,
        ]);
      }
      setNewLabel('');
      handleClose();
      queryClient.invalidateQueries({ queryKey: ['taxonomy'] });
      queryClient.invalidateQueries({ queryKey: ['metadata'] });
    }
    submit();
  }

  return (
    <div className="input-modal">
      <div className="input-modal-content" ref={ref}>
        <div className="input-modal-header">
          <div className="input-modal-title">
            {currentValue ? `Edit Tag` : `New Tag`}
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
            onChange={(e) => setNewLabel(e.currentTarget.value)}
            value={newLabel}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSubmit();
              }
            }}
          />
          <button onClick={handleApplyELO}>Apply ELO</button>
          <button onClick={handleApplyWeight}>Apply Weight</button>
          <button onClick={handleSubmit}>
            {currentValue ? 'Save' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
