import React, { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import Layout from './layout';
import useFileDrop from '../../hooks/useFileDrop';

export function Panels() {
  const { libraryService } = useContext(GlobalStateContext);
  const { dropRef, isOver, canDrop } = useFileDrop();

  return (
    <>
      <div className="drag-handle" />
      <div
        ref={dropRef}
        className={`Panels${isOver && canDrop ? ' file-drop-active' : ''}`}
        onContextMenu={(e) => {
          e.preventDefault();
          libraryService.send('SHOW_COMMAND_PALETTE', {
            position: { x: e.clientX, y: e.clientY },
          });
        }}
      >
        <Layout />
      </div>
    </>
  );
}
