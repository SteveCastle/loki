import React, { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import Layout from './layout';

export function Panels() {
  const { libraryService } = useContext(GlobalStateContext);

  return (
    <>
      <div className="drag-handle" />
      <div
        className="Panels"
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
