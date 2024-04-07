import React, { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import './media-error.css';

export default function MediaError({ path }: { path: string }) {
  const { libraryService } = useContext(GlobalStateContext);

  return (
    <div className="MediaError">
      <p>Media unavailable.</p>
      <button onClick={() => libraryService.send('UPDATE_FILE_PATH', { path })}>
        Find Media
      </button>
      {/* <button
        onClick={() =>
          libraryService.send('UPDATE_FILE_PATH', { path, updateAll: true })
        }
      >
        Find All With Same Base Path
      </button> */}
    </div>
  );
}
