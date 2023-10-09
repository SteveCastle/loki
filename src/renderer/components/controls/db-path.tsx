import { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';

import './db-path.css';

export default function DbPathWidget() {
  const { libraryService } = useContext(GlobalStateContext);
  const dbPath = useSelector(
    libraryService,
    (state) => state.context.dbPath,
    (a, b) => a === b
  );
  return (
    <div className="DbPathWidget">
      <input disabled value={dbPath} />
      <button onClick={() => libraryService.send('CHANGE_DB_PATH')}>
        Change
      </button>
    </div>
  );
}
