import { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { send, capabilities } from '../../platform';
import './setup-wizard.css';

export function SetupWizard() {
  const { libraryService } = useContext(GlobalStateContext);
  const dbPath = useSelector(
    libraryService,
    (state) => state.context.dbPath,
    (a, b) => {
      return a === b;
    }
  );
  return (
    <div className="SetupWizard">
      <div className="menuBar">
        <div className="windowControls">
          <span
            className="closeControl"
            onClick={() =>
              send('shutdown', [])
            }
          />
          <span
            className="windowedControl"
            onClick={() =>
              send('minimize', [])
            }
          />
          <span
            className="fullScreenControl"
            onClick={() =>
              send('toggle-fullscreen', [])
            }
          />
        </div>
      </div>
      <div className="innerContainer">
        <h1>Setup Wizard</h1>
        <p className={'path'}>{dbPath}</p>
        <button
          onClick={() => {
            libraryService.send('SELECT_DB');
          }}
        >
          Select Database
        </button>
      </div>
    </div>
  );
}
