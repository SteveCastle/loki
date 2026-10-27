import { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import { send } from '../../platform';
import { useServerInstall } from '../../hooks/useServerInstall';
import { ServerOffer } from './server-install';
import './welcome.css';

// First launch of the desktop app (library state 'welcome'): pick what to
// look at, and answer the one-time media server offer, on one screen —
// instead of a native file picker followed by a modal on top of it.
export function Welcome() {
  const { libraryService } = useContext(GlobalStateContext);
  const { status } = useServerInstall();
  const offerServer =
    !!status && status.supported && !status.prompted && !status.installedPath;

  return (
    <div className="Welcome">
      <div className="welcome-titlebar">
        <div className="windowControls">
          <span className="closeControl" onClick={() => send('shutdown', [])} />
          <span className="windowedControl" onClick={() => send('minimize', [])} />
          <span
            className="fullScreenControl"
            onClick={() => send('toggle-fullscreen', [])}
          />
        </div>
      </div>
      <div className="welcome-body">
        <h1>Welcome to Lowkey Media Viewer</h1>
        <p className="welcome-sub">
          Point it at a folder of images, videos, audio or comics. Nothing is
          copied or uploaded; your files stay where they are.
        </p>
        <div className="welcome-actions">
          <button
            className="primary"
            onClick={() => libraryService.send('SELECT_DIRECTORY')}
          >
            Open a folder
          </button>
          <button onClick={() => libraryService.send('SELECT_FILE')}>
            Open files
          </button>
        </div>
        {offerServer && (
          <div className="welcome-server">
            <ServerOffer showClose={false} />
          </div>
        )}
      </div>
    </div>
  );
}
