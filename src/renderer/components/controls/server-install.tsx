import React, { useEffect, useState } from 'react';
import { isElectron } from '../../platform';
import {
  useServerInstall,
  isServerInstallBusy,
  describeServerInstall,
} from '../../hooks/useServerInstall';
import './server-install.css';

// Palette (server tab) control: offers to install the media server when it
// isn't on this machine, or to start it when it is installed but not
// running. Renders nothing once the server answers.
export function ServerInstallWidget() {
  const { status, progress, install, start } = useServerInstall();
  if (!isElectron || !status) return null;

  const busy = isServerInstallBusy(progress);
  if (status.running && !busy && progress.phase !== 'error') return null;
  if (!status.supported && !status.installedPath) return null;

  const installed = !!status.installedPath;
  return (
    <div className="ServerInstallWidget">
      <div className="server-install-label">
        {installed
          ? 'Lowkey Media Server is installed but not running.'
          : 'Lowkey Media Server is not installed.'}
      </div>
      {(busy || progress.phase === 'error') && (
        <div
          className={`server-install-progress ${
            progress.phase === 'error' ? 'error' : ''
          }`}
        >
          {describeServerInstall(progress)}
        </div>
      )}
      <button disabled={busy} onClick={() => (installed ? start() : install())}>
        {installed ? 'Start server' : 'Install media server'}
      </button>
    </div>
  );
}

// How long after mount to look for a server. The check is a localhost probe
// plus a few fs.existsSync calls in main — cheap, but nothing about it should
// sit on the launch → first-media path.
const PROMPT_DELAY_MS = 2500;

// First-run offer, shown once: install and start the server, or decline and
// never be asked again (the palette widget remains as the way back).
export function ServerInstallPrompt() {
  const [armed, setArmed] = useState(false);
  const [open, setOpen] = useState(false);
  const { status, progress, install, dismissPrompt } =
    useServerInstall(armed);

  useEffect(() => {
    if (!isElectron) return undefined;
    const t = setTimeout(() => setArmed(true), PROMPT_DELAY_MS);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (status && !status.prompted && status.supported) setOpen(true);
  }, [status]);

  if (!open) return null;

  const busy = isServerInstallBusy(progress);
  const done = progress.phase === 'done';
  const failed = progress.phase === 'error';
  const started = busy || done || failed;

  const decline = () => {
    dismissPrompt();
    setOpen(false);
  };

  return (
    <div className="server-install-overlay">
      <div className="server-install-modal">
        <h3>Install Lowkey Media Server?</h3>
        {!started && (
          <p>
            The media server adds AI tagging, visual and text search, face
            grouping, transcription and background jobs to the viewer. It runs
            on this computer and starts with it. We can download and set it up
            for you now.
          </p>
        )}
        {started && (
          <p className={failed ? 'error' : ''}>
            {describeServerInstall(progress)}
          </p>
        )}
        {progress.phase === 'downloading' && progress.total ? (
          <div className="server-install-bar">
            <div
              style={{
                width: `${Math.min(
                  100,
                  ((progress.received || 0) / progress.total) * 100
                )}%`,
              }}
            />
          </div>
        ) : null}
        {done && (
          <p>
            Your browser has opened to finish setup (create your account and
            pick what to install). Then sign in from the database tab of the
            command palette.
          </p>
        )}
        {failed && (
          <p>You can try again later from the database tab of the command palette.</p>
        )}
        <div className="server-install-actions">
          {!started && (
            <>
              <button onClick={decline}>No thanks</button>
              <button className="primary" onClick={() => install()}>
                Install server
              </button>
            </>
          )}
          {busy && <button disabled>Installing…</button>}
          {(done || failed) && (
            <button className="primary" onClick={() => setOpen(false)}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
