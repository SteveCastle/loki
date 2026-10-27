import React, { useCallback, useContext, useEffect, useRef, useState } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { isElectron } from '../../platform';
import {
  useServerInstall,
  isServerInstallBusy,
  describeServerInstall,
  type ServerInstallProgress,
} from '../../hooks/useServerInstall';
import './server-install.css';

type SetupPhase = 'idle' | 'running' | 'done' | 'error';

/**
 * Runs the server's setup wizard + sign-in in the viewer's own window and
 * signs the viewer in with the key it hands back.
 */
function useServerSetup(runSetup: () => Promise<{ token?: string; error?: string }>) {
  const { libraryService } = useContext(GlobalStateContext);
  const [phase, setPhase] = useState<SetupPhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const run = useCallback(async () => {
    setPhase('running');
    setError(null);
    try {
      const res = await runSetup();
      if (!res?.token) throw new Error(res?.error || 'Setup did not finish');
      libraryService.send({ type: 'SET_AUTH_TOKEN', token: res.token });
      libraryService.send({ type: 'SET_CAN_WRITE', canWrite: true });
      setPhase('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase('error');
    }
  }, [runSetup, libraryService]);
  return { phase, error, run };
}

function ProgressBar({ progress }: { progress: ServerInstallProgress }) {
  if (progress.phase !== 'downloading' || !progress.total) return null;
  return (
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
  );
}

// Palette (server tab) section: what state the local media server is in,
// the one action that moves it forward (install / start / upgrade / finish
// setup), and the "start with the viewer" setting.
export function ServerInstallWidget() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const { status, progress, install, start, setAutoStart, runSetup } =
    useServerInstall();
  const setup = useServerSetup(runSetup);
  if (!isElectron || !status) return null;
  if (!status.supported && !status.installedPath) return null;

  const busy = isServerInstallBusy(progress);
  const installed = !!status.installedPath;

  let label: string;
  if (status.running) {
    label = `Lowkey Media Server is running${
      status.serverVersion && status.serverVersion !== 'dev'
        ? ` (v${status.serverVersion})`
        : ''
    }.`;
  } else if (installed) {
    label = 'Lowkey Media Server is installed but not running.';
  } else {
    label = 'Lowkey Media Server is not installed.';
  }

  let action: React.ReactNode = null;
  if (!installed) {
    action = (
      <button disabled={busy} onClick={() => install()}>
        Install media server
      </button>
    );
  } else if (status.updateAvailable) {
    action = (
      <button disabled={busy} onClick={() => install()}>
        Upgrade server to v{status.appVersion}
      </button>
    );
  } else if (!status.running) {
    action = (
      <button disabled={busy} onClick={() => start()}>
        Start server
      </button>
    );
  } else if (!authToken) {
    action = (
      <button disabled={setup.phase === 'running'} onClick={setup.run}>
        {setup.phase === 'running'
          ? 'Finish in the setup window…'
          : 'Finish setup & sign in'}
      </button>
    );
  }

  return (
    <div className="ServerInstallWidget">
      <div className="server-install-label">{label}</div>
      {(busy || progress.phase === 'error') && (
        <div
          className={`server-install-progress ${
            progress.phase === 'error' ? 'error' : ''
          }`}
        >
          {describeServerInstall(progress)}
        </div>
      )}
      {setup.phase === 'error' && setup.error && (
        <div className="server-install-progress error">{setup.error}</div>
      )}
      {action}
      {installed && (
        <label className="server-install-toggle">
          <input
            type="checkbox"
            checked={status.autoStart}
            onChange={(e) => setAutoStart(e.target.checked)}
          />
          Start the media server with the viewer
        </label>
      )}
    </div>
  );
}

/**
 * The install offer itself, shared by the welcome screen (first launch) and
 * the one-time modal. Install → download → start → the server's setup wizard
 * and sign-in in a viewer window, then done.
 */
export function ServerOffer({
  onClose,
  showClose = true,
}: {
  onClose?: () => void;
  showClose?: boolean;
}) {
  const { progress, install, dismissPrompt, runSetup } = useServerInstall();
  const setup = useServerSetup(runSetup);
  const [declined, setDeclined] = useState(false);

  // Setup follows the install on its own: the user already said yes.
  const setupStarted = useRef(false);
  useEffect(() => {
    if (progress.phase === 'done' && !setupStarted.current) {
      setupStarted.current = true;
      setup.run();
    }
  }, [progress.phase, setup]);

  const busy = isServerInstallBusy(progress);
  const installFailed = progress.phase === 'error';
  const installed = progress.phase === 'done';
  const started = busy || installed || installFailed;

  if (declined) {
    return (
      <div className="server-offer">
        <p>
          No problem. You can add the media server any time from the database
          tab of the command palette.
        </p>
      </div>
    );
  }

  return (
    <div className="server-offer">
      <h3>Add Lowkey Media Server?</h3>
      {!started && (
        <p>
          The media server adds AI tagging, visual and text search, face
          grouping, transcription and background jobs. It runs privately on
          this computer and starts with the viewer. AI models download only
          when you first use a feature.
        </p>
      )}
      {busy && <p>{describeServerInstall(progress)}</p>}
      {installFailed && (
        <p className="error">{describeServerInstall(progress)}</p>
      )}
      <ProgressBar progress={progress} />
      {installed && setup.phase === 'running' && (
        <p>
          The server is running. Finish setting it up in the window that just
          opened: create your account and pick a library location.
        </p>
      )}
      {installed && setup.phase === 'done' && (
        <p>All set: the server is running and the viewer is signed in.</p>
      )}
      {installed && setup.phase === 'error' && (
        <p>
          Setup isn&apos;t finished yet{setup.error ? ` (${setup.error})` : ''}.
          You can pick it up again from the database tab of the command palette.
        </p>
      )}
      {installFailed && (
        <p>You can try again later from the database tab of the command palette.</p>
      )}
      <div className="server-install-actions">
        {!started && (
          <>
            <button
              onClick={() => {
                dismissPrompt();
                setDeclined(true);
                onClose?.();
              }}
            >
              Not now
            </button>
            <button className="primary" onClick={() => install()}>
              Install server
            </button>
          </>
        )}
        {busy && <button disabled>Installing…</button>}
        {installed && setup.phase === 'error' && (
          <button onClick={setup.run}>Open setup again</button>
        )}
        {showClose && (installFailed || (installed && setup.phase !== 'running')) && (
          <button className="primary" onClick={onClose}>
            Close
          </button>
        )}
      </div>
    </div>
  );
}

// How long after mount to look for a server. The check is a localhost probe
// plus a few fs.existsSync calls in main — cheap, but nothing about it should
// sit on the launch → first-media path.
const PROMPT_DELAY_MS = 2500;

// The one-time offer as a modal, for viewers past the welcome screen that
// were never asked (or that left the welcome screen mid-install — the
// install carries on in main, so pick its progress back up here).
export function ServerInstallPrompt() {
  const [armed, setArmed] = useState(false);
  const [open, setOpen] = useState(false);
  const { status } = useServerInstall(armed);

  useEffect(() => {
    if (!isElectron) return undefined;
    const t = setTimeout(() => setArmed(true), PROMPT_DELAY_MS);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!status) return;
    if (
      (!status.prompted && status.supported) ||
      isServerInstallBusy(status.progress)
    ) {
      setOpen(true);
    }
  }, [status]);

  if (!open) return null;
  return (
    <div className="server-install-overlay">
      <div className="server-install-modal">
        <ServerOffer onClose={() => setOpen(false)} />
      </div>
    </div>
  );
}
