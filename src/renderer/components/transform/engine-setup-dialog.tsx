import { useEffect } from 'react';
import { fmtSize } from '../../onboarding/requirements';
import {
  beginEngineDownload,
  cancelEngineSetup,
  continueEngineSetupInBackground,
  useEngineSetup,
} from './engine-setup';
import './transform-flow.css';

// The "one-time download" confirmation shown before the first AI job on a
// machine. Mounted once at the app root next to the Transform flow.
export default function EngineSetupDialog() {
  const s = useEngineSetup();

  useEffect(() => {
    if (!s) return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        if (s.phase === 'downloading') continueEngineSetupInBackground();
        else cancelEngineSetup();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [s]);

  if (!s) return null;
  const pct = s.bytesTotal > 0 ? Math.min(100, Math.round((s.bytesDone / s.bytesTotal) * 100)) : 0;

  return (
    <div className="ts-backdrop ts-setup-backdrop">
      <div
        className="ts-panel ts-setup"
        role="alertdialog"
        aria-modal="true"
        aria-label="One-time download"
        onKeyDown={(e) => e.stopPropagation()}
        onKeyUp={(e) => e.stopPropagation()}
      >
        <header className="ts-header">
          <div className="ts-title">
            <span className="ts-title-main">One-time download</span>
            <span className="ts-title-sub">{s.need.title}</span>
          </div>
        </header>

        <div className="ts-body ts-setup-body">
          {s.phase === 'confirm' && (
            <>
              <p className="ts-lead">
                {s.need.title} runs entirely on this computer. It needs {fmtSize(s.need.bytes)} of files, downloaded
                once and kept for next time.
              </p>
              <ul className="ts-setup-list">
                {s.need.items.map((d) => (
                  <li key={d.id}>
                    <span>{d.name}</span>
                    <span className="ts-setup-size">{fmtSize(d.size_bytes)}</span>
                  </li>
                ))}
              </ul>
              <p className="ts-note">
                Needs an NVIDIA RTX 40-series (or newer) GPU with 24 GB of video memory. The download can be stopped
                and resumed; your job is queued as soon as it finishes.
              </p>
            </>
          )}
          {s.phase === 'downloading' && (
            <>
              <p className="ts-lead">
                Downloading {fmtSize(s.bytesDone) || '0 KB'} of {fmtSize(s.bytesTotal)} ({pct}%)…
              </p>
              <div className="ts-setup-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
                <div className="ts-setup-bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="ts-note">
                Your job is queued automatically when this finishes. You can keep working: the download continues in
                the background.
              </p>
            </>
          )}
          {s.phase === 'failed' && (
            <>
              <p className="ts-lead">The download did not finish.</p>
              <p className="ts-error">{s.error}</p>
              <p className="ts-note">What was already downloaded is kept, so retrying picks up where it stopped.</p>
            </>
          )}
        </div>

        <div className="ts-footer">
          {s.phase === 'confirm' && (
            <>
              <button type="button" className="ts-btn ghost" onClick={cancelEngineSetup}>
                Not now
              </button>
              <button type="button" className="ts-btn primary" onClick={beginEngineDownload} autoFocus>
                Download {fmtSize(s.need.bytes)} and continue
              </button>
            </>
          )}
          {s.phase === 'downloading' && (
            <>
              <button type="button" className="ts-btn ghost" onClick={cancelEngineSetup}>
                Cancel download
              </button>
              <button type="button" className="ts-btn primary" onClick={continueEngineSetupInBackground} autoFocus>
                Continue in background
              </button>
            </>
          )}
          {s.phase === 'failed' && (
            <>
              <button type="button" className="ts-btn ghost" onClick={cancelEngineSetup}>
                Close
              </button>
              <button type="button" className="ts-btn primary" onClick={beginEngineDownload} autoFocus>
                Retry
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
