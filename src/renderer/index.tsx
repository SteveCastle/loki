// FIRST import, and it must stay first: boot-clock timestamps the earliest
// moment our code runs, which is what makes the module-graph evaluation cost
// measurable. See boot-clock.ts.
import './boot-clock';
import { createRoot } from 'react-dom/client';
import invariant from 'tiny-invariant';
import { GlobalStateProvider } from './state';
import App from './App';
import { logEvent } from './platform';
import { beginStartupTrace, markStartup } from './startup-marks';

// Everything above this line has now been evaluated — React, XState, the
// platform layer, and the whole component tree.
beginStartupTrace();
markStartup('imports-evaluated');

// Capture renderer crashes/unhandled rejections to the app log file so a
// frozen or blank UI in the field leaves a diagnosable trail.
window.addEventListener('error', (e) => {
  logEvent({
    scope: 'window.onerror',
    message: e.message || 'window error',
    data: { filename: e.filename, lineno: e.lineno, colno: e.colno },
    error: e.error,
  });
});
window.addEventListener('unhandledrejection', (e) => {
  logEvent({
    scope: 'unhandledrejection',
    message: 'Unhandled promise rejection in renderer',
    error: (e as PromiseRejectionEvent).reason,
  });
});

const container = document.getElementById('root');
invariant(container, 'root element not found');
const root = createRoot(container);
markStartup('react-render');
root.render(
  <GlobalStateProvider>
    <App />
  </GlobalStateProvider>
);
