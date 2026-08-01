// Gate for the ✨ "search by meaning" (vector search) toggle in QueryInput.
// Text→image search runs on the media-server embedding backend, so in
// Electron it requires a reachable server AND a logged-in session (Bearer
// token) — without both, the toggle must not be offered at all. In web mode
// the SPA is served by that same server and search (including inference) is
// open to read-only visitors, so it is always available.
//
// Also enforces the gate against the sticky shared meaning mode: if the
// server drops or the user logs out while meaning mode is on, the mode is
// forced off so Enter can't silently commit visual: predicates that would
// fail. Parents hide the toggle by passing onSubmitVisual only when this
// returns true (QueryInput renders the button off that prop's presence).

import { useEffect } from 'react';
import { isElectron } from '../platform';
import useJobServerAvailable from './useJobServerAvailable';
import { useMeaningMode } from './useMeaningMode';

export default function useVisualSearchAvailable(
  authToken?: string | null
): boolean {
  // Shared module-level probe — cheap to call from multiple surfaces. In web
  // mode the live SSE bus answers without spending a socket.
  const serverAvailable = useJobServerAvailable(authToken);
  const available =
    !isElectron || (serverAvailable === true && Boolean(authToken));

  const { meaningMode, setMeaningMode } = useMeaningMode();
  useEffect(() => {
    if (!available && meaningMode) setMeaningMode(false);
  }, [available, meaningMode, setMeaningMode]);

  return available;
}
