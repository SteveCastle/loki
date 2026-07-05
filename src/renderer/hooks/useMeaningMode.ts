import { useState, useEffect, useCallback } from 'react';

// "Search by meaning" (✨ vector search) toggle. Module-level shared state so
// the toggle is sticky: it survives the command palette closing/reopening and
// stays wherever the user set it until they flip it back themselves. A
// listener registry keeps every mounted surface (palette + taxonomy sidebar)
// in sync — same pattern as useSearchHistory.
let meaningModeOn = false;
const listeners = new Set<(on: boolean) => void>();

export function useMeaningMode() {
  const [meaningMode, setLocal] = useState(meaningModeOn);

  useEffect(() => {
    listeners.add(setLocal);
    // Pull the current shared value into this instance (it may have been
    // toggled while this surface was unmounted).
    setLocal(meaningModeOn);
    return () => {
      listeners.delete(setLocal);
    };
  }, []);

  const setMeaningMode = useCallback((on: boolean) => {
    meaningModeOn = on;
    listeners.forEach((listener) => listener(on));
  }, []);

  return { meaningMode, setMeaningMode };
}
