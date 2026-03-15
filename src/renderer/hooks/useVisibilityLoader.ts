import { useState, useEffect, useRef } from 'react';

/**
 * Hook that delays triggering a load until the component has been
 * mounted and visible for a minimum duration. This prevents loading
 * resources for items that are quickly scrolled past.
 *
 * @param delayMs - How long the component must be mounted before allowing load (0 = immediate)
 * @returns shouldLoad - Whether the component should start loading its content
 */
export function useVisibilityLoader(delayMs = 0): boolean {
  // If no delay, return true immediately without any timer overhead
  const [shouldLoad, setShouldLoad] = useState(delayMs === 0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (delayMs === 0) {
      setShouldLoad(true);
      return;
    }

    timerRef.current = setTimeout(() => {
      setShouldLoad(true);
    }, delayMs);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [delayMs]);

  return shouldLoad;
}
