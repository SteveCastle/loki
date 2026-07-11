import React, { useEffect, useState } from 'react';
import { isElectron } from '../platform';
import { initAccess } from '../access';
import { OnboardingWizard } from './OnboardingWizard';

interface OnboardingState {
  shown: boolean;
}

// OnboardingGate renders the welcome wizard once on first run in web mode.
// In Electron, it never renders (the desktop app has its own setup flow).
export const OnboardingGate: React.FC = () => {
  const [visible, setVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (isElectron) {
      // Electron has its own setup flow.
      setLoaded(true);
      return;
    }
    let cancelled = false;
    // Await the access state (don't read the cache synchronously — this can
    // mount before the fetch resolves): view-only public visitors never see
    // the wizard (its model-download actions are admin-gated anyway).
    initAccess()
      .then((access) => {
        if (cancelled) return null;
        if (!access.canWrite) {
          setLoaded(true);
          return null;
        }
        return fetch('/api/onboarding/state');
      })
      .then((r) => {
        if (cancelled || r === null) return null;
        return r.ok ? r.json() : ({ shown: true } as OnboardingState);
      })
      .then((s: OnboardingState | null) => {
        if (cancelled || s === null) return;
        setVisible(!s.shown);
        setLoaded(true);
      })
      .catch(() => {
        if (cancelled) return;
        setLoaded(true);
      });
    return () => { cancelled = true; };
  }, []);

  const onDismiss = async () => {
    setVisible(false);
    try {
      await fetch('/api/onboarding/dismiss', { method: 'POST' });
    } catch {
      /* best-effort */
    }
  };

  if (!loaded || !visible) return null;
  return <OnboardingWizard onDismiss={onDismiss} />;
};
