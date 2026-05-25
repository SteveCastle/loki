import React, { useEffect, useState } from 'react';
import { isElectron } from '../platform';
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
      setLoaded(true);
      return;
    }
    let cancelled = false;
    fetch('/api/onboarding/state')
      .then((r) => (r.ok ? r.json() : { shown: true } as OnboardingState))
      .then((s: OnboardingState) => {
        if (cancelled) return;
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
