import React, { useState } from 'react';
import { BundledPanel } from './BundledPanel';
import { ModelsPanel } from './ModelsPanel';
import { OptionalPanel } from './OptionalPanel';
import { useDepsStatus } from './useDepsStatus';
import styles from './styles.module.css';

interface Props {
  onDismiss?: () => void;
  embedded?: boolean;
}

export const OnboardingWizard: React.FC<Props> = ({ onDismiss, embedded }) => {
  const { status, refresh, error } = useDepsStatus();
  const [step, setStep] = useState(0);
  const steps = [
    { title: 'Welcome', render: () => <BundledPanel items={status} /> },
    { title: 'Optional tools', render: () => <OptionalPanel items={status} /> },
    { title: 'AI models', render: () => <ModelsPanel items={status} onChange={refresh} /> },
  ];
  const last = step === steps.length - 1;

  if (embedded) {
    return (
      <div>
        {error && <div className={styles.error}>Status error: {error}</div>}
        <BundledPanel items={status} />
        <OptionalPanel items={status} />
        <ModelsPanel items={status} onChange={refresh} />
      </div>
    );
  }

  return (
    <div className={styles.wizardOverlay}>
      <div className={styles.wizardModal}>
        <header>
          <h1>Welcome to Lowkey Media Server</h1>
          <p>Step {step + 1} of {steps.length}: {steps[step].title}</p>
        </header>
        {error && <div className={styles.error}>Status error: {error}</div>}
        {steps[step].render()}
        <div className={styles.actions} style={{ marginTop: '1.5rem' }}>
          <button type="button" className={styles.btn} onClick={onDismiss}>Skip — I'll do this later</button>
          {step > 0 && <button type="button" className={styles.btn} onClick={() => setStep(step - 1)}>Back</button>}
          {!last && <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={() => setStep(step + 1)}>Next</button>}
          {last && <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={onDismiss}>Finish</button>}
        </div>
      </div>
    </div>
  );
};
