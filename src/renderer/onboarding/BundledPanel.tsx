import React from 'react';
import type { DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; }

export const BundledPanel: React.FC<Props> = ({ items }) => {
  const bundled = items.filter((i) => i.category === 'bundled');
  const allReady = bundled.every((i) => i.state === 'ready');
  return (
    <section className={styles.panel}>
      <header>
        <h2>Bundled tools</h2>
        <p>These ship with the server and need no setup.</p>
      </header>
      <ul className={styles.list}>
        {bundled.map((i) => (
          <li key={i.id} className={styles[i.state] || styles.row}>
            <span className={styles.icon} aria-hidden>{i.state === 'ready' ? 'OK' : i.state === 'missing' ? 'X' : '!'}</span>
            <span className={styles.name}>{i.name}</span>
            {i.version && <span className={styles.version}>{i.version}</span>}
            {i.state !== 'ready' && i.error && <span className={styles.error}>{i.error}</span>}
          </li>
        ))}
      </ul>
      {!allReady && (
        <p className={styles.warn}>
          Something is wrong with the server install. Please reinstall the server.
        </p>
      )}
    </section>
  );
};
