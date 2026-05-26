import React, { useState } from 'react';
import type { DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; }

export const OptionalPanel: React.FC<Props> = ({ items }) => {
  const optional = items.filter((i) => i.category === 'optional');
  return (
    <section className={styles.panel}>
      <header>
        <h2>Optional tools</h2>
        <p>Install these yourself if you want the features they unlock. The server runs fine without them.</p>
      </header>
      <ul className={styles.list}>
        {optional.map((i) => <OptionalRow key={i.id} item={i} />)}
      </ul>
    </section>
  );
};

const OptionalRow: React.FC<{ item: DepStatus }> = ({ item }) => {
  const hint = item.detail || {};
  const [open, setOpen] = useState(false);
  const cmds: Array<{ os: string; label: string; command: string }> = hint.commands || [];
  const desc: string = hint.description || '';
  const docsURL: string = hint.docs_url || '';
  return (
    <li className={styles[item.state] || styles.row}>
      <div className={styles.head}>
        <span className={styles.icon} aria-hidden>{item.state === 'installed' ? 'OK' : '-'}</span>
        <span className={styles.name}>{item.name}</span>
        {item.version && <span className={styles.version}>{item.version}</span>}
        <button type="button" className={styles.disclose} onClick={() => setOpen((o) => !o)}>
          {open ? 'Hide install commands' : 'Show install commands'}
        </button>
      </div>
      {open && (
        <div className={styles.detail}>
          {desc && <p>{desc}</p>}
          <ul className={styles.cmds}>
            {cmds.map((c) => (
              <li key={c.os + c.label}>
                <strong>{c.os} ({c.label}):</strong>
                <code>{c.command}</code>
              </li>
            ))}
          </ul>
          {docsURL && <a href={docsURL} target="_blank" rel="noreferrer">Documentation</a>}
        </div>
      )}
    </li>
  );
};
