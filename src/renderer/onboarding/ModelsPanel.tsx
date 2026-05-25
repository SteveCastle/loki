import React from 'react';
import { cancelModelDownload, deleteModel, startModelDownload, type DepStatus } from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; onChange: () => void; }

export const ModelsPanel: React.FC<Props> = ({ items, onChange }) => {
  const models = items.filter((i) => i.category === 'model');
  return (
    <section className={styles.panel}>
      <header>
        <h2>AI models</h2>
        <p>Download what you want; you can come back anytime.</p>
      </header>
      <ul className={styles.list}>
        {models.map((m) => <ModelRow key={m.id} item={m} onChange={onChange} />)}
      </ul>
    </section>
  );
};

const fmtSize = (n?: number): string => {
  if (!n) return '';
  const mb = n / 1024 / 1024;
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
};

const ModelRow: React.FC<{ item: DepStatus; onChange: () => void }> = ({ item, onChange }) => {
  const inst = item.detail || {};
  const done: number = inst.bytes_done ?? 0;
  const total: number = inst.bytes_total ?? item.size_bytes ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;

  const onDownload = async () => { await startModelDownload(item.id); onChange(); };
  const onCancel   = async () => { await cancelModelDownload(item.id); onChange(); };
  const onDelete   = async () => { await deleteModel(item.id); onChange(); };

  return (
    <li className={styles[item.state] || styles.row}>
      <div className={styles.head}>
        <span className={styles.icon} aria-hidden>{stateIcon(item.state)}</span>
        <span className={styles.name}>{item.name} <span className={styles.version}>{fmtSize(item.size_bytes)}</span></span>
        <div className={styles.actions}>
          {item.state === 'installed' && (
            <button type="button" className={`${styles.btn} ${styles.btnDanger}`} onClick={onDelete}>Delete</button>
          )}
          {(item.state === 'missing' || item.state === 'failed' || item.state === 'cancelled') && (
            <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={onDownload}>Download</button>
          )}
          {(item.state === 'downloading' || item.state === 'queued' || item.state === 'verifying') && (
            <button type="button" className={styles.btn} onClick={onCancel}>Cancel</button>
          )}
        </div>
      </div>
      {(item.state === 'downloading' || item.state === 'queued' || item.state === 'verifying') && total > 0 && (
        <div className={styles.detail}>
          <div className={styles.progressBar}><div className={styles.progressBarFill} style={{ width: `${pct}%` }} /></div>
          <small>{fmtSize(done)} / {fmtSize(total)} ({pct}%) {inst.current_file && `- ${inst.current_file}`}</small>
        </div>
      )}
      {item.error && <div className={styles.error}>{item.error}</div>}
    </li>
  );
};

function stateIcon(s: string): string {
  switch (s) {
    case 'installed': return 'OK';
    case 'downloading':
    case 'queued':
    case 'verifying': return '...';
    case 'failed': return 'X';
    default: return '-';
  }
}
