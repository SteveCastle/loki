import React from 'react';
import {
  cancelModelDownload,
  deleteModel,
  isDownloadableState,
  isDownloadingState,
  startModelDownload,
  type DepStatus,
} from './api';
import styles from './styles.module.css';

interface Props { items: DepStatus[]; onChange: () => void; }

export const ModelsPanel: React.FC<Props> = ({ items, onChange }) => {
  const models = items.filter((i) => i.category === 'model' || i.category === 'tool');
  return (
    <section className={styles.panel}>
      <header>
        <h2>AI features</h2>
        <p>Each download unlocks a feature. Pick what you want — you can come back anytime.</p>
      </header>
      <ul className={styles.list}>
        {models.map((m) => <ModelRow key={m.id} item={m} onChange={onChange} />)}
      </ul>
    </section>
  );
};

const fmtSize = (n?: number): string => {
  if (!n) return '';
  const kb = n / 1024;
  // Small models (face detectors are ~230 KB) must not round down to "0 MB".
  if (kb < 1024) return `${Math.max(1, Math.round(kb))} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
};

const ModelRow: React.FC<{ item: DepStatus; onChange: () => void }> = ({ item, onChange }) => {
  const inst = item.detail || {};
  const done: number = inst.bytes_done ?? 0;
  const total: number = inst.bytes_total ?? item.size_bytes ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
  // Archive unpack phase: byte counts are UNCOMPRESSED output (often much
  // larger than the download itself) — label them as such.
  const extracting = item.state === 'extracting';

  const onDownload = async () => { await startModelDownload(item.id); onChange(); };
  const onCancel   = async () => { await cancelModelDownload(item.id); onChange(); };
  const onDelete   = async () => { await deleteModel(item.id); onChange(); };

  // Lead with the feature the download unlocks; the model name is detail.
  const primary = item.feature || item.name;
  const secondary = item.feature ? item.name : '';
  // A user-supplied binary (configured path) can satisfy a tool without our download.
  const viaConfiguredPath = inst.source === 'configured_path';

  return (
    <li className={styles[item.state] || styles.row}>
      <div className={styles.head}>
        <span className={styles.icon} aria-hidden>{stateIcon(item.state)}</span>
        <span className={styles.name}>
          {primary} <span className={styles.version}>{secondary && `${secondary} · `}{fmtSize(item.size_bytes)}</span>
        </span>
        <div className={styles.actions}>
          {item.state === 'installed' && !viaConfiguredPath && (
            <button type="button" className={`${styles.btn} ${styles.btnDanger}`} onClick={onDelete}>Delete</button>
          )}
          {isDownloadableState(item.state) && (
            <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={onDownload}>Download</button>
          )}
          {isDownloadingState(item.state) && (
            <button type="button" className={styles.btn} onClick={onCancel}>Cancel</button>
          )}
        </div>
      </div>
      {item.description && item.state !== 'installed' && (
        <div className={styles.detail}><small>{item.description}</small></div>
      )}
      {viaConfiguredPath && (
        <div className={styles.detail}><small>Using the binary configured in settings: {item.path}</small></div>
      )}
      {item.state === 'installed' && !viaConfiguredPath && item.path && (
        <div className={styles.detail}><small>Installed at: {item.path}</small></div>
      )}
      {isDownloadingState(item.state) && total > 0 && (
        <div className={styles.detail}>
          <div className={styles.progressBar}><div className={styles.progressBarFill} style={{ width: `${pct}%` }} /></div>
          <small>
            {extracting
              ? `Unpacking (one-time): ${fmtSize(done)} / ${fmtSize(total)} on disk (${pct}%)`
              : `${fmtSize(done)} / ${fmtSize(total)} (${pct}%)`}
            {!extracting && inst.current_file && ` - ${inst.current_file}`}
          </small>
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
    case 'extracting':
    case 'queued':
    case 'verifying': return '...';
    case 'failed': return 'X';
    default: return '-';
  }
}
