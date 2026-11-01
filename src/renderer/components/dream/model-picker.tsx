// The image model picker: one of the diffusion models the server's loki-dream can run (dream-models.ts), or the
// server's default. Built for long lists: a closed button that names the choice, opening inline into a search box,
// architecture filters with counts, and a keyboard-navigable list grouped by architecture, each model with its variant
// and whether it is ready or what a first run downloads.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { downloadGb, effectiveDreamModel, sameModelFile, variantLabel, type DreamCatalog, type DreamModel } from './dream-models';
import './model-picker.css';

export interface ModelPickerProps {
  catalog: DreamCatalog;
  /** The chosen file; '' = the server's default. */
  value: string;
  /** model = the entry picked (undefined for the default). */
  onChange: (file: string, model: DreamModel | undefined) => void;
  onRefresh: () => void;
  /** A label for the closed button (screen readers). */
  label?: string;
}

const ARCH_ORDER = ['moss', 'krea2', 'flux2', 'anima', 'sdxl'];

function archRank(a: string): number {
  const i = ARCH_ORDER.indexOf(a);
  return i < 0 ? ARCH_ORDER.length : i;
}

/** Every term of the query appears in the model's label, file, architecture, base model or format. */
export function matchesQuery(m: DreamModel, query: string): boolean {
  const hay = [m.label, m.file, m.archName, m.arch, m.baseModel, m.format, m.distilled ? 'distilled turbo' : 'base'].join(' ').toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((t) => hay.includes(t));
}

function Status({ m }: { m: DreamModel }) {
  const gb = downloadGb(m);
  if (!gb) return <span className="mp-status ready">Ready</span>;
  const what = !m.local ? 'model' : m.textEncoder && !m.textEncoder.present ? 'text encoder' : 'VAE';
  return (
    <span className="mp-status dl" title={`The first run downloads ${gb} GB (${what}${!m.local && downloadGb({ ...m, local: true }) ? ' and more' : ''})`}>
      ↓ {gb} GB
    </span>
  );
}

function ArchBadge({ m }: { m: DreamModel }) {
  return <span className={`mp-arch mp-arch-${m.arch}`}>{m.archName}</span>;
}

type Row =
  | { kind: 'default'; model: DreamModel | undefined }
  | { kind: 'head'; arch: string; name: string; count: number }
  | { kind: 'model'; model: DreamModel };

export default function ModelPicker({ catalog, value, onChange, onRefresh, label = 'Model' }: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [arch, setArch] = useState<string>('');
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const chosen = value ? catalog.models.find((m) => sameModelFile(m.file, value)) : undefined;
  const fallback = effectiveDreamModel(catalog, '');

  // architectures present, with counts, in a stable order
  const archs = useMemo(() => {
    const by = new Map<string, { name: string; count: number }>();
    catalog.models.forEach((m) => {
      const e = by.get(m.arch);
      if (e) e.count += 1;
      else by.set(m.arch, { name: m.archName, count: 1 });
    });
    return Array.from(by.entries())
      .sort((a, b) => archRank(a[0]) - archRank(b[0]) || a[1].name.localeCompare(b[1].name))
      .map(([id, e]) => ({ id, ...e }));
  }, [catalog.models]);

  const rows: Row[] = useMemo(() => {
    const shown = catalog.models.filter((m) => (!arch || m.arch === arch) && matchesQuery(m, query));
    const out: Row[] = [];
    if (!query && !arch) out.push({ kind: 'default', model: fallback });
    const groups = new Map<string, DreamModel[]>();
    shown.forEach((m) => groups.set(m.arch, [...(groups.get(m.arch) || []), m]));
    Array.from(groups.entries())
      .sort((a, b) => archRank(a[0]) - archRank(b[0]))
      .forEach(([a, ms]) => {
        out.push({
          kind: 'head',
          arch: a,
          name: ms[0].archName,
          count: ms.length,
        });
        // ready models first, then by name
        ms.sort((x, y) => Number(downloadGb(x) > 0) - Number(downloadGb(y) > 0) || x.label.localeCompare(y.label, undefined, { numeric: true }));
        ms.forEach((m) => out.push({ kind: 'model', model: m }));
      });
    return out;
  }, [catalog.models, arch, query, fallback]);

  const selectable = rows.map((r, i) => (r.kind === 'head' ? -1 : i)).filter((i) => i >= 0);

  // open: start on the chosen row
  useEffect(() => {
    if (!open) return;
    const at = rows.findIndex((r) => (r.kind === 'default' ? !value : r.kind === 'model' && sameModelFile(r.model.file, value)));
    setActive(at >= 0 ? at : selectable[0] ?? 0);
    window.setTimeout(() => searchRef.current?.focus(), 0);
    // only when it opens
  }, [open]);

  // keep the active row valid as the filter changes
  useEffect(() => {
    if (open && !selectable.includes(active)) setActive(selectable[0] ?? 0);
  }, [rows]);

  useEffect(() => {
    if (!open) return undefined;
    const close = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  useEffect(() => {
    listRef.current?.querySelector(`[data-row="${active}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  const pick = (r: Row | undefined) => {
    if (!r || r.kind === 'head') return;
    if (r.kind === 'default') onChange('', undefined);
    else onChange(r.model.file, r.model);
    setOpen(false);
    setQuery('');
  };

  const onKey = (e: React.KeyboardEvent) => {
    const pos = selectable.indexOf(active);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive(selectable[Math.min(selectable.length - 1, pos + 1)] ?? active);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive(selectable[Math.max(0, pos - 1)] ?? active);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      pick(rows[active]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
    }
  };

  const shownModel = chosen || (!value ? fallback : undefined);
  const count = catalog.models.length;

  return (
    <div className={`mp-root${open ? ' open' : ''}`} ref={rootRef}>
      <button type="button" className="mp-button" aria-haspopup="listbox" aria-expanded={open} aria-label={label} onClick={() => setOpen((o) => !o)}>
        {shownModel ? <ArchBadge m={shownModel} /> : null}
        <span className="mp-button-name">
          {value ? (chosen ? chosen.label : value.replace(/\.safetensors$/i, '')) : `Default${fallback ? ` · ${fallback.label}` : ''}`}
        </span>
        <span className="mp-button-sub">
          {shownModel ? variantLabel(shownModel) : value ? 'not listed by the server' : catalog.loading ? 'loading models…' : ''}
        </span>
        {shownModel ? <Status m={shownModel} /> : null}
        <span className="mp-caret" aria-hidden>
          ▾
        </span>
      </button>

      {open && (
        <div className="mp-panel" onKeyDown={onKey}>
          <div className="mp-search-row">
            <input
              ref={searchRef}
              className="mp-search"
              value={query}
              placeholder={count ? `Search ${count} model${count === 1 ? '' : 's'}…` : 'Search models…'}
              aria-label="Search models"
              aria-controls="mp-list"
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="button" className="mp-refresh" title="Rescan the server's model folders" disabled={catalog.loading} onClick={onRefresh}>
              {catalog.loading ? '…' : '↻'}
            </button>
          </div>
          {archs.length > 1 && (
            <div className="mp-filters" role="tablist" aria-label="Architecture">
              <button type="button" role="tab" aria-selected={!arch} className={`mp-filter${!arch ? ' active' : ''}`} onClick={() => setArch('')}>
                All <small>{count}</small>
              </button>
              {archs.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  role="tab"
                  aria-selected={arch === a.id}
                  className={`mp-filter${arch === a.id ? ' active' : ''}`}
                  onClick={() => setArch(arch === a.id ? '' : a.id)}
                >
                  {a.name} <small>{a.count}</small>
                </button>
              ))}
            </div>
          )}
          <div className="mp-list" id="mp-list" role="listbox" aria-label="Models" ref={listRef}>
            {rows.map((r, i) => {
              if (r.kind === 'head') {
                return (
                  <div key={`h-${r.arch}`} className="mp-group">
                    {r.name} <small>{r.count}</small>
                  </div>
                );
              }
              const selected = r.kind === 'default' ? !value : sameModelFile(r.model.file, value);
              const m = r.kind === 'default' ? r.model : r.model;
              return (
                <div
                  key={r.kind === 'default' ? 'default' : r.model.file}
                  data-row={i}
                  role="option"
                  aria-selected={selected}
                  className={`mp-row${i === active ? ' active' : ''}${selected ? ' selected' : ''}`}
                  onMouseEnter={() => setActive(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pick(r)}
                >
                  <span className="mp-check" aria-hidden>
                    {selected ? '✓' : ''}
                  </span>
                  <span className="mp-row-main">
                    <span className="mp-row-name">{r.kind === 'default' ? `Server default${m ? ` · ${m.label}` : ''}` : r.model.label}</span>
                    <span className="mp-row-file">
                      {r.kind === 'default'
                        ? 'follows the server setting (dreamModel)'
                        : [r.model.file.replace(/\.safetensors$/i, ''), r.model.format, r.model.sizeGb ? `${r.model.sizeGb} GB` : '', r.model.baseModel]
                            .filter(Boolean)
                            .join(' · ')}
                    </span>
                  </span>
                  {m ? <span className="mp-row-variant">{variantLabel(m)}</span> : null}
                  {m ? <Status m={m} /> : null}
                </div>
              );
            })}
            {rows.filter((r) => r.kind === 'model').length === 0 && (
              <div className="mp-empty">
                {catalog.loading
                  ? 'Loading the server’s models…'
                  : query || arch
                  ? 'No model matches.'
                  : catalog.note || catalog.error || 'The server lists no models.'}
              </div>
            )}
          </div>
          {(catalog.note || catalog.error || catalog.where) && (
            <div className={`mp-foot${catalog.error ? ' warn' : ''}`}>{catalog.error || catalog.note || `Models found by loki-dream on ${catalog.where}.`}</div>
          )}
        </div>
      )}
    </div>
  );
}
