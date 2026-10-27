import React, { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type { LayerKind } from './scene-api';

// The user's saved prompts, per layer kind (an edit's prompt says what the area
// becomes, a motion's how it moves: neither reads right as the other). Kept in
// this browser's storage like the remembered layer defaults; every Scene window
// shares it (storage events keep open windows in step).

const KEY = 'loki.scene.prompts.v1';

export interface SavedPrompt {
  id: string;
  name: string;
  prompt: string;
  saved: number;
  /** When it was last put in the prompt box: the library lists recent ones first. */
  used?: number;
}

type Library = Partial<Record<LayerKind, SavedPrompt[]>>;

const NONE: SavedPrompt[] = [];
let cache: Library | null = null;
const listeners = new Set<() => void>();

function read(): Library {
  if (cache) return cache;
  try {
    cache = JSON.parse(window.localStorage.getItem(KEY) || '{}') as Library;
  } catch {
    cache = {};
  }
  return cache;
}

function write(next: Library): void {
  cache = next;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // storage unavailable: the library lasts for this window only
  }
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  const onStorage = (e: StorageEvent) => {
    if (e.key !== KEY) return;
    cache = null;
    cb();
  };
  window.addEventListener('storage', onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener('storage', onStorage);
  };
}

export function usePromptLibrary(kind: LayerKind): SavedPrompt[] {
  return useSyncExternalStore(subscribe, () => read()[kind] || NONE, () => NONE);
}

/** Saves `prompt` under `name`; one with the same name (any case) or the same text is replaced. */
export function savePrompt(kind: LayerKind, name: string, prompt: string): SavedPrompt | null {
  const n = name.trim();
  const p = prompt.trim();
  if (!n || !p) return null;
  const entry: SavedPrompt = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, name: n, prompt: p, saved: Date.now() };
  const rest = (read()[kind] || NONE).filter((s) => s.name.toLowerCase() !== n.toLowerCase() && s.prompt !== p);
  write({ ...read(), [kind]: [...rest, entry] });
  return entry;
}

export function deletePrompt(kind: LayerKind, id: string): void {
  write({ ...read(), [kind]: (read()[kind] || NONE).filter((s) => s.id !== id) });
}

function markUsed(kind: LayerKind, id: string): void {
  write({ ...read(), [kind]: (read()[kind] || NONE).map((s) => (s.id === id ? { ...s, used: Date.now() } : s)) });
}

/** Tests: forget the in-memory copy so the next read goes back to localStorage. */
export function resetPromptLibraryCache(): void {
  cache = null;
}

/** A name for a new saved prompt: its first few words. */
export function suggestName(prompt: string): string {
  const flat = prompt.replace(/\s+/g, ' ').trim();
  if (flat.length <= 32) return flat;
  const cut = flat.slice(0, 32);
  const sp = cut.lastIndexOf(' ');
  return `${(sp > 12 ? cut.slice(0, sp) : cut).replace(/[.,;:]+$/, '')}…`;
}

/** Saved prompts whose name or text holds every word of the query, most recently used (else saved) first. */
export function searchPrompts(list: SavedPrompt[], query: string): SavedPrompt[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return list
    .filter((s) => {
      const hay = `${s.name}\n${s.prompt}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    })
    .sort((a, b) => (b.used || b.saved) - (a.used || a.saved));
}

/**
 * The prompt box's header: its label, "☆ Save" (names the prompt in the box; "★ Saved" once it is) and "Library",
 * which opens a searchable list of the saved prompts for this kind of layer. Choosing one replaces the prompt.
 */
export function PromptLibrary({ kind, label, prompt, onUse }: { kind: LayerKind; label: string; prompt: string; onUse: (prompt: string) => void }) {
  const list = usePromptLibrary(kind);
  const [open, setOpen] = useState(false);
  const [naming, setNaming] = useState<string | null>(null);
  const trimmed = prompt.trim();
  const current = trimmed ? list.find((s) => s.prompt === trimmed) : undefined;

  const commitName = () => {
    if (naming !== null && savePrompt(kind, naming, trimmed)) setNaming(null);
  };

  return (
    <div className="sc-promptlib">
      <div className="sc-promptlib-head">
        <label className="sc-label" htmlFor="sc-prompt">
          {label}
        </label>
        {naming === null && (
          <span className="sc-promptlib-actions">
            {current ? (
              <span className="sc-chip sc-promptlib-saved" title={`Saved as “${current.name}”`}>
                ★ Saved
              </span>
            ) : (
              <button type="button" className="sc-chip" disabled={!trimmed} onClick={() => setNaming(suggestName(trimmed))} title={trimmed ? 'Keep this prompt in your library' : 'Write a prompt to save it'}>
                ☆ Save
              </button>
            )}
            <button
              type="button"
              className={`sc-chip${open ? ' on' : ''}`}
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              title={list.length ? 'Search your saved prompts' : `No saved ${kind === 'motion' ? 'motion' : 'edit'} prompts yet: write one and press Save`}
            >
              Library{list.length ? ` ${list.length}` : ''} {open ? '▴' : '▾'}
            </button>
          </span>
        )}
      </div>
      {naming !== null && (
        <div className="sc-promptlib-name">
          <input
            className="sc-input"
            autoFocus
            value={naming}
            placeholder="Name"
            aria-label="Saved prompt name"
            onChange={(e) => setNaming(e.target.value)}
            onFocus={(e) => e.target.select()}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                commitName();
              } else if (e.key === 'Escape') {
                e.preventDefault();
                e.stopPropagation();
                setNaming(null);
              }
            }}
          />
          <button type="button" className="sc-btn sc-btn-primary" disabled={!naming.trim()} onClick={commitName}>
            Save
          </button>
          <button type="button" className="sc-btn sc-btn-quiet" onClick={() => setNaming(null)}>
            Cancel
          </button>
        </div>
      )}
      {open && (
        <LibraryList
          kind={kind}
          list={list}
          current={current?.id}
          onUse={(s) => {
            markUsed(kind, s.id);
            onUse(s.prompt);
            setOpen(false);
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  );
}

function LibraryList({ kind, list, current, onUse, onClose }: { kind: LayerKind; list: SavedPrompt[]; current?: string; onUse: (s: SavedPrompt) => void; onClose: () => void }) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const hits = useMemo(() => searchPrompts(list, query), [list, query]);

  useEffect(() => setCursor(0), [query]);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-i="${cursor}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [cursor]);

  if (!list.length) {
    return <div className="sc-promptlib-empty">No saved {kind === 'motion' ? 'motion' : 'edit'} prompts yet. Write one, then press ☆ Save.</div>;
  }
  return (
    <div className="sc-promptlib-panel">
      <input
        className="sc-input"
        autoFocus
        value={query}
        placeholder={`Search ${list.length} saved prompt${list.length === 1 ? '' : 's'}`}
        aria-label="Search saved prompts"
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            setCursor((c) => Math.min(hits.length - 1, c + 1));
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setCursor((c) => Math.max(0, c - 1));
          } else if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            if (hits[cursor]) onUse(hits[cursor]);
          } else if (e.key === 'Escape') {
            e.preventDefault();
            e.stopPropagation();
            onClose();
          }
        }}
      />
      <div className="sc-promptlib-list" ref={listRef} role="listbox" aria-label="Saved prompts">
        {hits.length === 0 && <div className="sc-promptlib-empty">Nothing matches “{query}”.</div>}
        {hits.map((s, i) => (
          <div key={s.id} data-i={i} role="option" aria-selected={i === cursor} className={`sc-promptlib-row${i === cursor ? ' cursor' : ''}${s.id === current ? ' current' : ''}`} onMouseEnter={() => setCursor(i)}>
            <button type="button" className="sc-promptlib-use" onClick={() => onUse(s)} title={s.prompt}>
              <span className="sc-promptlib-rowname">{s.name}</span>
              <span className="sc-promptlib-text">{s.prompt}</span>
            </button>
            <button type="button" className="sc-btn sc-btn-quiet sc-queue-x" onClick={() => deletePrompt(kind, s.id)} aria-label={`Delete saved prompt ${s.name}`} title="Delete">
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
