import React, { useEffect, useMemo, useRef, useState } from 'react';
import { availability, classify, intentById, type IntentId } from '../transform/intents';
import {
  deleteTemplate,
  restoreTemplate,
  templateSummary,
  useAllTemplates,
  type PromptTemplate,
} from '../transform/store';
import { IntentIcon } from '../transform/icons';

// The palette's "Saved" list: one searchable place for everything the user
// saved to run again — saved Transform tasks (prompts and preset settings,
// kept in this browser by the Transform store) and the media server's
// Workflows (a saved DAG of tasks). Typing filters both; a Transform opens
// the Transform flow to check and run, a Workflow runs on the selection.

export interface SavedWorkflow {
  id: string;
  name: string;
}

export type SavedEntry =
  | { kind: 'workflow'; key: string; name: string; detail: string; wf: SavedWorkflow }
  | { kind: 'transform'; key: string; name: string; detail: string; intent: IntentId; t: PromptTemplate };

/**
 * Every saved item that can run on this selection, by name: the saved
 * transforms whose intent fits `transformPaths` (none when Transform can't
 * run here), and every workflow.
 */
export function savedEntries(
  workflows: SavedWorkflow[],
  templates: Array<{ intent: IntentId; t: PromptTemplate }>,
  transformPaths: string[]
): SavedEntry[] {
  const inputs = classify(transformPaths);
  const out: SavedEntry[] = [];
  if (transformPaths.length) {
    templates.forEach(({ intent, t }) => {
      if (!availability(intent, inputs).ok) return;
      out.push({ kind: 'transform', key: `t:${intent}:${t.id}`, name: t.name, detail: `${intentById(intent).title} · ${templateSummary(intent, t)}`, intent, t });
    });
  }
  workflows.forEach((wf) => out.push({ kind: 'workflow', key: `w:${wf.id}`, name: wf.name, detail: 'Workflow', wf }));
  return out.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
}

/** Every word of the query appears in the name, the summary or the kind ("workflow", "transform", "edit"…). */
export function matchesSearch(entry: SavedEntry, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const kind = entry.kind === 'workflow' ? 'workflow dag' : 'transform';
  const hay = `${entry.name} ${entry.detail} ${kind}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

export default function SavedPicker({
  workflows,
  transformPaths,
  onRunWorkflow,
  onOpenTransform,
}: {
  workflows: SavedWorkflow[];
  /** The selection Transform would act on; [] when it can't run here. */
  transformPaths: string[];
  onRunWorkflow: (wf: SavedWorkflow) => void;
  onOpenTransform: (intent: IntentId, t: PromptTemplate) => void;
}) {
  const templates = useAllTemplates();
  const [search, setSearch] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [undo, setUndo] = useState<{ intent: IntentId; template: PromptTemplate; index: number } | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const entries = useMemo(() => savedEntries(workflows, templates, transformPaths), [workflows, templates, transformPaths]);
  const filtered = useMemo(() => entries.filter((e) => matchesSearch(e, search)), [entries, search]);
  const transforms = entries.filter((e) => e.kind === 'transform').length;

  useEffect(() => {
    setSelectedIdx(0);
  }, [search]);
  useEffect(() => {
    setSelectedIdx((i) => Math.min(i, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  useEffect(() => {
    const item = listRef.current?.children[selectedIdx] as HTMLElement | undefined;
    item?.scrollIntoView?.({ block: 'nearest' });
  }, [selectedIdx]);

  const run = (e: SavedEntry) => {
    if (e.kind === 'workflow') onRunWorkflow(e.wf);
    else onOpenTransform(e.intent, e.t);
  };

  const remove = (e: SavedEntry) => {
    if (e.kind !== 'transform') return;
    const removed = deleteTemplate(e.intent, e.t.id);
    if (removed) setUndo({ intent: e.intent, ...removed });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const current = filtered[selectedIdx];
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && current) {
      e.preventDefault();
      e.stopPropagation();
      run(current);
    } else if (e.key === 'Delete' && e.shiftKey && current?.kind === 'transform') {
      e.preventDefault();
      e.stopPropagation();
      remove(current);
    }
  };

  if (!entries.length && !undo) return null;
  const kinds = [transforms ? 'transforms' : '', entries.length > transforms ? 'workflows' : ''].filter(Boolean).join(' and ');

  return (
    <div className="context-palette-workflows saved-picker" onKeyDown={handleKeyDown}>
      <div className="workflow-picker-header">
        <span className="action-group-title">Saved</span>
        <span className="workflow-count">
          {search ? `${filtered.length} of ${entries.length}` : entries.length}
        </span>
      </div>
      {entries.length > 0 && (
        <input
          className="workflow-search"
          type="text"
          placeholder={`Search saved ${kinds || 'items'}…`}
          aria-label="Search saved transforms and workflows"
          title="↑↓ to pick · Enter to open · Shift+Delete removes a saved transform"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      )}
      {undo && (
        <div className="saved-undo" role="status">
          <span>
            Deleted “{undo.template.name}”
          </span>
          <button
            type="button"
            onClick={() => {
              restoreTemplate(undo.intent, undo.template, undo.index);
              setUndo(null);
            }}
          >
            Undo
          </button>
        </div>
      )}
      <div className="workflow-list saved-list" ref={listRef} role="listbox" aria-label="Saved transforms and workflows">
        {filtered.map((e, i) => (
          <div
            key={e.key}
            role="option"
            aria-selected={i === selectedIdx}
            className={`workflow-item saved-item ${e.kind}${i === selectedIdx ? ' selected' : ''}`}
            onClick={() => run(e)}
            onMouseEnter={() => setSelectedIdx(i)}
            title={e.kind === 'workflow' ? `Run the workflow “${e.name}” on the selection` : `Open “${e.name}” in Transform to check and run\n${e.detail}`}
          >
            <span className="saved-item-icon" aria-hidden="true">
              {e.kind === 'transform' ? <IntentIcon id={e.intent} size={14} /> : <WorkflowGlyph />}
            </span>
            <span className="saved-item-text">
              <span className="saved-item-name">{e.name}</span>
              <span className="saved-item-detail">
                <span className={`saved-kind ${e.kind}`}>{e.kind === 'workflow' ? 'Workflow' : 'Transform'}</span>
                {e.kind === 'transform' ? e.detail : ''}
              </span>
            </span>
            {e.kind === 'transform' && (
              <button
                type="button"
                className="saved-item-delete"
                aria-label={`Delete saved transform ${e.name}`}
                title="Delete (Shift+Delete)"
                onClick={(ev) => {
                  ev.stopPropagation();
                  remove(e);
                }}
              >
                ×
              </button>
            )}
          </div>
        ))}
        {entries.length > 0 && filtered.length === 0 && <div className="workflow-empty">No matches</div>}
      </div>
    </div>
  );
}

/** Three linked nodes: a DAG. */
function WorkflowGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="6" height="6" rx="1.5" />
      <rect x="15" y="3" width="6" height="6" rx="1.5" />
      <rect x="9" y="15" width="6" height="6" rx="1.5" />
      <path d="M6 9v2.5a1.5 1.5 0 0 0 1.5 1.5H12m6-4v2.5a1.5 1.5 0 0 1-1.5 1.5H12m0 0v2" />
    </svg>
  );
}
