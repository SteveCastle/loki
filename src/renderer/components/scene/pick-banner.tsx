import React, { useContext, useEffect } from 'react';
import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import { answerPick, listenForPickRequests, pickable, usePickRequest } from './pick-bridge';
import { refKind } from './refs';
import './pick-banner.css';

// The viewer's side of picking Scene editor references: while an editor waits,
// a banner says so and adds the item on screen; the context palette's "Add to
// Scene" adds a whole selection (ScenePickPaletteSection). Everything else, from
// search to tags to browsing, is the viewer as usual.

function currentPath(libraryService: any): string | undefined {
  const ctx = libraryService.getSnapshot().context;
  return filter(ctx.libraryLoadId, ctx.textFilter, ctx.library, ctx.settings.filters, ctx.settings.sortBy)[ctx.cursor]?.path;
}

const kindsLabel = (kinds: string[]) => (kinds.length === 1 ? `${kinds[0]}s` : `${kinds.slice(0, -1).join(', ')} or ${kinds[kinds.length - 1]}`);

export default function ScenePickBanner() {
  const { libraryService } = useContext(GlobalStateContext);
  const req = usePickRequest();
  useEffect(() => listenForPickRequests(), []);
  // Esc cancels the pick (only while one is open; the viewer keeps Esc otherwise).
  useEffect(() => {
    if (!req) return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && e.shiftKey) answerPick(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [req]);
  if (!req) return null;
  const here = currentPath(libraryService);
  const addable = here ? pickable([here], req, refKind) : [];
  return (
    <div className="scene-pick-banner" role="status">
      <span className="scene-pick-dot" aria-hidden />
      <span className="scene-pick-text">
        <strong>Choosing references</strong> for {req.label} · {kindsLabel(req.kinds)}. Search and browse as usual; select several with
        Shift+right-click / Ctrl+click and use <em>Add to Scene</em> in the palette.
      </span>
      <button
        type="button"
        className="scene-pick-btn primary"
        disabled={!addable.length}
        title={addable.length ? here : 'The item on screen is not a kind this layer can use'}
        onClick={() => answerPick(addable)}
      >
        Add this item
      </button>
      <button type="button" className="scene-pick-btn" onClick={() => answerPick(null)} title="Shift+Esc">
        Cancel
      </button>
    </div>
  );
}

/** The context palette's action while a pick is open: add the palette's working set. */
export function ScenePickPaletteSection({ paths, onDone }: { paths: string[]; onDone: () => void }) {
  const req = usePickRequest();
  if (!req) return null;
  const add = pickable(paths, req, refKind);
  return (
    <div className="context-palette-merge scene-pick-section">
      <div className="workflow-picker-header">
        <span className="action-group-title">Scene references</span>
      </div>
      <button
        type="button"
        className="merge-selection-btn"
        disabled={!add.length}
        onClick={() => {
          answerPick(add);
          onDone();
        }}
      >
        {add.length ? `Add ${add.length} to ${req.label}` : `Nothing here is a ${kindsLabel(req.kinds).replace(/s$/, '')}`}
      </button>
      {paths.length > add.length && add.length > 0 && <span className="merge-selection-note">{paths.length - add.length} of the selection can’t be used by this layer</span>}
    </div>
  );
}
