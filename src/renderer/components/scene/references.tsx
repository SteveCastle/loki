import React, { useState } from 'react';
import { mediaUrl } from '../../platform';
import { hasVoice, itemKind, newRefId, refKind, refKinds, refTokens, remapRefTokens, selfToken, tokenIssues, type RefKind, type SceneRefItem } from './refs';
import type { LayerOptions, SceneLayer } from './scene-api';

// A layer's References: the crop itself (always first) and the images, videos
// or audio it is given besides, each with the token its prompt names it by.
// Click a token to insert it in the prompt; reordering or removing renumbers
// the prompt's tokens so its words keep pointing at the same files. A paint
// layer can be a reference too: the server renders it over the picture this
// layer is made from, lined up with the crop (rough colour to finish, a
// sketch of where things go).

const MAX: Record<RefKind, number> = { image: 8, video: 3, audio: 3 };

export interface ReferencesProps {
  layer: SceneLayer;
  /** Insert text at the prompt's cursor. */
  onInsert: (token: string) => void;
  onChange: (refs: SceneRefItem[], prompt: string) => void;
  /** Pick in a picker window (the viewer's search and browsing); resolves null when cancelled. */
  onPickLibrary: (kinds: RefKind[]) => { result: Promise<string[] | null>; cancel: () => void };
  /** Pick files with the system dialog (or the server's file browser on the web). */
  onPickFiles: (kinds: RefKind[]) => Promise<string[]>;
  /** The scene's paint layers this layer can reference. */
  paintLayers: SceneLayer[];
}

function Thumb({ path }: { path: string }) {
  const k = refKind(path);
  if (k === 'image') return <img className="sc-ref-thumb" src={mediaUrl(path)} alt="" loading="lazy" />;
  if (k === 'video') return <video className="sc-ref-thumb" src={mediaUrl(path)} muted preload="metadata" />;
  return (
    <span className="sc-ref-thumb sc-ref-audio" aria-hidden>
      ♪
    </span>
  );
}

const basename = (p: string) => p.split(/[\\/]/).pop() || p;

function PaintThumb({ layer }: { layer: SceneLayer | undefined }) {
  if (!layer || !layer.mask) {
    return (
      <span className="sc-ref-thumb sc-ref-audio" aria-hidden>
        ✎
      </span>
    );
  }
  return <img className="sc-ref-thumb sc-ref-paint" src={layer.mask} alt="" />;
}

export default function References({ layer: l, onInsert, onChange, onPickLibrary, onPickFiles, paintLayers }: ReferencesProps) {
  const o: LayerOptions = l.options || {};
  const refs: SceneRefItem[] = o.refs || [];
  const voice = hasVoice(o);
  const tokens = refTokens(l.kind, refs, voice);
  const kinds = refKinds(l.kind);
  const issues = tokenIssues(l.prompt, l.kind, refs, voice);
  const [pending, setPending] = useState<null | (() => void)>(null);
  const [note, setNote] = useState<string | null>(null);

  const apply = (next: SceneRefItem[]) => onChange(next, remapRefTokens(l.prompt, l.kind, refs, next, voice));

  const add = (paths: string[]) => {
    const have = new Set(refs.map((r) => r.path.toLowerCase()));
    const counts: Record<RefKind, number> = { image: 0, video: 0, audio: voice ? 1 : 0 };
    refs.forEach((r) => {
      const k = itemKind(r);
      if (k) counts[k] += 1;
    });
    const next = [...refs];
    let skipped = 0;
    paths.forEach((p) => {
      const k = refKind(p);
      if (!k || !kinds.includes(k) || have.has(p.toLowerCase()) || counts[k] >= MAX[k]) {
        skipped += 1;
        return;
      }
      counts[k] += 1;
      have.add(p.toLowerCase());
      next.push({ id: newRefId(), path: p });
    });
    setNote(skipped ? `${skipped} skipped (already added, not a kind this layer takes, or over the engine’s limit)` : null);
    if (next.length !== refs.length) apply(next);
  };

  const fromLibrary = () => {
    const req = onPickLibrary(kinds);
    setPending(() => req.cancel);
    req.result.then((paths) => {
      setPending(null);
      if (paths && paths.length) add(paths);
    });
  };

  const linked = new Set(refs.map((r) => r.layer).filter(Boolean));
  const addPaint = (id: string) => {
    const p = paintLayers.find((x) => x.id === id);
    if (!p || linked.has(id)) return;
    if (refs.filter((r) => itemKind(r) === 'image').length >= MAX.image) {
      setNote('Over the engine’s limit of images');
      return;
    }
    apply([...refs, { id: newRefId(), path: '', layer: id, note: p.name || 'my rough painting' }]);
  };
  const paintName = (id: string) => paintLayers.find((x) => x.id === id)?.name || 'Paint (deleted)';

  const move = (i: number, d: -1 | 1) => {
    const j = i + d;
    if (j < 0 || j >= refs.length) return;
    const next = [...refs];
    [next[i], next[j]] = [next[j], next[i]];
    apply(next);
  };

  return (
    <div className="sc-refs">
      <div className="sc-refs-head">
        <span className="sc-label" title={`Besides this area, the engine can be given ${kinds.join(', ')} to use as the prompt says`}>
          References
        </span>
        <span className="sc-refs-add">
          <button type="button" className="sc-btn sc-btn-tiny" disabled={!!pending} onClick={fromLibrary} title="Pick in a library window: search, tags and browsing as usual; it closes when you're done">
            + From library
          </button>
          <button type="button" className="sc-btn sc-btn-quiet sc-btn-tiny" disabled={!!pending} onClick={() => onPickFiles(kinds).then(add)} title="Files outside the library">
            Files…
          </button>
          {paintLayers.some((p) => !linked.has(p.id)) && (
            <select
              className="sc-select sc-btn-tiny"
              value=""
              onChange={(e) => e.target.value && addPaint(e.target.value)}
              title="Give the engine one of this scene's paint layers, painted over the picture and lined up with this area"
              aria-label="Add a paint layer as a reference"
            >
              <option value="">+ Paint layer…</option>
              {paintLayers
                .filter((p) => !linked.has(p.id))
                .map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name || 'Paint'}
                  </option>
                ))}
            </select>
          )}
        </span>
      </div>
      {pending && (
        <div className="sc-ref-pending">
          Choose in the picker window…
          <button type="button" className="sc-btn sc-btn-quiet sc-btn-tiny" onClick={() => pending()}>
            Cancel
          </button>
        </div>
      )}
      <ul className="sc-ref-list">
        <li className="sc-ref self">
          <span className="sc-ref-thumb sc-ref-self" aria-hidden>
            ▣
          </span>
          <button type="button" className="sc-token" onClick={() => onInsert(selfToken(l.kind))} title="Insert in the prompt">
            {selfToken(l.kind)}
          </button>
          <span className="sc-faint">this area (the engine’s main input)</span>
        </li>
        {voice && l.kind === 'motion' && (
          <li className="sc-ref self">
            <span className="sc-ref-thumb sc-ref-audio" aria-hidden>
              ♪
            </span>
            <button type="button" className="sc-token" onClick={() => onInsert('<Audio 1>')} title="Insert in the prompt">
              {'<Audio 1>'}
            </button>
            <span className="sc-faint">the dialogue voice</span>
          </li>
        )}
        {refs.map((r, i) => (
          <li key={r.id} className={`sc-ref${tokens[i] ? '' : ' unusable'}`}>
            {r.layer ? <PaintThumb layer={paintLayers.find((x) => x.id === r.layer)} /> : <Thumb path={r.path} />}
            {tokens[i] ? (
              <button type="button" className="sc-token" onClick={() => onInsert(tokens[i] as string)} title={`Insert in the prompt · ${r.layer ? `paint layer “${paintName(r.layer)}”, over the picture` : r.path}`}>
                {tokens[i]}
              </button>
            ) : (
              <span className="sc-token off" title="An Edit layer only uses images">
                unused
              </span>
            )}
            <input
              className="sc-input sc-ref-note"
              value={r.note || ''}
              placeholder={r.layer ? paintName(r.layer) : basename(r.path)}
              title="What it is (e.g. “the red jacket”): Motion prompts introduce the reference with it"
              onChange={(e) => onChange(refs.map((x) => (x.id === r.id ? { ...x, note: e.target.value } : x)), l.prompt)}
              aria-label="What this reference is"
            />
            <span className="sc-ref-tools">
              <button type="button" disabled={i === 0} onClick={() => move(i, -1)} title="Move up (renumbers the prompt)">
                ▲
              </button>
              <button type="button" disabled={i === refs.length - 1} onClick={() => move(i, 1)} title="Move down (renumbers the prompt)">
                ▼
              </button>
              <button type="button" className="sc-row-x" onClick={() => apply(refs.filter((x) => x.id !== r.id))} title="Remove">
                ×
              </button>
            </span>
          </li>
        ))}
      </ul>
      {note && <div className="sc-faint sc-fit">{note}</div>}
      {issues.length > 0 && (
        <div className="sc-warn">
          {issues.join(', ')} {issues.length > 1 ? "don't" : "doesn't"} name anything this layer has: add the reference, or fix the token.
        </div>
      )}
    </div>
  );
}
