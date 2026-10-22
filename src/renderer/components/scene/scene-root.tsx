import React, { useCallback, useEffect, useState } from 'react';
import type { Root } from 'react-dom/client';
import { sceneApi, type Scene } from './scene-api';
import SceneEditor from './scene-editor';
import './scene-editor.css';

// Boot of a Scene editor window (index.tsx hands over when the page has ?scene=).
// The param is JSON {path, time} (a file to open its latest scene for, or start
// one from) or {id} (a scene to reopen).

type Launch = { path?: string; time?: number; id?: string };

function parseLaunch(raw: string): Launch | null {
  try {
    const v = JSON.parse(raw);
    if (v && (typeof v.path === 'string' || typeof v.id === 'string')) return v as Launch;
  } catch {
    // a bare path from a hand-written URL
    if (raw) return { path: raw };
  }
  return null;
}

function SceneRoot({ launch }: { launch: Launch | null }) {
  const [scene, setScene] = useState<Scene | null>(null);
  const [error, setError] = useState<string | null>(launch ? null : 'Nothing to open: the window was started without a file.');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!launch) return undefined;
    let alive = true;
    setError(null);
    const req = launch.id ? sceneApi.get(launch.id) : sceneApi.open(launch.path as string, launch.time || 0);
    req.then(
      (s) => alive && setScene(s),
      (e) => alive && setError(e instanceof Error ? e.message : String(e))
    );
    return () => {
      alive = false;
    };
  }, [launch, attempt]);

  useEffect(() => {
    document.title = scene ? `${scene.name} · Scene editor` : 'Scene editor';
  }, [scene]);

  const onSwitch = useCallback((next: Scene) => setScene(next), []);

  if (scene) return <SceneEditor key={scene.id} initial={scene} onSwitch={onSwitch} />;
  return (
    <div className="sc-app sc-boot">
      {error ? (
        <div className="sc-boot-card">
          <strong>Could not open the scene</strong>
          <p>{error}</p>
          {launch && (
            <button type="button" className="sc-btn sc-btn-primary" onClick={() => setAttempt((n) => n + 1)}>
              Try again
            </button>
          )}
        </div>
      ) : (
        <div className="sc-boot-card">
          <span className="sc-spinner" aria-hidden />
          <p>Preparing the scene…</p>
        </div>
      )}
    </div>
  );
}

export function renderSceneEditor(root: Root, raw: string): void {
  document.body.classList.add('sc-body');
  root.render(<SceneRoot launch={parseLaunch(raw)} />);
}
