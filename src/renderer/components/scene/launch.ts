// Opens the Scene editor in its own window: an Electron window (scene-window.ts)
// or, in the web build, a popup of this same SPA. Either way the page boots with
// ?scene=<JSON {path, time}> and index.tsx renders the editor instead of the viewer.
import { isElectron, send } from '../../platform';
import { mediaKind } from '../transform/intents';

/** Whether a file can start a scene: a still, or a video (its current frame). */
export function canStartScene(path: string | undefined): boolean {
  if (!path) return false;
  const kind = mediaKind(path);
  return kind === 'image' || kind === 'video';
}

export function sceneParam(path: string, time = 0): string {
  return JSON.stringify({ path, time });
}

export function openSceneEditor(path: string, time?: number): void {
  if (!canStartScene(path)) return;
  const t = Number.isFinite(time) ? (time as number) : 0;
  if (isElectron) {
    send('open-scene-editor', [path, t]);
    return;
  }
  const url = `${window.location.pathname}?scene=${encodeURIComponent(sceneParam(path, t))}`;
  // Named per file: launching the same file again reuses its popup.
  const name = `loki-scene-${Math.abs(hash(path)).toString(36)}`;
  const win = window.open(url, name, 'popup,width=1480,height=940');
  if (win) win.focus();
}

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return h;
}
