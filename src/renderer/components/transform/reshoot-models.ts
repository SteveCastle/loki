// The server's extra video models (config.json reshootModels, GET /api/reshoot/models), offered beside H3 and FastH3
// Omni in the Transform flow's Model control and the Scene editor's motion Engine control. The list is fetched each
// time a control mounts (config edits show up after a server restart) and kept between mounts so it does not flicker.
import { useEffect, useState } from 'react';
import { mediaServerBase } from '../../platform';
import type { ReshootModel } from './intents';

export interface ReshootModelChoice {
  file: string;
  label: string;
  recipe: ReshootModel;
}

let cached: ReshootModelChoice[] = [];

/** The answer's models, each checked (it is data from the server). */
export function parseReshootModels(body: unknown): ReshootModelChoice[] {
  const list = body && typeof body === 'object' ? (body as { models?: unknown }).models : null;
  if (!Array.isArray(list)) return [];
  return list
    .filter((m) => m && typeof m.file === 'string' && m.file)
    .map((m) => ({ file: m.file, label: typeof m.label === 'string' && m.label ? m.label : m.file, recipe: m.recipe === 'fast' ? 'fast' : 'h3' }));
}

/**
 * The models, loaded with `load` (the answer's body, or null) whenever `key` changes; a null key loads nothing. An
 * older server has no such endpoint: no extra models.
 */
export function useReshootModelList(key: string | null, load: (signal: AbortSignal) => Promise<unknown>): ReshootModelChoice[] {
  const [models, setModels] = useState<ReshootModelChoice[]>(cached);
  useEffect(() => {
    if (key === null) return undefined;
    const ctl = new AbortController();
    const giveUp = window.setTimeout(() => ctl.abort(), 10000);
    load(ctl.signal)
      .then((body) => {
        cached = parseReshootModels(body);
        setModels(cached);
      })
      .catch(() => undefined)
      .finally(() => window.clearTimeout(giveUp));
    return () => {
      window.clearTimeout(giveUp);
      ctl.abort();
    };
    // load is the caller's way to fetch; key says when to fetch again
  }, [key]);
  return models;
}

export function useReshootModels(authToken: string | null): ReshootModelChoice[] {
  return useReshootModelList(authToken, (signal) =>
    fetch(`${mediaServerBase}/api/reshoot/models`, { headers: { Authorization: `Bearer ${authToken}` }, signal }).then((res) => (res.ok ? res.json() : null)),
  );
}
