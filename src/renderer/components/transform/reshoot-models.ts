// The server's extra video models (config.json reshootModels, GET /api/reshoot/models), offered in the Transform
// flow's Model control beside H3 and FastH3 Omni. The list is fetched each time the flow opens (config edits show up
// after a server restart) and kept between openings so the control does not flicker.
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

export function useReshootModels(authToken: string | null): ReshootModelChoice[] {
  const [models, setModels] = useState<ReshootModelChoice[]>(cached);
  useEffect(() => {
    if (!authToken) return undefined;
    const ctl = new AbortController();
    const giveUp = window.setTimeout(() => ctl.abort(), 10000);
    fetch(`${mediaServerBase}/api/reshoot/models`, { headers: { Authorization: `Bearer ${authToken}` }, signal: ctl.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        // an older server has no such endpoint: no extra models
        cached = parseReshootModels(body);
        setModels(cached);
      })
      .catch(() => undefined)
      .finally(() => window.clearTimeout(giveUp));
    return () => {
      window.clearTimeout(giveUp);
      ctl.abort();
    };
  }, [authToken]);
  return models;
}
