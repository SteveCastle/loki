// The upload queue drives a single progress toast: ADD_TOAST with a
// caller-supplied id, then UPDATE_TOAST to advance the count in place (no
// stacking), then a final UPDATE_TOAST to a short duration. These tests pin
// that machine behavior.

jest.mock('../renderer/hooks/useSessionStore', () => ({
  initSessionStore: jest.fn().mockResolvedValue({}),
  getSessionValue: jest.fn(),
  setSessionValue: jest.fn(),
  setSessionValues: jest.fn(),
  clearSessionKeys: jest.fn(),
  flushSession: jest.fn(),
  hasPersistedLibrary: jest.fn(() => false),
  hasPersistedTextFilter: jest.fn(() => false),
  hasPersistedTags: jest.fn(() => false),
}));

jest.mock('../renderer/platform', () => ({
  invoke: jest.fn(),
  send: jest.fn(),
  on: jest.fn(() => () => {}),
  isElectron: true,
  mediaServerBase: 'http://localhost:10111',
  capabilities: {},
  appArgs: { dbPath: 'web' },
  store: {
    get: (_k: string, d: any) => d,
    set: jest.fn(),
    getMany: (pairs: [string, any][]) =>
      Object.fromEntries(pairs.map(([k, def]) => [k, def])),
  },
  sessionStore: {},
  loadMediaFromDB: jest.fn(),
  loadMediaByQuery: jest.fn(),
  mediaUrl: (p: string) => p,
  hlsUrl: null,
  transcript: {},
}));

jest.mock('../renderer/access', () => ({
  getAccess: () => ({
    loggedIn: true,
    publicAccess: false,
    canWrite: true,
    defaultStartPath: '',
  }),
  initAccess: jest.fn().mockResolvedValue({}),
  deriveCanWrite: () => true,
}));

import { interpret } from 'xstate';
// eslint-disable-next-line import/first
import { libraryMachine } from '../renderer/state';

function startService() {
  const service = interpret(libraryMachine);
  service.start();
  return service;
}

describe('progress toast (ADD_TOAST id + UPDATE_TOAST)', () => {
  it('ADD_TOAST honors a caller-supplied id', () => {
    const service = startService();
    service.send({
      type: 'ADD_TOAST',
      data: { id: 'upload-1', type: 'info', title: 'Uploading…', message: '0 / 3' },
    });
    const toasts = service.getSnapshot().context.toasts;
    const t = (toasts as any[]).find((x: any) => x.id === 'upload-1');
    expect(t).toBeTruthy();
    expect(t.message).toBe('0 / 3');
    service.stop();
  });

  it('UPDATE_TOAST advances the same toast in place without stacking', () => {
    const service = startService();
    service.send({
      type: 'ADD_TOAST',
      data: { id: 'upload-2', type: 'info', title: 'Uploading…', message: '0 / 3' },
    });
    service.send({ type: 'UPDATE_TOAST', data: { id: 'upload-2', message: '1 / 3' } });
    service.send({ type: 'UPDATE_TOAST', data: { id: 'upload-2', message: '2 / 3' } });
    service.send({
      type: 'UPDATE_TOAST',
      data: { id: 'upload-2', type: 'success', title: 'Import complete', message: 'Added 3 files', durationMs: 4000 },
    });

    const mine = (service.getSnapshot().context.toasts as any[]).filter(
      (x: any) => x.id === 'upload-2'
    );
    expect(mine).toHaveLength(1); // never stacked
    expect(mine[0].message).toBe('Added 3 files');
    expect(mine[0].type).toBe('success');
    expect(mine[0].durationMs).toBe(4000);
    service.stop();
  });

  it('UPDATE_TOAST for an unknown id is a no-op', () => {
    const service = startService();
    const before = service.getSnapshot().context.toasts.length;
    service.send({ type: 'UPDATE_TOAST', data: { id: 'nope', message: 'x' } });
    expect(service.getSnapshot().context.toasts.length).toBe(before);
    service.stop();
  });
});
