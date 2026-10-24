/**
 * The media-server credential is one shared thing across the viewer and the
 * Scene window (server-auth.ts), and a server redirect to its login page is
 * an auth failure, never "cannot reach the media server".
 */
const mockStoreData: Record<string, unknown> = {};

jest.mock('../renderer/platform', () => ({
  isElectron: true,
  mediaServerBase: 'http://localhost:10111',
  store: {
    get: (k: string, d?: unknown) => (k in mockStoreData ? mockStoreData[k] : d),
    set: (k: string, v: unknown) => {
      mockStoreData[k] = v;
    },
    getMany: () => ({}),
  },
}));

import {
  checkServerToken,
  forgetServerToken,
  rememberToken,
  serverAuthHeaders,
  tokensByDb,
} from '../renderer/server-auth';
import { sceneApi, SceneApiError } from '../renderer/components/scene/scene-api';

const fetchMock = jest.fn();

beforeAll(() => {
  (global as any).fetch = fetchMock;
  if (!(AbortSignal as any).timeout) {
    (AbortSignal as any).timeout = () => new AbortController().signal;
  }
});

beforeEach(() => {
  fetchMock.mockReset();
  for (const k of Object.keys(mockStoreData)) delete mockStoreData[k];
});

describe('serverAuthHeaders', () => {
  it('sends the token the viewer stored, read fresh on every call', () => {
    expect(serverAuthHeaders()).toEqual({});
    mockStoreData.authToken = 'lk_one';
    expect(serverAuthHeaders()).toEqual({ Authorization: 'Bearer lk_one' });
    // A sign-in in the viewer is visible to the next request right away.
    mockStoreData.authToken = 'lk_two';
    expect(serverAuthHeaders()).toEqual({ Authorization: 'Bearer lk_two' });
  });
});

describe('checkServerToken', () => {
  it('reads /auth/status, and treats a dead server as unknown, not invalid', async () => {
    fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ loggedIn: true }) });
    await expect(checkServerToken('lk_a')).resolves.toBe('valid');
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:10111/auth/status');
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer lk_a' });

    fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ loggedIn: false }) });
    await expect(checkServerToken('lk_stale')).resolves.toBe('invalid');

    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    await expect(checkServerToken('lk_a')).resolves.toBe('unreachable');
  });
});

describe('per-library tokens', () => {
  it('remembers a key per library and forgets it on sign-out', () => {
    rememberToken('C:/a.sqlite', 'lk_a');
    rememberToken('C:/b.sqlite', 'lk_b');
    expect(tokensByDb()).toEqual({ 'C:/a.sqlite': 'lk_a', 'C:/b.sqlite': 'lk_b' });
    forgetServerToken('C:/a.sqlite');
    expect(tokensByDb()).toEqual({ 'C:/b.sqlite': 'lk_b' });
  });
});

describe('Scene API auth failures', () => {
  it('reports a redirect to the login page as not signed in', async () => {
    mockStoreData.authToken = 'lk_stale';
    fetchMock.mockResolvedValueOnce({ type: 'opaqueredirect', status: 0, ok: false });
    const err = await sceneApi.list().catch((e) => e);
    expect(err).toBeInstanceOf(SceneApiError);
    expect(err.status).toBe(401);
    expect(err.message).toMatch(/Not signed in/);
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer lk_stale' });
    expect(fetchMock.mock.calls[0][1].redirect).toBe('manual');
  });

  it('reports a 401 as not signed in', async () => {
    fetchMock.mockResolvedValueOnce({
      type: 'basic',
      status: 401,
      ok: false,
      text: async () => 'Unauthorized',
    });
    const err = await sceneApi.list().catch((e) => e);
    expect(err.status).toBe(401);
    expect(err.message).toMatch(/Not signed in/);
  });

  it('keeps "cannot reach" for a real network failure', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    const err = await sceneApi.list().catch((e) => e);
    expect(err.status).toBe(0);
    expect(err.message).toBe('cannot reach the media server');
  });
});
