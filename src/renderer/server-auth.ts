// The desktop app's media-server credential, shared by every window: the
// viewer and the Scene editor (and any other window booted from this bundle)
// read the same electron-store 'authToken', which the viewer's state machine
// writes on sign-in (SET_AUTH_TOKEN). Nothing caches it per window, so a
// sign-in — or a sign-out — in the viewer applies to every open window on
// its next request.
//
// API keys live in the server's LIBRARY database, so a key is only good for
// the library it was issued in: switching databases strands it. The viewer
// keeps the key it got for each library (TOKENS_BY_DB_KEY) and swaps them as
// the library changes (hooks/useServerAuthSync.ts), so switching back and forth
// between libraries never means signing in again.
//
// Kept free of React and the state machine: the Scene window imports this
// without booting the viewer.
import { isElectron, mediaServerBase, store } from './platform';

const TOKENS_BY_DB_KEY = 'authTokensByDb';

/** The credential every window sends (Electron); the web build rides its cookie. */
export function storedServerToken(): string | null {
  if (!isElectron) return null;
  return (store.get('authToken', null) as string | null) || null;
}

export function serverAuthHeaders(): Record<string, string> {
  const token = storedServerToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export type TokenCheck = 'valid' | 'invalid' | 'unreachable';

/** Asks the server whether a credential is good for its current library. */
export async function checkServerToken(token: string): Promise<TokenCheck> {
  try {
    const res = await fetch(`${mediaServerBase}/auth/status`, {
      headers: { Authorization: `Bearer ${token}` },
      credentials: 'omit',
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return 'unreachable';
    const data = (await res.json()) as { loggedIn?: boolean };
    return data.loggedIn ? 'valid' : 'invalid';
  } catch {
    return 'unreachable';
  }
}

export function tokensByDb(): Record<string, string> {
  const v = store.get(TOKENS_BY_DB_KEY, {});
  return v && typeof v === 'object' ? (v as Record<string, string>) : {};
}

export function rememberToken(dbPath: string, token: string) {
  const map = tokensByDb();
  if (map[dbPath] === token) return;
  store.set(TOKENS_BY_DB_KEY, { ...map, [dbPath]: token });
}

/** An explicit sign-out: don't hand this library's key back on the next switch. */
export function forgetServerToken(dbPath: string) {
  const map = tokensByDb();
  if (!(dbPath in map)) return;
  const { [dbPath]: _dropped, ...rest } = map;
  store.set(TOKENS_BY_DB_KEY, rest);
}
