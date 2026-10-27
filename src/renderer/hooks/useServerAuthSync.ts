// Viewer-window side of the shared media-server credential (server-auth.ts).
import { useContext, useEffect, useRef } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../state';
import { isElectron } from '../platform';
import { checkServerToken, rememberToken, tokensByDb } from '../server-auth';

// The viewer posts its dbPath to the server after load-db; give the server a
// moment to finish switching before asking it about credentials.
const SWITCH_SETTLE_MS = 1500;

/**
 * Viewer-window hook: keeps the shared credential matched to the library the
 * server has open. On a library change the token is checked; one the new
 * library doesn't know is swapped for the key this viewer previously got in
 * that library, or cleared with a toast saying why. A server that can't be
 * reached changes nothing.
 */
export function useServerAuthSync() {
  const { libraryService } = useContext(GlobalStateContext);
  const dbPath = useSelector(libraryService, (s) => s.context.dbPath);
  const authToken = useSelector(libraryService, (s) => s.context.authToken);

  useEffect(() => {
    if (!isElectron || !dbPath || dbPath === 'web' || !authToken) return undefined;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      const current = await checkServerToken(authToken);
      if (cancelled || current === 'unreachable') return;
      if (current === 'valid') {
        rememberToken(dbPath, authToken);
        return;
      }
      const saved = tokensByDb()[dbPath];
      if (saved && saved !== authToken) {
        const check = await checkServerToken(saved);
        if (cancelled) return;
        if (check === 'valid') {
          libraryService.send({ type: 'SET_AUTH_TOKEN', token: saved });
          return;
        }
      }
      libraryService.send({ type: 'SET_AUTH_TOKEN', token: null });
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'info',
          title: 'Signed out of the media server',
          message:
            'This library has its own accounts. Sign in again from the database tab of the command palette.',
          durationMs: 8000,
        },
      });
    }, SWITCH_SETTLE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [dbPath, authToken, libraryService]);

  // Switched to a library this viewer has a key for while signed out: pick
  // that key back up. Keyed on the library only — signing out of the current
  // one (forgetServerToken) must stay signed out.
  const lastDb = useRef<string | null>(null);
  useEffect(() => {
    const changed = lastDb.current !== dbPath;
    lastDb.current = dbPath;
    if (!changed) return undefined;
    if (!isElectron || !dbPath || dbPath === 'web' || authToken) return undefined;
    const saved = tokensByDb()[dbPath];
    if (!saved) return undefined;
    let cancelled = false;
    const t = window.setTimeout(async () => {
      if ((await checkServerToken(saved)) === 'valid' && !cancelled) {
        libraryService.send({ type: 'SET_AUTH_TOKEN', token: saved });
      }
    }, SWITCH_SETTLE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [dbPath]);
}
