import { isElectron } from './platform';

// Access state for the web build: whether the visitor is signed in and
// whether the server is running in Allow Public Access mode. Electron is
// always full-featured, so this module is effectively a no-op there.

export interface AccessInfo {
  loggedIn: boolean;
  publicAccess: boolean;
  canWrite: boolean;
}

// Flag-off + logged-out never renders the SPA (the server 302s /app/ to
// /login), so the permissive branch just preserves the existing
// 401 -> /login behavior if it somehow occurs.
export function deriveCanWrite(a: {
  loggedIn: boolean;
  publicAccess: boolean;
}): boolean {
  return a.loggedIn || !a.publicAccess;
}

let cached: AccessInfo = { loggedIn: true, publicAccess: false, canWrite: true };

// Fetches /auth/status before the state machine starts so canWrite is
// correct in the machine's initial context (no hidden-then-shown flicker).
export async function initAccess(): Promise<AccessInfo> {
  if (isElectron) return cached;
  try {
    const res = await fetch('/auth/status', {
      credentials: 'include',
      headers: { Accept: 'application/json' },
    });
    const data = await res.json();
    cached = {
      loggedIn: !!data.loggedIn,
      publicAccess: !!data.publicAccess,
      canWrite: deriveCanWrite({
        loggedIn: !!data.loggedIn,
        publicAccess: !!data.publicAccess,
      }),
    };
  } catch {
    // Server unreachable: keep the permissive default — the server still
    // gates every write, this flag only controls what UI is shown.
  }
  return cached;
}

export function getAccess(): AccessInfo {
  return cached;
}
