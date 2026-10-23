import React, { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import './login-widget.css';
import { useSelector } from '@xstate/react';
import { mediaServerBase, isElectron, invoke } from '../../platform';
import { initAccess } from '../../access';

export default function LoginWidget() {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Verify stored token is still valid on mount
  useEffect(() => {
    const verifyToken = async () => {
      if (!authToken) return;

      try {
        // Use the health endpoint to verify the token works
        const response = await fetch(`${mediaServerBase}/health`, {
          headers: {
            Authorization: `Bearer ${authToken}`,
          },
        });
        if (!response.ok) {
          // Token is invalid, clear it
          libraryService.send({ type: 'SET_AUTH_TOKEN', token: null });
        }
      } catch (e) {
        // Server not available, keep the token for when it comes back
      }
    };
    verifyToken();
  }, [authToken, libraryService]);

  // Desktop: sign in through the browser like `lokictl login` — the server's
  // own login + approve page hands this app an API key (see main/server-login.ts).
  const [waiting, setWaiting] = useState(false);
  const handleBrowserLogin = async () => {
    setWaiting(true);
    setError(null);
    try {
      const res = (await invoke('server-browser-login')) as {
        token?: string;
        error?: string;
      };
      if (!res?.token) throw new Error(res?.error || 'Sign-in failed');
      libraryService.send({ type: 'SET_AUTH_TOKEN', token: res.token });
      libraryService.send({ type: 'SET_CAN_WRITE', canWrite: true });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Sign-in failed';
      if (message !== 'Sign-in cancelled') setError(message);
    } finally {
      setWaiting(false);
    }
  };

  const openConfigDir = isElectron ? (
    <button
      type="button"
      className="config-dir-link"
      onClick={() => invoke('open-config-dir')}
      title="The folder holding config.json and the server's data"
    >
      Open config folder
    </button>
  ) : null;

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch(`${mediaServerBase}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Login failed');
      }

      const data = await response.json();
      const token = data.token;

      if (!token) {
        throw new Error('No token received');
      }

      // Store the actual JWT token
      libraryService.send({ type: 'SET_AUTH_TOKEN', token });
      // Signing in always grants full features — flips a public-access
      // view-only session to the complete UI without a reload.
      libraryService.send({ type: 'SET_CAN_WRITE', canWrite: true });
      setPassword('');
      setUsername('');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLogout = async () => {
    libraryService.send({ type: 'SET_AUTH_TOKEN', token: null });
    if (!isElectron) {
      // Clear the auth cookie too, then re-derive access: on a
      // public-access server the UI drops to view-only; otherwise the
      // server-rendered login page takes over.
      try {
        await fetch('/auth/logout', {
          method: 'POST',
          credentials: 'include',
        });
      } catch {
        // best effort — the token above is already gone
      }
      const access = await initAccess();
      if (!access.loggedIn && !access.publicAccess) {
        window.location.href = '/login';
        return;
      }
      libraryService.send({ type: 'SET_CAN_WRITE', canWrite: access.canWrite });
    }
  };

  if (authToken) {
    return (
      <div className="LoginWidget logged-in">
        <div className="status">
          <span className="indicator">●</span> Authenticated
        </div>
        <button onClick={handleLogout} className="logout-button">
          Logout
        </button>
        {openConfigDir}
      </div>
    );
  }

  if (isElectron) {
    return (
      <div className="LoginWidget">
        {waiting ? (
          <>
            <div className="status">
              Approve the sign-in in your browser…
            </div>
            <button onClick={() => invoke('server-browser-login-cancel')}>
              Cancel
            </button>
          </>
        ) : (
          <button onClick={handleBrowserLogin}>Sign in with browser</button>
        )}
        {error && <div className="error-message">{error}</div>}
        {openConfigDir}
      </div>
    );
  }

  return (
    <div className="LoginWidget">
      <form onSubmit={handleLogin} className="login-form">
        <div className="input-group">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={isSubmitting}
          />
        </div>
        <div className="input-group">
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isSubmitting}
          />
        </div>
        {error && <div className="error-message">{error}</div>}
        <button type="submit" disabled={isSubmitting || !username || !password}>
          {isSubmitting ? '...' : 'Login'}
        </button>
      </form>
    </div>
  );
}
