import React, { useContext, useEffect, useState } from 'react';
import { GlobalStateContext } from '../../state';
import './login-widget.css';
import { useSelector } from '@xstate/react';

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
        const response = await fetch('http://localhost:8090/health', {
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

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8090/auth/login', {
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
      setPassword('');
      setUsername('');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLogout = () => {
    libraryService.send({ type: 'SET_AUTH_TOKEN', token: null });
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
