// Sign the viewer in to the media server through the browser, the same way
// `lokictl login` does (RFC 8252 loopback flow with PKCE, served by the
// server's /auth/cli/authorize + /auth/cli/token): listen on an ephemeral
// 127.0.0.1 port, open the server's approve page in the user's browser (it
// bounces through /login when there is no session), receive a one-time code
// on the loopback callback and exchange it, with the verifier, for an lk_ API
// key. The key is what the renderer keeps as its bearer token.
import { ipcMain, net, shell } from 'electron';
import crypto from 'crypto';
import fs from 'fs';
import http from 'http';
import os from 'os';
import path from 'path';
import type { AddressInfo } from 'net';
import { detectMediaServerBase, mediaServerConfigPath } from './media-server-base';

const LOGIN_TIMEOUT_MS = 3 * 60_000;

const b64url = (buf: Buffer) => buf.toString('base64url');

const callbackPage = (title: string, text: string) => `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Lowkey Media Viewer</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#101216;color:#d7dae0;font:15px/1.5 "Segoe UI",system-ui,sans-serif}
div{text-align:center}p{color:#9aa0ab}</style></head>
<body><div><h1>${title}</h1><p>${text}</p></div></body></html>`;

export interface BrowserLoginResult {
  token?: string;
  username?: string;
  error?: string;
}

let cancelInFlight: (() => void) | null = null;

/** Waits on the loopback listener for the browser's callback; resolves with the code. */
function waitForCode(server: http.Server, state: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => done(new Error('Timed out waiting for the browser')), LOGIN_TIMEOUT_MS);
    let settled = false;
    const done = (err: Error | null, code?: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      cancelInFlight = null;
      if (err) reject(err);
      else resolve(code as string);
    };
    cancelInFlight = () => done(new Error('Sign-in cancelled'));
    server.on('request', (req, res) => {
      const u = new URL(req.url || '/', 'http://127.0.0.1');
      if (u.pathname !== '/callback') {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      const q = u.searchParams;
      if (q.get('state') !== state) {
        res.end(callbackPage('Sign-in failed', 'State mismatch. Go back to the viewer and try again.'));
        done(new Error('State mismatch in the browser callback'));
      } else if (q.get('error')) {
        res.end(callbackPage('Sign-in cancelled', 'You can close this tab.'));
        done(new Error('Sign-in was denied in the browser'));
      } else if (!q.get('code')) {
        res.end(callbackPage('Sign-in failed', 'No authorization code was returned.'));
        done(new Error('The browser callback had no code'));
      } else {
        res.end(callbackPage('✓ Lowkey Media Viewer is signed in', 'You can close this tab and return to the viewer.'));
        done(null, q.get('code') as string);
      }
    });
  });
}

async function browserLogin(): Promise<BrowserLoginResult> {
  cancelInFlight?.();
  const base = detectMediaServerBase();
  const state = b64url(crypto.randomBytes(16));
  const verifier = b64url(crypto.randomBytes(32));
  const challenge = b64url(crypto.createHash('sha256').update(verifier).digest());

  const server = http.createServer();
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });
  try {
    const { port } = server.address() as AddressInfo;
    const keyName = `Lowkey Media Viewer@${os.hostname()}`.slice(0, 64);
    const authURL =
      `${base}/auth/cli/authorize?port=${port}&state=${encodeURIComponent(state)}` +
      `&code_challenge=${encodeURIComponent(challenge)}&name=${encodeURIComponent(keyName)}`;
    const codeP = waitForCode(server, state);
    await shell.openExternal(authURL);
    const code = await codeP;

    const res = await net.fetch(`${base}/auth/cli/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, code_verifier: verifier }),
    });
    const data = (await res.json().catch(() => ({}))) as { key?: string; username?: string; error?: string };
    if (!res.ok || !data.key) return { error: data.error || `Sign-in failed (HTTP ${res.status})` };
    return { token: data.key, username: data.username };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  } finally {
    // Let the success page finish writing before the listener goes away.
    setTimeout(() => server.close(), 2000);
  }
}

export function registerServerLoginHandlers(): void {
  ipcMain.handle('server-browser-login', () => browserLogin());
  ipcMain.handle('server-browser-login-cancel', () => cancelInFlight?.());
  // The folder holding config.json (shared by the server and the viewer).
  ipcMain.handle('open-config-dir', () => {
    const dir = path.dirname(mediaServerConfigPath());
    fs.mkdirSync(dir, { recursive: true });
    return shell.openPath(dir);
  });
}
