/*
 * Lowkey Studio — talking to a Lowkey Media Server.
 *
 * Studio runs entirely in the tab; the one thing it hands off is work a
 * browser can't do, such as training a Gaussian splat from a video (COLMAP
 * + Brush on the server's GPU, the `splat` task). This module is the whole
 * client: connection settings, authenticated fetches, uploads with progress,
 * and job polling. UI-free; app.js owns the dialog.
 *
 * Auth is an API key in a header (`Authorization: Bearer lk_…`), created on
 * the server's Config page. The server answers Studio's origins (the docs
 * site, a local dev server, the viewer's studio:// window) with
 * header-credential CORS only — no cookies ever cross.
 */

const SERVER_KEY = 'lowkey-studio.server';

/** The viewer can hand its server over: studio://app/index.html?server=… */
const launchServer = new URLSearchParams(location.search).get('server');

export function getServer() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SERVER_KEY)) ?? {}; } catch {}
  return {
    url: (saved.url || launchServer || 'http://localhost:10111').replace(/\/+$/, ''),
    key: saved.key || '',
  };
}

export function setServer({ url, key }) {
  try {
    localStorage.setItem(SERVER_KEY, JSON.stringify({
      url: String(url || '').trim().replace(/\/+$/, ''),
      key: String(key || '').trim(),
    }));
  } catch {}
}

export const serverConfigured = () => !!getServer().key;

class ServerError extends Error {
  constructor(message, status = 0) { super(message); this.status = status; }
}

function headers(extra = {}) {
  const { key } = getServer();
  return key ? { Authorization: `Bearer ${key}`, Accept: 'application/json', ...extra } : { Accept: 'application/json', ...extra };
}

async function call(path, { method = 'GET', body = null, json = true, timeout = 15000 } = {}) {
  const { url } = getServer();
  let res;
  try {
    res = await fetch(url + path, {
      method,
      headers: headers(body && typeof body === 'string' ? { 'Content-Type': 'application/json' } : {}),
      body,
      credentials: 'omit',
      redirect: 'error',
      signal: AbortSignal.timeout(timeout),
    });
  } catch (e) {
    throw new ServerError(`can't reach ${url} — is the media server running?`);
  }
  if (res.status === 401) throw new ServerError('the server rejected the API key', 401);
  if (res.status === 403) throw new ServerError('the server refused (create a user on its setup page first?)', 403);
  if (!res.ok) throw new ServerError(`${method} ${path}: HTTP ${res.status} ${(await res.text().catch(() => '')).slice(0, 160)}`, res.status);
  return json ? res.json() : res;
}

/** Cheap authenticated probe: resolves when url + key both work. */
export async function testConnection() {
  await call('/api/jobs/for-path?path=studio-connection-test');
  return true;
}

/** Upload files (no library ingest); resolves to their server paths. */
export function uploadFiles(files, onProgress = () => {}) {
  const { url } = getServer();
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    form.append('autoIngest', 'false');
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${url}/api/upload`);
    for (const [k, v] of Object.entries(headers())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onerror = () => reject(new ServerError(`upload to ${url} failed — is the media server running?`));
    xhr.onload = () => {
      if (xhr.status === 401) return reject(new ServerError('the server rejected the API key', 401));
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 300 || !body?.success) {
        return reject(new ServerError(body?.error || `upload failed: HTTP ${xhr.status}`, xhr.status));
      }
      resolve(body.files);
    };
    xhr.send(form);
  });
}

/** Queue one task; `paths` become its input list. Resolves to the job id. */
export async function createJob(command, paths, fields = {}) {
  const input = `${command} "${paths.join('\n')}"`;
  const strFields = Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, String(v)]));
  const body = await call('/create', { method: 'POST', body: JSON.stringify({ input, fields: strFields }) });
  if (!body?.id) throw new ServerError('the server queued the job but returned no id');
  return body.id;
}

/** {state, progress_done, progress_total, output_files, log} */
export const jobStatus = (id) => call(`/api/job/${encodeURIComponent(id)}?tail=6`);

export const cancelJob = (id) => call(`/job/${encodeURIComponent(id)}/cancel`, { method: 'POST', json: false });

/** Download a file the server produced. */
export async function fetchServerFile(path) {
  const res = await call(`/media/file?path=${encodeURIComponent(path)}`, { json: false, timeout: 600000 });
  return res.blob();
}
