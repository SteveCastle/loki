import { useContext, useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { useDepRequirement } from '../../onboarding/useDepRequirement';
import { fmtSize, depsApiBase } from '../../onboarding/requirements';
import { SparkleIcon } from './section-action-icons';
import './embeddings.css';

// The visual-embedding card in the metadata tab: shows which embedding models
// have a stored vector for this file, and lets the user generate (or delete +
// regenerate) the vector for any supported model. Talks to the media server:
//   GET    /api/index/models       — supported model registry
//   GET    /api/embeddings?path=   — stored rows for this file
//   DELETE /api/embeddings?path=&model=
//   POST   /create                 — `embed --model=<id> "<path>"` job

type EmbeddingRow = { model: string; dim: number; created_at: number };
type EmbedModelInfo = {
  id: string;
  display_name: string;
  dim: number;
  active: boolean;
  multimodal: boolean;
};

// How long a "Generating…" state may run before we assume the job failed and
// re-enable the button (the job's real error surfaces in the jobs toast list).
const PENDING_TIMEOUT_MS = 5 * 60 * 1000;

const authHeaders = (token?: string | null): HeadersInit =>
  token ? { Authorization: `Bearer ${token}` } : {};

async function getJSON<T>(url: string, token?: string | null): Promise<T> {
  const res = await fetch(url, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function ModelRow({
  model,
  row,
  pending,
  canRun,
  onGenerate,
}: {
  model: EmbedModelInfo;
  row?: EmbeddingRow;
  pending: boolean;
  canRun: boolean;
  onGenerate: (modelId: string, regenerate: boolean) => void;
}) {
  // Each embed model is also a downloadable dependency under the same id, so
  // offer the one-time download in place when it isn't installed yet.
  const dep = useDepRequirement(model.id);

  let action = null;
  if (pending) {
    action = (
      <button className="embed-action" disabled>
        <span className="embed-spinner" />
        <span>Generating…</span>
      </button>
    );
  } else if (dep.needsDownload) {
    action = (
      <button
        className="embed-action"
        onClick={() => {
          dep.download().catch((err) => console.error('model download:', err));
        }}
        title={`One-time model download (${fmtSize(dep.dep?.size_bytes)}); embedding runs locally`}
      >
        <SparkleIcon />
        <span>Get model ({fmtSize(dep.dep?.size_bytes)})</span>
      </button>
    );
  } else if (dep.downloading) {
    action = (
      <button className="embed-action" disabled>
        <span className="embed-spinner" />
        <span>Downloading… {dep.pct}%</span>
      </button>
    );
  } else if (canRun) {
    action = (
      <button
        className="embed-action"
        onClick={() => onGenerate(model.id, !!row)}
        title={
          row
            ? 'Delete the stored vector and re-embed this file'
            : 'Embed this file with this model'
        }
      >
        <SparkleIcon />
        <span>{row ? 'Regenerate' : 'Generate'}</span>
      </button>
    );
  }

  return (
    <div className="embed-row">
      <div className="embed-info">
        <span className="embed-name">
          {model.display_name}
          {model.active && <span className="embed-badge">active</span>}
        </span>
        <span className={`embed-status ${row ? 'stored' : ''}`}>
          {row ? `${row.dim}-dim vector stored` : 'not generated'}
        </span>
      </div>
      {action}
    </div>
  );
}

export default function Embeddings({ path }: { path: string }) {
  const { libraryService } = useContext(GlobalStateContext);
  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  // Generate/regenerate/delete-vector controls — whole card hidden for
  // view-only public visitors (guard below the hooks).
  const canWrite = useSelector(
    libraryService,
    (state) => state.context.canWrite
  );
  const queryClient = useQueryClient();
  // model id -> Date.now() when its embed job was created
  const [pending, setPending] = useState<Record<string, number>>({});

  // enabled: canWrite — the card renders null for view-only visitors, but
  // hooks still run, and these endpoints are admin-only; don't spray
  // doomed requests.
  const modelsQuery = useQuery<{ models: EmbedModelInfo[] }, Error>(
    ['embed-models'],
    () => getJSON(`${depsApiBase}/api/index/models`, authToken),
    { staleTime: 60_000, retry: false, enabled: canWrite }
  );

  const pendingCount = Object.keys(pending).length;
  const embedsQuery = useQuery<{ embeddings: EmbeddingRow[] }, Error>(
    ['embeddings', path],
    () =>
      getJSON(
        `${depsApiBase}/api/embeddings?path=${encodeURIComponent(path)}`,
        authToken
      ),
    {
      retry: false,
      refetchInterval: pendingCount > 0 ? 3000 : false,
      enabled: canWrite,
    }
  );

  const rows = embedsQuery.data?.embeddings ?? [];
  const byModel = new Map(rows.map((r) => [r.model, r]));

  // A pending model whose vector (re)appeared is done; regenerate deletes the
  // old row before queueing, so presence always means the fresh vector landed.
  // Stale pendings (failed jobs) unlock after a timeout.
  useEffect(() => {
    const now = Date.now();
    const done = Object.keys(pending).filter(
      (m) => byModel.has(m) || now - pending[m] > PENDING_TIMEOUT_MS
    );
    if (done.length > 0) {
      setPending((prev) => {
        const next = { ...prev };
        for (const m of done) delete next[m];
        return next;
      });
    }
    // Intentionally keyed on the fetched data only: `pending`/`byModel` are
    // derived inside and adding them would re-run the effect on its own set.
  }, [embedsQuery.data]);

  // Reset in-flight state when the user moves to a different file.
  useEffect(() => setPending({}), [path]);

  const handleGenerate = async (modelId: string, regenerate: boolean) => {
    try {
      if (regenerate) {
        // The embed task skips already-embedded paths, so drop the stored
        // vector first (this also removes it from the live search index).
        const res = await fetch(
          `${depsApiBase}/api/embeddings?path=${encodeURIComponent(path)}&model=${encodeURIComponent(modelId)}`,
          { method: 'DELETE', headers: authHeaders(authToken) }
        );
        if (!res.ok) throw new Error(`delete failed (HTTP ${res.status})`);
        await queryClient.invalidateQueries({ queryKey: ['embeddings', path] });
      }
      const res = await fetch(`${depsApiBase}/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(authToken) },
        body: JSON.stringify({ input: `embed --model=${modelId} "${path}"` }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPending((prev) => ({ ...prev, [modelId]: Date.now() }));
    } catch (error) {
      console.error('Failed to create embed job:', error);
      libraryService.send({
        type: 'ADD_TOAST',
        data: {
          type: 'error',
          title: 'Failed to Create Job',
          message: 'Could not communicate with job service',
        },
      });
    }
  };

  // If the media server (or its embeddings API) is unreachable — e.g. the
  // Electron app running without the local server — render no card at all
  // rather than one that can't work. The component owns its whole section so
  // nothing (not even the header) is left behind in that case.
  if (!canWrite) return null;
  if (modelsQuery.isError || embedsQuery.isError) return null;

  const models = modelsQuery.data?.models ?? [];
  // Vectors stored under models no longer in the registry (older experiments)
  // are still shown so it's clear they exist; they just can't be regenerated.
  const legacy = rows.filter((r) => !models.find((m) => m.id === r.model));

  if (models.length === 0 && legacy.length === 0) return null;

  return (
    <div className="section">
      <h2>Embeddings</h2>
      <div className="Embeddings">
        {models.map((m) => (
          <ModelRow
            key={m.id}
            model={m}
            row={byModel.get(m.id)}
            pending={m.id in pending}
            canRun={!embedsQuery.isLoading}
            onGenerate={handleGenerate}
          />
        ))}
        {legacy.map((r) => (
          <div className="embed-row" key={r.model}>
            <div className="embed-info">
              <span className="embed-name">
                {r.model}
                <span className="embed-badge legacy">legacy</span>
              </span>
              <span className="embed-status stored">
                {r.dim}-dim vector stored
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
