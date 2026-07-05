// Maps job-creation surfaces (task chips, per-file generate buttons) to the
// dependency each one needs, so the UI can offer a one-click download at the
// point of use instead of letting the job fail later in its log.
import { mediaServerBase } from '../platform';

// The deps API lives on the media server. In web mode the SPA is served by
// that same server, so relative URLs work; in Electron the SPA must reach the
// local server directly (same base the job endpoints already use).
export const depsApiBase = mediaServerBase;

export interface TaskRequirement {
  /** id in /api/deps/status */
  depId: string;
  /** What the user gets, e.g. "Auto-tagging" */
  feature: string;
  /**
   * downloadable — server can install it via /api/deps/models/{id}/download.
   * external — user installs it themselves (e.g. Ollama); never blocks the
   * action because other providers may be configured.
   */
  kind: 'downloadable' | 'external';
}

/** Keyed by the Generate chip labels in the context palette. */
export const TASK_REQUIREMENTS: Record<string, TaskRequirement> = {
  Tags: { depId: 'wd-eva02-large-tagger-v3', feature: 'Auto-tagging', kind: 'downloadable' },
  Descriptions: { depId: 'ollama', feature: 'AI descriptions', kind: 'external' },
  Transcripts: { depId: 'faster-whisper', feature: 'Transcription', kind: 'downloadable' },
  Embeddings: { depId: 'siglip2-base-patch16-224', feature: 'Visual similarity search', kind: 'downloadable' },
  // Face scanning also needs the (tiny) YuNet detector; SFace is the big
  // download and the default recognizer, so it's the one gated here. A
  // configured bring-your-own recognizer still requires YuNet, which the
  // server reports politely in the job log if missing.
  Faces: { depId: 'sface', feature: 'Face recognition', kind: 'downloadable' },
};

export function fmtSize(n?: number): string {
  if (!n) return '';
  const mb = n / 1024 / 1024;
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}
