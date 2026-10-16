import React, { useMemo, useState } from 'react';
import { parseGenerationMetadata } from '../../generation-metadata';
import copyIcon from '../../../../assets/copy.svg';

import './generation-info.css';

/* The panel's "Generation" section: what the AI tool that made this file
 * wrote into it. The prompt is the thing people read; the workflow JSON is
 * the thing they take with them (drop it onto ComfyUI to rebuild the
 * graph), so it gets a one-click copy rather than a wall of text. */

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy: ', err);
    return false;
  }
}

function CopyButton({ text, label, title }: { text: string; label?: string; title: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className={'gen-copy' + (label ? ' labelled' : '')}
      title={title}
      onClick={async (e) => {
        e.stopPropagation();
        if (await copyText(text)) {
          setDone(true);
          setTimeout(() => setDone(false), 1200);
        }
      }}
    >
      <img src={copyIcon} alt="" />
      {label && <span>{done ? 'Copied' : label}</span>}
    </button>
  );
}

const sizeOf = (text: string) => {
  const bytes = new Blob([text]).size;
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

export default function GenerationInfo({ chunks }: { chunks?: Record<string, string> | null }) {
  const info = useMemo(() => parseGenerationMetadata(chunks), [chunks]);
  const [showRaw, setShowRaw] = useState(false);
  if (!info) return null;

  const copyables: Array<{ key: string; label: string; title: string; text: string }> = [];
  if (info.workflowJson)
    copyables.push({ key: 'workflow', label: 'Copy workflow JSON', title: `the full ComfyUI workflow (${sizeOf(info.workflowJson)}) — paste or drop it into ComfyUI to rebuild the graph`, text: info.workflowJson });
  if (info.promptJson)
    copyables.push({ key: 'prompt', label: 'Copy API prompt JSON', title: `the ComfyUI API-format prompt (${sizeOf(info.promptJson)}) — what the server actually ran`, text: info.promptJson });
  if (info.parametersText)
    copyables.push({ key: 'parameters', label: 'Copy parameters', title: 'the whole WebUI parameters block — paste it into the PNG Info tab', text: info.parametersText });

  const rawEntries = Object.entries(info.raw);

  return (
    <div className="section GenerationInfo">
      <h2>Generation · {info.generator}</h2>

      {info.prompt !== undefined && (
        <div className="gen-block">
          <div className="gen-label">
            Prompt
            <CopyButton text={info.prompt} title="copy the prompt" />
          </div>
          <div className="gen-text">{info.prompt || <span className="gen-empty">(empty)</span>}</div>
        </div>
      )}
      {info.negativePrompt && (
        <div className="gen-block">
          <div className="gen-label">
            Negative prompt
            <CopyButton text={info.negativePrompt} title="copy the negative prompt" />
          </div>
          <div className="gen-text">{info.negativePrompt}</div>
        </div>
      )}

      {info.params.length > 0 && (
        <div className="gen-params">
          {info.params.map(([label, value]) => (
            <div className="gen-param" key={label} title={`${label}: ${value}`}>
              <span className="gen-param-key">{label}</span>
              <span className="gen-param-val">{value}</span>
            </div>
          ))}
        </div>
      )}

      {copyables.length > 0 && (
        <div className="gen-actions">
          {copyables.map((c) => (
            <CopyButton key={c.key} text={c.text} label={c.label} title={c.title} />
          ))}
        </div>
      )}

      {rawEntries.length > 0 && (
        <div className="gen-raw">
          <button type="button" className="gen-raw-toggle" onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? '▾' : '▸'} embedded text ({rawEntries.length})
          </button>
          {showRaw &&
            rawEntries.map(([key, text]) => (
              <div className="gen-block" key={key}>
                <div className="gen-label">
                  {key} <span className="gen-dim">{sizeOf(text)}</span>
                  <CopyButton text={text} title={`copy the raw "${key}" text`} />
                </div>
                <div className="gen-text gen-mono">
                  {text.length > 2000 ? `${text.slice(0, 2000)}…` : text}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
