import React from 'react';
import type { IntentId } from './intents';

// Stroke icons (24x24) for the Transform intents. Kept inline so the studio
// has no asset dependencies in either the Electron or web build.
const P: Record<IntentId, React.ReactNode> = {
  restore: (
    <>
      <path d="M12 3l1.8 4.6L18.5 9l-4.7 1.4L12 15l-1.8-4.6L5.5 9l4.7-1.4z" />
      <path d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z" />
      <path d="M5 16l.6 1.4L7 18l-1.4.6L5 20l-.6-1.4L3 18l1.4-.6z" />
    </>
  ),
  upscale: (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4l-7 7" />
      <path d="M10 20H4v-6" />
      <path d="M4 20l7-7" />
    </>
  ),
  wallpaper: (
    <>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8 20h8M12 16v4" />
      <path d="M3 13l4-4 4 4 3-3 7 6" />
    </>
  ),
  edit: (
    <>
      <path d="M4 20l1-4L16.5 4.5a2 2 0 013 3L8 19z" />
      <path d="M14 7l3 3" />
      <path d="M4 20h5" />
    </>
  ),
  combine: (
    <>
      <rect x="3" y="7" width="12" height="12" rx="2" />
      <path d="M9 7V5a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2h-2" />
    </>
  ),
  alive: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M10 8.5l5.5 3.5-5.5 3.5z" />
    </>
  ),
  direct: (
    <>
      <path d="M4 8h16v11a1 1 0 01-1 1H5a1 1 0 01-1-1z" />
      <path d="M4 8l2.5-4 3 1.2L12 3l3 1.2L17.5 3 20 8" />
    </>
  ),
};

export function IntentIcon({ id, size = 22 }: { id: IntentId; size?: number }) {
  return (
    <svg
      className="ts-icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {P[id]}
    </svg>
  );
}

export function GlyphDie({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="4" y="4" width="16" height="16" rx="3" />
      <circle cx="9" cy="9" r="1" fill="currentColor" />
      <circle cx="15" cy="15" r="1" fill="currentColor" />
      <circle cx="15" cy="9" r="1" fill="currentColor" />
      <circle cx="9" cy="15" r="1" fill="currentColor" />
    </svg>
  );
}

export function GlyphMusic({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 18V6l10-2v12" />
      <circle cx="6.5" cy="18" r="2.5" />
      <circle cx="16.5" cy="16" r="2.5" />
    </svg>
  );
}

export function GlyphCheck({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12.5l4.5 4.5L19 7.5" />
    </svg>
  );
}
