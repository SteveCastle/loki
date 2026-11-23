/**
 * Shared vector icons for the section-action pills (Generate / Regenerate and
 * the Customize-prompt toggle). Kept as inline SVGs sized in `em` so they scale
 * with the button font-size and stay crisp at any zoom. They inherit color via
 * `currentColor`, so hover/active states need no extra wiring.
 */

export function SparkleIcon() {
  return (
    <svg
      className="section-action-icon"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 2.25l1.85 5.4a3 3 0 0 0 1.88 1.88L21 11.38l-5.27 1.85a3 3 0 0 0-1.88 1.88L12 20.5l-1.85-5.39a3 3 0 0 0-1.88-1.88L3 11.38l5.27-1.85a3 3 0 0 0 1.88-1.88L12 2.25z" />
      <path
        d="M18.75 14.5l.78 2.22a1.5 1.5 0 0 0 .95.95l2.27.78-2.27.78a1.5 1.5 0 0 0-.95.95l-.78 2.22-.78-2.22a1.5 1.5 0 0 0-.95-.95L14.75 18.45l2.27-.78a1.5 1.5 0 0 0 .95-.95l.78-2.22z"
        opacity="0.6"
      />
    </svg>
  );
}

export function TuneIcon() {
  return (
    <svg
      className="section-action-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden="true"
      focusable="false"
    >
      <line x1="4" y1="8" x2="20" y2="8" />
      <line x1="4" y1="16" x2="20" y2="16" />
      <circle cx="9" cy="8" r="2.6" fill="currentColor" stroke="none" />
      <circle cx="15" cy="16" r="2.6" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function CopyIcon() {
  return (
    <svg
      className="section-action-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

export function CheckIcon() {
  return (
    <svg
      className="section-action-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <polyline points="4 12.5 9.5 18 20 6" />
    </svg>
  );
}
