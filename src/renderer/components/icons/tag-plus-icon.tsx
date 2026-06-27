import * as React from 'react';

// A tag with a "+" badge — used to apply a tag to the current media item
// (as opposed to adding it to the search query).
export default function TagPlusIcon(
  props: React.SVGProps<SVGSVGElement>
): JSX.Element {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {/* Tag body (lower-left) */}
      <path d="M2 10 10 2h4a2 2 0 0 1 2 2v4l-8 8a2 2 0 0 1-2.83 0L2 12.83A2 2 0 0 1 2 10z" />
      {/* Tag hole */}
      <circle cx="6.4" cy="6.4" r="1.1" fill="currentColor" stroke="none" />
      {/* Plus badge (upper-right) */}
      <path d="M19 13v6M16 16h6" />
    </svg>
  );
}
