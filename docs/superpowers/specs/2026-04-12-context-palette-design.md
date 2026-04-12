# Contextual Command Palette

## Overview

A shift+right-click contextual palette for bulk operations on the current library selection. Styled like the existing command palette but focused on server task actions (transcripts, auto-tagging, descriptions). Each action creates a single server task with a query matching the active library.

## Trigger Mechanism

Shift+Right-Click on panels, detail view, or list items sends a new `SHOW_CONTEXT_PALETTE` event with click position. Regular right-click continues to send `SHOW_COMMAND_PALETTE`.

The existing right-click handlers in `panels.tsx`, `detail.tsx`, and `list-item.tsx` check `e.shiftKey` to decide which event to fire.

### State Machine Addition

New parallel context field in `LibraryState`:

```typescript
contextPalette: {
  display: boolean;
  position: { x: number; y: number };
}
```

New events:
- `SHOW_CONTEXT_PALETTE` — sets `display: true` and position, hides command palette
- `HIDE_CONTEXT_PALETTE` — sets `display: false`

The context palette and command palette are mutually exclusive — opening one closes the other.

## Query Construction

A utility function `buildQueryFromState(context)` constructs a query string from the current renderer state:

- **DB mode** (`currentStateType === 'db'`): Builds from `dbQuery.tags` — e.g. `tag:landscape AND tag:outdoor` (EXCLUSIVE) or `tag:landscape OR tag:outdoor` (INCLUSIVE). Uses the `filteringMode` setting.
- **Search mode** (`currentStateType === 'search'`): Builds from `textFilter` combined with `dbQuery.tags` if present — e.g. `description:sunset AND tag:landscape`.
- **FS mode** (`currentStateType === 'fs'`): Builds from the loaded directory path — e.g. `pathdir:C:/Users/Pictures`.

The query string is base64-encoded and passed as `--query64=<encoded>` in the task input sent to the server.

The item count in the palette header comes from `filter()` applied to the current library state (same pipeline as the list view), so the count matches what the user sees on screen.

## Actions

Six actions — generate and regenerate variants for each task type:

| Action | Server Command |
|--------|---------------|
| Generate Transcripts | `metadata --type transcript --apply all --query64=<q>` |
| Regenerate Transcripts | `metadata --type transcript --apply all --overwrite --query64=<q>` |
| Generate Tags | `autotag --query64=<q>` |
| Regenerate Tags | `autotag --overwrite --query64=<q>` |
| Generate Descriptions | `metadata --type description --apply all --query64=<q>` |
| Regenerate Descriptions | `metadata --type description --apply all --overwrite --query64=<q>` |

### Task Creation

Each action sends a POST to `http://localhost:8090/create`:

```json
{
  "input": "<command string with --query64>"
}
```

Includes `Authorization: Bearer <token>` header if `authToken` is available in state.

On success: palette closes, toast appears via existing toast system.
On failure: error toast via `libraryService.send('ADD_TOAST', ...)`.

## Component Structure

### New Files

- `src/renderer/components/controls/context-palette.tsx`
- `src/renderer/components/controls/context-palette.css`

### Layout

```
ContextPalette (fixed position at click coords)
├── Header
│   └── Context info: "{mode description} — {N} items"
│       Examples:
│       - "3 tags selected — 482 items"
│       - "Search: sunset — 127 items"
│       - "Directory: Pictures — 54 items"
├── Action List
│   ├── Transcripts section
│   │   ├── Generate Transcripts (row)
│   │   └── Regenerate Transcripts (row)
│   ├── Tags section
│   │   ├── Generate Tags (row)
│   │   └── Regenerate Tags (row)
│   └── Descriptions section
│       ├── Generate Descriptions (row)
│       └── Regenerate Descriptions (row)
└── Footer
    └── Active jobs list (from server)
```

### Positioning

Same approach as command palette:
- Fixed position at click coordinates
- Clamped to viewport with 8px margins
- Uses `useLayoutEffect` + `useComponentSize` for measurement
- Hidden off-screen initially, shown after measurement

### Styling

Matches existing command palette dark theme:
- Semi-transparent background (opacity 0.9)
- Teal accent color (#00c896)
- Same z-index layer (9999)
- Action rows with hover highlight
- Section headers to group action types

### Close Behavior

Closes on:
- Click outside (`useOnClickOutside`)
- Escape key
- Library change
- Opening the regular command palette

### Server Health

On open, checks `localhost:8090/health`:
- If unavailable: shows "Job Service Required" message in place of action list
- If available: shows actions normally

## Job Status Footer

When the palette is open:
1. Fetch current job list from server on open
2. Subscribe to SSE stream at `localhost:8090/stream` for real-time updates
3. Display running/pending jobs with command type and state (e.g. "Transcripts — running")
4. Unsubscribe from SSE when palette closes

The server is the single source of truth for job state. No job tracking in XState.

## Files Modified

- `src/renderer/state.tsx` — add `contextPalette` to context, `SHOW_CONTEXT_PALETTE` / `HIDE_CONTEXT_PALETTE` events
- `src/renderer/components/layout/panels.tsx` — shift+right-click handler
- `src/renderer/components/detail/detail.tsx` — shift+right-click handler
- `src/renderer/components/list/list-item.tsx` — shift+right-click handler
- `src/renderer/components/controls/context-palette.tsx` — new component
- `src/renderer/components/controls/context-palette.css` — new styles

## New Utility

`buildQueryFromState(context)` — pure function that takes the XState context and returns a query string. Could live in the context-palette module or a shared utility, depending on whether it's needed elsewhere.
