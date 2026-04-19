# Lowkey Media Viewer 2.8.0

Welcome to 2.8.0 — a big release packed with new ways to open, curate, and automate your media library.

## What's New

### Open Comic Book Archives

`.cbz` and `.zip` archives now open like directories. Pick one from the file dialog, drag one onto the window, or double-click it in Explorer — the app extracts the contents and loads every page into the viewer automatically. Reopening the same archive is instant.

### Drag-and-Drop Import

Drop files straight onto the viewer to add them to your library. A visual drop zone appears when you drag files over the window, and dropped files are copied into your current folder. Hold **Shift** while dropping to move instead of copy.

### Context Palette

**Shift + right-click** on any media item to open a new context palette. Unlike the full command palette, this one shows only the actions relevant to what you clicked — run a workflow, tag, delete, copy, and more — without leaving the viewer. Jobs started from the palette stay visible in a footer so you can watch progress or cancel them.

### Tag & Category Descriptions

Tags and categories now support descriptions. Redesigned edit modals let you add notes about what a tag is for, what to include, or anything else worth remembering. Descriptions show up inline in the tagging panel.

### Better Tag Consolidation

Consolidating files for a tag or category now creates a properly named folder per tag, with subfolders for categories. Your filesystem mirrors your taxonomy.

### ELO Ranking Fixes

Battle mode's ELO rankings now sort in the correct direction, deduplicate results, and refresh the library immediately after you re-apply an ordering. Large collections behave much better.

## Media Server Features

If you run the companion Lowkey Media Server, 2.8.0 brings a lot of automation improvements.

### Visual Workflows

Build multi-step workflows in the Drawflow editor by connecting tasks together, then save them by name. Saved workflows appear as one-click actions in the viewer's context palette. Tasks in a workflow automatically chain their outputs together — the file one task produces becomes the input for the next.

### FFmpeg Presets

The single "ffmpeg" task has been split into dedicated preset tasks (convert to mp4, extract audio, and more). Each preset has its own options and picks sensible defaults so you can run common conversions without writing flags.

### Structured Task Options

Task editors now render the right UI widget for each option automatically — dropdowns, toggles, number fields — based on metadata the task declares. No more free-form JSON.

### Save File Task

A new Save File task copies a workflow's output to a destination of your choice, handling filename conflicts safely. Pair it with conversion tasks to build "convert and save" workflows end-to-end.

### Default Storage Root

The server config UI now has a toggle for a default storage root, replacing the older "Download Path" setting. Useful if you have multiple roots configured and want one to be the default target for new files.

## Fixes

- Import flow: atomic copy, correct category lookup when tagging imports, toast feedback on zero-count drops.
- Drag-and-drop: absolute paths in default destinations, forward slashes for S3 compatibility, trailing separators on Windows drive roots, skip `mkdir` when the folder already exists.
- Web mode: uploads now land in the folder you're currently browsing.
- Library refresh: uses `REFRESH_LIBRARY` to preserve cursor position on drop.
- Many smaller polish items across modals, config UI, and the job runner.

## Download

Grab the installer for your platform from the [Releases page](https://github.com/SteveCastle/loki/releases).

— The Lowkey team
