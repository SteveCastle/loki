# Shrike Chrome Extension

A Chrome extension for creating Shrike tasks directly from your browser. Quickly send the current page URL to your Shrike job server with custom commands and arguments.

> **Firefox User?** See the [Firefox Extension](../firefox-extension/README.md) for Firefox-specific installation instructions.

## Features

- **One-Click Task Creation**: Create tasks with the current page URL pre-filled
- **Command Selection**: Choose from all available Shrike commands (gallery-dl, yt-dlp, ffmpeg, ingest, etc.)
- **Custom Arguments**: Add optional arguments for fine-grained control
- **Live Job Status**: Real-time updates on running jobs via Server-Sent Events (SSE)
- **Persistent Preferences**: Your last used command and arguments are saved

## Prerequisites

- Google Chrome or Chromium-based browser (Edge, Brave, etc.)
- Shrike server running on `http://localhost:8090`

## Installation

### Option 1: Load as Unpacked Extension (Development Mode)

1. **Generate Icons** (if not already present):

   ```powershell
   # Navigate to the chrome-extension directory
   cd chrome-extension

   # Run the icon generator script
   node generate-icons.js
   ```

   Or manually create icon files (see [Creating Icons](#creating-icons) below).

2. **Open Chrome Extensions Page**:

   - Navigate to `chrome://extensions/` in your browser
   - Or go to Menu → More Tools → Extensions

3. **Enable Developer Mode**:

   - Toggle the "Developer mode" switch in the top-right corner

4. **Load the Extension**:

   - Click "Load unpacked"
   - Select the `chrome-extension` folder from this repository

5. **Pin the Extension** (optional but recommended):
   - Click the puzzle piece icon in the Chrome toolbar
   - Find "Shrike" and click the pin icon

### Option 2: Pack and Install

1. Generate icons and go to `chrome://extensions/`
2. Enable Developer mode
3. Click "Pack extension"
4. Select the `chrome-extension` folder
5. This creates a `.crx` file you can distribute

## Usage

### Creating a Task

1. Navigate to any web page you want to process
2. Click the Shrike extension icon in your toolbar
3. Select a command from the dropdown:
   - **gallery-dl**: Download images/galleries from websites
   - **yt-dlp**: Download videos from YouTube, Vimeo, etc.
   - **ffmpeg**: Process media files
   - **ingest**: Add media to the Shrike database
   - **metadata**: Generate descriptions, hashes, etc.
   - And more...
4. Add optional arguments (e.g., `--format best` for yt-dlp)
5. The current page URL is auto-filled, but you can modify it
6. Click "Create Task"

### Monitoring Jobs

- The extension shows a real-time list of active and recent jobs
- Status indicators:
  - 🔘 Gray: Pending
  - 🔵 Blue (pulsing): Running
  - 🟢 Green: Completed
  - 🟡 Yellow: Cancelled
  - 🔴 Red: Error
- Click the refresh button to reconnect if disconnected
- Click "Open Shrike Web UI →" for the full job management interface

### Examples

**Download a YouTube video:**

```
Command: yt-dlp
Args: --format bestvideo+bestaudio
URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

**Download a gallery:**

```
Command: gallery-dl
Args: --dest C:\Downloads\Gallery
URL: https://example.com/gallery/123
```

**Ingest a folder:**

```
Command: ingest
Args: -r
URL: C:\Media\NewFiles
```

## Creating Icons

The extension requires PNG icons in multiple sizes. You can create them using any of these methods:

### Method 1: Use the Generator Script

```bash
cd chrome-extension
node generate-icons.js
```

This creates placeholder icons with the Shrike logo.

### Method 2: Manual Creation

Create the following PNG files in the `icons/` folder:

- `icon16.png` (16x16 pixels)
- `icon32.png` (32x32 pixels)
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

You can use any image editor or convert the project's `assets/logo.ico` file.

### Method 3: Use Online Converter

1. Take the `assets/logo.ico` from the main Shrike project
2. Use an online ICO to PNG converter
3. Resize to required dimensions
4. Save in the `icons/` folder

## Configuration

The extension connects to `http://localhost:8090` by default. If you need to change this:

1. Open `popup.js`
2. Modify the `API_BASE` constant at the top of the file
3. Update `host_permissions` in `manifest.json` accordingly
4. Reload the extension

## Troubleshooting

### Extension shows "Disconnected"

- Ensure the Shrike server is running (`shrike.exe`)
- Check that port 8090 is accessible
- Try clicking the refresh button

### "Failed to create task" error

- Verify the Shrike server is running
- Check the browser console for detailed error messages
- Ensure the URL/input is valid for the selected command

### Jobs not updating in real-time

- The extension uses SSE for live updates
- If disconnected, click the refresh button
- Check that no firewall is blocking the connection

### Icons not showing

- Ensure PNG files exist in the `icons/` folder
- Run the icon generator script or create icons manually
- Reload the extension after adding icons

## Development

### File Structure

```
chrome-extension/
├── manifest.json      # Extension configuration
├── popup.html         # Extension popup UI
├── popup.css          # Styles
├── popup.js           # Main functionality
├── generate-icons.js  # Icon generator script
├── icons/
│   ├── icon16.png
│   ├── icon32.png
│   ├── icon48.png
│   └── icon128.png
└── README.md
```

### Making Changes

1. Edit the source files as needed
2. Go to `chrome://extensions/`
3. Click the reload button on the Shrike extension
4. Test your changes

### API Reference

The extension uses these Shrike API endpoints:

- `POST /create` - Create a new job
- `GET /stream` - SSE for job updates
- `GET /health` - Server health check
- `POST /job/{id}/cancel` - Cancel a job

See the main Shrike `API_DOCUMENTATION.md` for full details.

## License

This extension is part of the Shrike project. See the main repository LICENSE file.
