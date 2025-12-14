# Shrike Firefox Extension

A Firefox extension for creating Shrike tasks directly from your browser. Quickly send the current page URL to your Shrike job server with custom commands and arguments.

## Features

- **One-Click Task Creation**: Create tasks with the current page URL pre-filled
- **Command Selection**: Choose from all available Shrike commands (gallery-dl, yt-dlp, ffmpeg, ingest, etc.)
- **Custom Arguments**: Add optional arguments for fine-grained control
- **Live Job Status**: Real-time updates on running jobs via Server-Sent Events (SSE)
- **Persistent Preferences**: Your last used command and arguments are saved

## Prerequisites

- Firefox 91.0 or later
- Shrike server running on `http://localhost:8090`

## Installation

### Option 1: Temporary Installation (Development/Testing)

This method is ideal for testing or development. The extension will be removed when Firefox restarts.

1. **Open Firefox Add-ons Debugging Page**:

   - Navigate to `about:debugging#/runtime/this-firefox` in Firefox
   - Or go to Menu → Add-ons and themes → (gear icon) → Debug Add-ons

2. **Load the Extension**:

   - Click "Load Temporary Add-on..."
   - Navigate to the `firefox-extension` folder
   - Select the `manifest.json` file

3. **Pin the Extension** (optional):
   - Click the puzzle piece icon in the Firefox toolbar
   - Find "Shrike" and click the gear icon → "Pin to Toolbar"

### Option 2: Permanent Installation (Unsigned - Developer Mode)

For permanent installation without Mozilla signing, you need to use Firefox Developer Edition or Firefox Nightly:

1. **Use Firefox Developer Edition or Nightly**:

   - Download from [Firefox Developer Edition](https://www.mozilla.org/firefox/developer/) or [Firefox Nightly](https://www.mozilla.org/firefox/nightly/)

2. **Disable Signature Enforcement**:

   - Navigate to `about:config`
   - Search for `xpinstall.signatures.required`
   - Set it to `false`

3. **Package the Extension**:

   ```bash
   cd firefox-extension
   zip -r shrike-firefox.xpi *
   ```

4. **Install the XPI**:
   - Open `about:addons`
   - Click the gear icon → "Install Add-on From File..."
   - Select the `shrike-firefox.xpi` file

### Option 3: Install as Signed Add-on (Production)

For distribution or use in standard Firefox, you need to sign the extension:

1. Create a [Mozilla Add-ons Developer Account](https://addons.mozilla.org/developers/)
2. Submit the extension for signing (can be unlisted for personal use)
3. Download the signed XPI and install via `about:addons`

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
cd firefox-extension
node generate-icons.js
```

This creates placeholder icons with the Shrike logo.

### Method 2: Manual Creation

Create the following PNG files in the `icons/` folder:

- `icon16.png` (16x16 pixels)
- `icon48.png` (48x48 pixels)
- `icon128.png` (128x128 pixels)

You can use any image editor or convert the project's `assets/logo.ico` file.

## Configuration

The extension connects to `http://localhost:8090` by default. If you need to change this:

1. Open `popup.js`
2. Modify the `API_BASE` constant at the top of the file
3. Update the permissions in `manifest.json` accordingly
4. Reload the extension

## Troubleshooting

### Extension shows "Disconnected"

- Ensure the Shrike server is running (`shrike.exe`)
- Check that port 8090 is accessible
- Try clicking the refresh button

### "Failed to create task" error

- Verify the Shrike server is running
- Check the browser console (F12 → Console) for detailed error messages
- Ensure the URL/input is valid for the selected command

### Jobs not updating in real-time

- The extension uses SSE for live updates
- If disconnected, click the refresh button
- Check that no firewall is blocking the connection

### Icons not showing

- Ensure PNG files exist in the `icons/` folder
- Run the icon generator script or create icons manually
- Reload the extension after adding icons

### "Temporary add-on" message

- This is normal when loading via `about:debugging`
- For permanent installation, see Option 2 or Option 3 above

## Development

### File Structure

```
firefox-extension/
├── manifest.json      # Extension configuration (Manifest V2)
├── popup.html         # Extension popup UI
├── popup.css          # Styles
├── popup.js           # Main functionality (uses browser.* API)
├── generate-icons.js  # Icon generator script
├── icons/
│   ├── icon16.png
│   ├── icon48.png
│   └── icon128.png
└── README.md
```

### Differences from Chrome Extension

- Uses Manifest V2 (Firefox Manifest V3 support is still maturing)
- Uses `browser.*` API instead of `chrome.*`
- Uses `browser_action` instead of `action`
- Includes `browser_specific_settings` for Firefox compatibility

### Making Changes

1. Edit the source files as needed
2. Go to `about:debugging#/runtime/this-firefox`
3. Click "Reload" on the Shrike extension
4. Test your changes

### API Reference

The extension uses these Shrike API endpoints:

- `POST /create` - Create a new job
- `GET /stream` - SSE for job updates
- `GET /tasks` - List available commands
- `GET /jobs/list` - List current jobs
- `POST /job/{id}/cancel` - Cancel a job
- `POST /job/{id}/remove` - Remove a job

See the main Shrike `API_DOCUMENTATION.md` for full details.

## License

This extension is part of the Shrike project. See the main repository LICENSE file.
