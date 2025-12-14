# Shrike Media Server

**⚠️ Alpha Software Warning:** Shrike Media Server is in the alpha phase of development. The server binds to port 8090 and exposes your media library over HTTP with **no authentication**. If you don't want your media accessible on your network, ensure the port is blocked by your firewall or only run Shrike on trusted networks.

Shrike Media Server is a companion for the Lowkey Media Viewer. It allows for managing long-running offline tasks like media tagging, transcript generation, media conversion, file serving, and media ingestion.

<img width="1628" height="494" alt="Screenshot 2025-09-20 083904" src="https://github.com/user-attachments/assets/e814d2a5-7088-46b2-9a8c-d537b989b018" />

<img width="1661" height="1606" alt="Screenshot 2025-09-20 082929" src="https://github.com/user-attachments/assets/850672bc-da70-4269-8049-cc651b0abed6" />

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation (End Users)](#installation-end-users)
- [Development Setup](#development-setup)
  - [Prerequisites](#prerequisites)
  - [Building from Source](#building-from-source)
  - [Project Structure](#project-structure)
  - [Embedded Binaries](#embedded-binaries)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tasks](#available-tasks)
- [API Documentation](#api-documentation)
- [Chrome Extension](#chrome-extension)
- [License](#license)

---

## Features

- **Job Queue Management**: Create, monitor, cancel, and manage long-running media processing jobs
- **Media Browser**: Browse, search, and preview media files with a modern web interface
- **Auto-Tagging**: Automatic image tagging using ONNX-based machine learning models (WD Tagger)
- **LLM Descriptions**: Generate image descriptions using Ollama vision models
- **Transcription**: Generate video transcripts using Faster Whisper
- **Media Processing**: FFmpeg integration for media conversion and manipulation
- **Media Ingestion**: Batch import media files into the database
- **Real-Time Updates**: Server-Sent Events (SSE) for live job status updates
- **System Tray**: Windows system tray integration for easy access
- **Database Persistence**: SQLite-backed job and media database

---

## System Requirements

- **Operating System**: Windows 10/11 (x64)
- **Disk Space**: ~500MB for binaries and embedded tools
- **RAM**: 4GB minimum, 8GB+ recommended for ML tagging

---

## Installation (End Users)

1. **Run the shrike.exe binary.** This will start the background server and create an icon in your Windows system tray for launching the WebUI.

   <img width="408" height="247" alt="Screenshot 2025-09-20 080659" src="https://github.com/user-attachments/assets/4b8a0141-08d4-4fb9-9c42-78db5dbd25ad" />

2. **Check Lowkey Database Path.** Open the Config tab in the Web UI. You should see the path to your Lowkey Media Database. If it's incorrect, you can manually change it here.

3. **Set Up Model Paths.** You'll need to point the server at a few files to enable auto tagging:

   - [Download Model Files for AutoTagger](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main)
   - [Install Ollama for LLM-based description generation](https://ollama.com/)
   - [Download Faster Whisper for Video Transcription](https://github.com/Purfview/whisper-standalone-win)

   <img width="1233" height="1693" alt="Screenshot 2025-09-20 080416" src="https://github.com/user-attachments/assets/5eb008ae-88fb-4519-af03-4e55afbb6601" />

---

## Development Setup

### Prerequisites

#### 1. Install Go

Shrike requires **Go 1.24.0** or later.

**Windows (Recommended):**

1. Download the latest Go installer from [https://go.dev/dl/](https://go.dev/dl/)
2. Run the MSI installer
3. Verify installation:
   ```powershell
   go version
   # Should output: go version go1.24.x windows/amd64
   ```

**Alternative (using winget):**

```powershell
winget install GoLang.Go
```

**Alternative (using Chocolatey):**

```powershell
choco install golang
```

#### 2. Install Git

Required for cloning the repository and Go module management.

```powershell
winget install Git.Git
```

#### 3. C Compiler (for CGO - SQLite)

The SQLite package (`modernc.org/sqlite`) is a pure Go implementation and does **not** require CGO. No C compiler is needed.

#### 4. Optional: External Tools

For full functionality, install these optional tools:

| Tool                                                                 | Purpose                             | Installation                             |
| -------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------- |
| [Ollama](https://ollama.com/)                                        | LLM-based image descriptions        | Download and run installer               |
| [ONNX Runtime](https://onnxruntime.ai/)                              | ML model inference for auto-tagging | Download DLL, configure path in settings |
| [Faster Whisper](https://github.com/Purfview/whisper-standalone-win) | Video transcription                 | Download, configure path in settings     |

> **Note:** FFmpeg, yt-dlp, gallery-dl, and ExifTool are embedded in the binary and extracted at runtime.

### Building from Source

#### 1. Clone the Repository

```powershell
git clone https://github.com/stevecastle/shrike.git
cd shrike
```

#### 2. Download Dependencies

```powershell
go mod download
```

#### 3. Verify Dependencies

```powershell
go mod verify
```

#### 4. Build the Executable

```powershell
# Standard build
go build -o shrike.exe .

# Build with optimizations (smaller binary, no debug info)
go build -ldflags="-s -w" -o shrike.exe .
```

#### 5. Run the Server

```powershell
.\shrike.exe
```

The server will start on `http://localhost:8090` and a system tray icon will appear.

### Development Mode

For faster iteration during development, you can use `go run`:

```powershell
go run .
```

### Running Tests

```powershell
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for a specific package
go test -v ./media/...
```

### Project Structure

```
shrike/
├── main.go                 # Application entry point, HTTP handlers, system tray
├── go.mod                  # Go module definition
├── go.sum                  # Dependency checksums
│
├── appconfig/              # Application configuration management
│   └── config.go           # Config loading/saving, defaults
│
├── assets/                 # Static assets
│   └── logo.ico            # System tray icon
│
├── client/                 # Frontend static files
│   └── static/
│       ├── styles.css      # Main stylesheet
│       └── details.css     # Job details page styles
│
├── embedexec/              # Embedded executable extraction
│   ├── embedexec.go        # Runtime extraction of embedded binaries
│   └── bin/                # Embedded executables (see below)
│
├── jobqueue/               # Job queue implementation
│   └── jobqueue.go         # Queue, job state, persistence
│
├── media/                  # Media database operations
│   ├── media.go            # Media queries, search, pagination
│   └── media_test.go       # Media tests
│
├── onnxtag/                # ONNX-based image tagging
│   ├── onnxtag.go          # Tagger implementation
│   └── config.go           # Tagger configuration
│
├── renderer/               # HTML template rendering
│   ├── renderer.go         # Template loading, middleware
│   └── templates/          # Go HTML templates
│       ├── index.go.html   # Home page
│       ├── jobs.go.html    # Jobs list
│       ├── detail.go.html  # Job detail view
│       ├── media.go.html   # Media browser
│       ├── config.go.html  # Configuration page
│       └── ...
│
├── runners/                # Job execution runners
│   └── runners.go          # Worker pool implementation
│
├── stream/                 # Server-Sent Events (SSE)
│   └── stream.go           # Real-time update streaming
│
├── tasks/                  # Task implementations
│   ├── registry.go         # Task registration
│   ├── autotag.go          # ONNX auto-tagging
│   ├── autotag_vision.go   # LLM vision tagging
│   ├── command.go          # Generic command execution
│   ├── ffmpeg.go           # FFmpeg processing
│   ├── media_ingest.go     # Media file ingestion
│   ├── media_metadata.go   # Metadata generation
│   ├── media_move.go       # File moving with DB update
│   ├── media_cleanup.go    # Orphan cleanup
│   └── ...
│
├── chrome-extension/       # Browser extension for sending URLs
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── popup.css
│
├── cmd/                    # Additional command-line tools
│   ├── onnxtag/            # Standalone ONNX tagger CLI
│   ├── dbcopy/             # Database copy utility
│   └── sbs/                # Side-by-side comparison tool
│
├── API_DOCUMENTATION.md    # Detailed API reference
└── README.md               # This file
```

### Embedded Binaries

Shrike embeds several third-party executables that are extracted to `%ProgramData%\Shrike\tmp\` at runtime:

| Binary                   | Purpose                         |
| ------------------------ | ------------------------------- |
| `ffmpeg.exe`             | Media conversion and processing |
| `ffprobe.exe`            | Media file analysis             |
| `ffplay.exe`             | Media playback                  |
| `yt-dlp.exe`             | Video downloading               |
| `gallery-dl.exe`         | Image gallery downloading       |
| `exiftool.exe`           | Metadata extraction             |
| `faster-whisper-xxl.exe` | Video transcription (optional)  |
| `dedupe.exe`             | Duplicate detection             |
| `createdump.exe`         | Crash dump creation             |

The embedded binaries are located in `embedexec/bin/`. To update them:

1. Download the new binary
2. Replace the file in `embedexec/bin/`
3. Rebuild the project

> **Note:** Including these binaries significantly increases the final executable size (~200MB+).

---

## Configuration

Configuration is stored at: `%APPDATA%\Lowkey Media Viewer\config.json`

### Configuration Options

```json
{
  "dbPath": "C:\\path\\to\\your\\database.db",

  "ollamaBaseUrl": "http://localhost:11434",
  "ollamaModel": "llama3.2-vision",
  "describePrompt": "Please describe this image...",
  "autotagPrompt": "Please analyze this image and select tags...",

  "onnxTagger": {
    "modelPath": "C:\\path\\to\\model.onnx",
    "labelsPath": "C:\\path\\to\\selected_tags.csv",
    "configPath": "C:\\path\\to\\config.json",
    "ortSharedLibraryPath": "C:\\path\\to\\onnxruntime.dll",
    "generalThreshold": 0.35,
    "characterThreshold": 0.85
  },

  "fasterWhisperPath": "C:\\path\\to\\faster-whisper-xxl.exe"
}
```

### ONNX Tagger Setup

1. Download model files from [HuggingFace](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main):

   - `model.onnx` - The neural network model
   - `selected_tags.csv` - Tag labels
   - `config.json` - Model configuration (optional)

2. Download [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases):

   - Get `onnxruntime-win-x64-*.zip`
   - Extract `onnxruntime.dll` from `lib/` folder

3. Configure paths in the Shrike Config tab

---

## Usage

Once the server is installed, Lowkey Media Viewer should detect it and be able to create jobs. You can also create bulk jobs on the results of any search from the Media Browser tab.

### Web Interface

Access the web interface at: `http://localhost:8090`

- **Home**: Quick job creation
- **Jobs**: View and manage all jobs
- **Media**: Browse and search media files
- **Config**: Configure server settings
- **Stats**: View database statistics

### System Tray

Right-click the Shrike icon in the system tray:

- **Open Web UI**: Launch browser to web interface
- **Quit**: Shutdown the server

---

## Available Tasks

| Task ID      | Name               | Description                               |
| ------------ | ------------------ | ----------------------------------------- |
| `wait`       | Wait               | Test task that waits 5 seconds            |
| `gallery-dl` | gallery-dl         | Download media from websites              |
| `yt-dlp`     | yt-dlp             | Download videos from YouTube, etc.        |
| `ffmpeg`     | ffmpeg             | Process media files                       |
| `ingest`     | Ingest Media Files | Scan directories and add to database      |
| `metadata`   | Generate Metadata  | Generate descriptions, hashes, dimensions |
| `autotag`    | Auto Tag (ONNX)    | Automatic image tagging with ML           |
| `move`       | Move Media Files   | Move files and update database            |
| `remove`     | Remove Media       | Remove entries from database              |
| `cleanup`    | CleanUp            | Remove orphaned database entries          |

---

## API Documentation

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for detailed HTTP API reference including:

- Endpoint specifications
- Request/response formats
- SSE streaming events
- Example curl commands

---

## Chrome Extension

A Chrome extension is included for quickly sending URLs to Shrike.

### Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome-extension/` folder

### Usage

1. Navigate to a page you want to download
2. Click the Shrike extension icon
3. Select the download method (yt-dlp, gallery-dl, etc.)
4. Job is created automatically

---

## Troubleshooting

### Port 8090 Already in Use

If another application is using port 8090, you'll need to modify the port in `main.go`:

```go
srv = &http.Server{
    Addr:    ":8090",  // Change this port
    Handler: mux,
}
```

### Database Connection Errors

1. Ensure the database path in config.json exists
2. Check file permissions on the database file
3. Ensure no other process has locked the database

### ONNX Tagger Not Working

1. Verify all model files are downloaded
2. Check ONNX Runtime DLL path is correct
3. Ensure model is compatible (ONNX opset 11+)
4. Check Windows Event Viewer for crash logs

### System Tray Icon Not Appearing

1. Check Windows notification area settings
2. Ensure hidden icons are visible
3. Try restarting Windows Explorer

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `go test ./...`
5. Submit a pull request

---

## License

See [LICENSE](./LICENSE) for details.
