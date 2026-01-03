# Lowkey Media Server

**⚠️ Alpha Software Warning:** Lowkey Media Server is in the alpha phase of development. The server binds to port 8090 and exposes your media library over HTTP with **no authentication**. If you don't want your media accessible on your network, ensure the port is blocked by your firewall or only run Lowkey Media Server on trusted networks.

Lowkey Media Server is a companion for the Lowkey Media Viewer. It allows for managing long-running offline tasks like media tagging, transcript generation, media conversion, file serving, and media ingestion.

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
  - [Dependencies](#dependencies)
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

- **Operating System**: Windows 10/11 (x64) or Linux (x64)
- **Disk Space**: ~500MB for binaries and embedded tools
- **RAM**: 4GB minimum, 8GB+ recommended for ML tagging

---

## Installation (End Users)

1. **Run the lowkeymediaserver.exe binary.** This will start the background server and create an icon in your Windows system tray for launching the WebUI.

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

Lowkey Media Server requires **Go 1.24.0** or later.

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

| Tool                                                                 | Purpose                             | Windows Installation                     | Linux Installation                       |
| -------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------- | ---------------------------------------- |
| [Ollama](https://ollama.com/)                                        | LLM-based image descriptions        | Download and run installer               | `curl -fsSL https://ollama.com/install.sh \| sh` |
| [ONNX Runtime](https://onnxruntime.ai/)                              | ML model inference for auto-tagging | Download DLL, configure path in settings | `apt install libonnxruntime` or download .so |
| [Faster Whisper](https://github.com/Purfview/whisper-standalone-win) | Video transcription                 | Download, configure path in settings     | Build from source or use Python version  |

> **Note:** FFmpeg, yt-dlp, and gallery-dl are managed as downloadable dependencies via the web UI.

### Building from Source

#### 1. Clone the Repository

```bash
git clone https://github.com/stevecastle/shrike.git
cd shrike
```

#### 2. Download Dependencies

```bash
go mod download
```

#### 3. Verify Dependencies

```bash
go mod verify
```

#### 4. Build the Executable

The project uses Go build tags to handle platform-specific code. Build for your target platform:

##### Building for Windows

```powershell
# Standard build (from Windows)
go build -o lowkeymediaserver.exe .

# Build with optimizations (smaller binary, no debug info)
go build -ldflags="-s -w" -o lowkeymediaserver.exe .
```

##### Building for Linux

```bash
# Standard build (from Linux)
go build -o lowkeymediaserver .

# Build with optimizations (smaller binary, no debug info)
go build -ldflags="-s -w" -o lowkeymediaserver .
```

##### Cross-Compilation

You can cross-compile from one platform to another using the `GOOS` and `GOARCH` environment variables:

```bash
# Build Windows executable from Linux/macOS
GOOS=windows GOARCH=amd64 go build -o lowkeymediaserver.exe .

# Build Linux executable from Windows (PowerShell)
$env:GOOS="linux"; $env:GOARCH="amd64"; go build -o lowkeymediaserver .

# Build Linux executable from Windows (Command Prompt)
set GOOS=linux
set GOARCH=amd64
go build -o lowkeymediaserver .
```

> **Note:** Cross-compilation works because the project uses pure Go dependencies (no CGO required).

#### 5. Run the Server

**Windows:**
```powershell
.\lowkeymediaserver.exe
```

**Linux:**
```bash
./lowkeymediaserver
```

The server will start on `http://localhost:8090`. On Windows, a system tray icon will appear.

### Development Mode

For faster iteration during development, you can use `go run`:

```bash
go run .
```

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for a specific package
go test -v ./media/...
```

### Project Structure

```
lowkeymediaserver/
├── main.go                 # Windows entry point, HTTP handlers, system tray
├── main_linux.go           # Linux entry point, HTTP handlers
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
├── deps/                   # Dependency management
│   ├── deps.go             # Dependency registry and core logic
│   ├── metadata.go         # Dependency metadata persistence
│   ├── paths.go            # Platform-specific paths
│   ├── exec.go             # Executable command builder
│   ├── ffmpeg.go           # FFmpeg dependency
│   ├── ytdlp.go            # yt-dlp dependency
│   ├── gallerydl.go        # gallery-dl dependency
│   ├── whisper.go          # Faster Whisper dependency
│   ├── onnx.go             # ONNX runtime/model dependency
│   └── ...                 # Other dependencies
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

### Dependencies

Lowkey Media Server uses a dependency management system that downloads executables on-demand rather than embedding them in the binary. This keeps the main executable small and allows for easy updates.

Dependencies are stored in:

- **Windows:** `%ProgramData%\Lowkey Media Server\deps\`
- **Linux:** `/var/lib/lowkeymediaserver/deps/`

| Dependency   | Purpose                         | Source                                |
| ------------ | ------------------------------- | ------------------------------------- |
| `ffmpeg`     | Media conversion and processing | BtbN/FFmpeg-Builds (GitHub)           |
| `yt-dlp`     | Video downloading               | yt-dlp/yt-dlp (GitHub)                |
| `gallery-dl` | Image gallery downloading       | mikf/gallery-dl (GitHub)              |
| `whisper`    | Video transcription             | Purfview/whisper-standalone-win       |
| `onnx`       | ML model inference              | HuggingFace/SmilingWolf               |

#### Managing Dependencies

1. Open the web interface at `http://localhost:8090`
2. Navigate to the **Dependencies** tab
3. Click **Download** next to any dependency you need
4. The server will download and extract the dependency automatically

Dependencies are downloaded from official sources (GitHub releases) and verified before installation. The status of each dependency is shown in the UI.

#### Fallback to System PATH

If a dependency is not installed via the web UI, the server will attempt to find the executable in your system PATH. This allows you to use system-installed versions of tools like FFmpeg if preferred.

---

## Configuration

Configuration file location:

- **Windows:** `%APPDATA%\Lowkey Media Viewer\config.json`
- **Linux:** `~/.config/lowkeymediaviewer/config.json`

### Configuration Options

**Windows Example:**
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

**Linux Example:**
```json
{
  "dbPath": "/home/user/media/database.db",

  "ollamaBaseUrl": "http://localhost:11434",
  "ollamaModel": "llama3.2-vision",
  "describePrompt": "Please describe this image...",
  "autotagPrompt": "Please analyze this image and select tags...",

  "onnxTagger": {
    "modelPath": "/home/user/models/model.onnx",
    "labelsPath": "/home/user/models/selected_tags.csv",
    "configPath": "/home/user/models/config.json",
    "ortSharedLibraryPath": "/usr/lib/libonnxruntime.so",
    "generalThreshold": 0.35,
    "characterThreshold": 0.85
  },

  "fasterWhisperPath": "/usr/local/bin/faster-whisper-xxl"
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

3. Configure paths in the Lowkey Media Server Config tab

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

Right-click the Lowkey Media Server icon in the system tray:

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

A Chrome extension is included for quickly sending URLs to Lowkey Media Server.

### Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome-extension/` folder

### Usage

1. Navigate to a page you want to download
2. Click the Lowkey Media Server extension icon
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

### System Tray Icon Not Appearing (Windows)

1. Check Windows notification area settings
2. Ensure hidden icons are visible
3. Try restarting Windows Explorer

### Linux-Specific Issues

**Permission denied when running:**
```bash
chmod +x lowkeymediaserver
./lowkeymediaserver
```

**Port 8090 requires root privileges:**
On Linux, ports below 1024 require root. Port 8090 should work without root, but if you encounter permission issues:
```bash
# Run with elevated permissions (not recommended for production)
sudo ./lowkeymediaserver

# Or use a reverse proxy like nginx to forward from port 80/443
```

**Missing shared libraries:**
If you encounter errors about missing libraries (e.g., libonnxruntime.so), ensure the library path is set:
```bash
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
./lowkeymediaserver
```

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
