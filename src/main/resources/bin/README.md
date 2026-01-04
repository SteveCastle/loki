# Platform-Specific Binaries

This directory contains platform-specific binary dependencies for the Lowkey Media Viewer application.

## Directory Structure

```
bin/
├── win32/        # Windows binaries (downloaded on-demand)
│   ├── download-binaries.sh  # Script to download binaries
│   ├── ffmpeg.exe      (downloaded, ~40MB)
│   └── ffprobe.exe     (downloaded, ~40MB)
├── darwin/       # macOS binaries (downloaded on-demand)
│   ├── download-binaries.sh  # Script to download binaries
│   ├── ffmpeg      (downloaded, ~60MB)
│   └── ffprobe     (downloaded, ~60MB)
└── linux/        # Linux binaries (downloaded on-demand)
    ├── download-binaries.sh  # Script to download binaries
    ├── ffmpeg      (downloaded, ~188MB)
    └── ffprobe     (downloaded, ~188MB)
```

## Binary Management Strategy

All platforms now use **on-demand download** during the build process to avoid storing large binaries in git.

## Automatic Download

The `npm run package` command automatically downloads binaries for your platform if they don't exist.

## Manual Download

### Windows
```bash
cd src/main/resources/bin/win32
./download-binaries.sh
```

### macOS
```bash
cd src/main/resources/bin/darwin
./download-binaries.sh
```

### Linux
```bash
cd src/main/resources/bin/linux
./download-binaries.sh
```

## Binary Sources

- **FFmpeg (Windows)**: Builds from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
- **FFmpeg (macOS)**: Builds from [evermeet.cx](https://evermeet.cx/ffmpeg/)
- **FFmpeg (Linux)**: Static builds from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)

## Removed Dependencies

**ExifTool** has been removed from the application. Image and video dimensions are now extracted using FFprobe.
