# Platform-Specific Binaries

This directory contains platform-specific binary dependencies for the Lowkey Media Viewer application.

## Directory Structure

```
bin/
├── win32/        # Windows binaries (committed to git)
│   ├── ffmpeg.exe
│   ├── ffprobe.exe
│   ├── exiftool.exe
│   └── exiftool_files/  # ExifTool dependencies
├── darwin/       # macOS wrapper scripts (use system binaries)
│   ├── ffmpeg
│   └── ffprobe
└── linux/        # Linux binaries (downloaded on-demand)
    ├── download-binaries.sh  # Script to download binaries
    ├── ffmpeg      (downloaded, ~188MB)
    ├── ffprobe     (downloaded, ~188MB)
    └── exiftool    (wrapper - uses system install)
```

## Binary Management Strategy

### Windows

All binaries are **committed to git** and bundled with the application. No additional installation required.

### macOS

The application uses **wrapper scripts** that delegate to system-installed binaries.

**Required installations:**

```bash
brew install ffmpeg
```

### Linux

FFmpeg and FFprobe are **downloaded on-demand** during the build process to avoid storing large binaries in git. ExifTool uses a wrapper script that requires system installation.

**Automatic download:**
The `npm run package` command automatically downloads Linux binaries if they don't exist.

**Manual download:**

```bash
cd src/main/resources/bin/linux
./download-binaries.sh
```

**Required system installations:**

```bash
# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl

# Fedora/RHEL
sudo dnf install perl-Image-ExifTool

# Arch
sudo pacman -S perl-image-exiftool
```

## Binary Sources

- **FFmpeg (Linux)**: Static builds from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (~130MB download)
- **FFmpeg (Windows)**: Windows builds (existing, committed)
- **ExifTool (Windows)**: ExifTool Windows distribution (existing, committed)

## Why Download Linux Binaries?

GitHub has a 100MB file size limit. The Linux FFmpeg binaries are ~188MB each. Instead of using Git LFS, we download them on-demand during the build process. This approach:

- Avoids git repository bloat
- Doesn't require Git LFS
- Always gets the latest static builds
- Only downloads when needed (Linux builds)

## Build Configuration

The electron-builder configuration in `package.json` copies only platform-specific binaries during the build process:

- Windows builds include `win32/*`
- macOS builds include `darwin/*`
- Linux builds include `linux/*` (after download)
