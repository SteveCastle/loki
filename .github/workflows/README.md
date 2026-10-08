# GitHub Workflows

This directory contains GitHub Actions workflows for automated CI/CD.

## Release Workflow

**File:** `release.yml`

**Trigger:** Pushes to the `master` branch

**Purpose:** Automatically builds, tests, and releases the Loki application.

### Workflow Steps

1. **Test Electron App** - Runs the Electron app test suite with `npm test`
2. **Test Go Server** - Runs the Go media-server tests with `go test ./...`
3. **Version and Tag** - Reads the version from `package.json` and creates a git tag (e.g., `v2.6.8`)
4. **Build Electron App** - Builds the Electron app for Windows using `npm package`
5. **Build Go Media Server** - Builds the Go server for Windows (amd64)
6. **Build AI tools** - Builds `loki-retouch` and `loki-reshoot` (`tools/`, Rust + CUDA) for Windows and Linux; non-blocking
7. **Generate Changelog** - Creates an AI-generated changelog from git commits (mainline only, `--first-parent`)
8. **Create Release** - Creates a GitHub release with all binaries and the changelog

### Key Features

- **Automatic Versioning:** Uses the version number from `package.json`
- **Windows Builds:** Creates binaries for Windows platform
- **Caching:** Uses GitHub Actions cache for faster builds
- **Artifact Management:** Automatically uploads and organizes build artifacts
- **Smart Tagging:** Only creates new tags if they don't already exist
- **Test Enforcement:** All tests must pass before releasing

### Requirements

- The workflow requires no additional secrets beyond the default `GITHUB_TOKEN`
- Version bumps should be done by updating `package.json` before pushing to master
- Ensure all tests pass before merging to master

### Customization

To change the release behavior, edit `.github/workflows/release.yml`:

- Add more platforms by modifying the `build-electron` and `build-go-server` jobs
- Update changelog format in the `generate-changelog` job
- Adjust test commands in the `test-electron` and `test-go` jobs

## AI tools Workflow

**File:** `tools.yml` (uses the composite action `.github/actions/build-ai-tool`)

**Trigger:** pull requests and non-`master` pushes that touch `tools/**`, the action, or the workflow itself.

**Purpose:** fast feedback for the standalone CUDA CLIs under `tools/`. Builds, tests (GPU-free tests only: hosted runners have no GPU),
smoke-tests (the executable must start and print help with no NVIDIA driver) and packages each tool on Windows and Linux. The same
action runs as the *Build AI tools* job of `release.yml`, whose archives (`loki-<tool>-<target>.zip` plus a `.zip.sha256`) are attached to the release. The media server's dependency manifest downloads them from `releases/latest/download/` and verifies the sidecar, so the asset names must stay unversioned.
See `tools/README.md`.

Notes:
- The release job is `continue-on-error: true` so a CUDA-toolkit install hiccup cannot block the app/server release. Remove it once the
  jobs are proven stable.
- CUDA 12.6.3 is pinned in the action (it must match the `cuda-12060` feature of `cudarc` in each `Cargo.toml`).
- Windows installs the full toolkit; Linux installs only `nvcc`, `cudart-dev` and `cccl` (verified sufficient to build both tools).
