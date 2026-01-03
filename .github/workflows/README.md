# GitHub Workflows

This directory contains GitHub Actions workflows for automated CI/CD.

## Release Workflow

**File:** `release.yml`

**Trigger:** Pushes to the `master` branch

**Purpose:** Automatically builds, tests, and releases the Loki application.

### Workflow Steps

1. **Test Electron App** - Runs the Electron app test suite with `yarn test`
2. **Test Go Server** - Runs the Go media-server tests with `go test ./...`
3. **Version and Tag** - Reads the version from `package.json` and creates a git tag (e.g., `v2.6.8`)
4. **Build Electron App** - Builds the Electron app for macOS, Linux, and Windows using `yarn package`
5. **Build Go Media Server** - Builds the Go server for multiple platforms:
   - Linux (amd64, arm64)
   - macOS (amd64, arm64)
   - Windows (amd64)
6. **Generate Changelog** - Creates an AI-generated changelog from git commits
7. **Create Release** - Creates a GitHub release with all binaries and the changelog

### Key Features

- **Automatic Versioning:** Uses the version number from `package.json`
- **Cross-Platform Builds:** Creates binaries for all major platforms
- **Caching:** Uses GitHub Actions cache for faster builds
- **Artifact Management:** Automatically uploads and organizes build artifacts
- **Smart Tagging:** Only creates new tags if they don't already exist

### Requirements

- The workflow requires no additional secrets beyond the default `GITHUB_TOKEN`
- Version bumps should be done by updating `package.json` before pushing to master
- Ensure all tests pass before merging to master

### Customization

To change the release behavior, edit `.github/workflows/release.yml`:
- Modify build targets in the `build-go-server` job matrix
- Update changelog format in the `generate-changelog` job
- Adjust test commands in the `test-electron` and `test-go` jobs
