# Build Setup Instructions

This document describes the build configuration for creating binaries across all supported platforms: Windows, macOS, and Linux.

## Supported Platforms

The release workflow automatically builds for the following platforms:

### Windows
- **Electron App**: Built as `.exe` installer (NSIS)
- **Go Media Server**: `media-server-windows-amd64.exe`
- **Runner**: `windows-latest`

### macOS
- **Electron App**: Built as `.dmg` for both arm64 and x64 architectures
- **Go Media Server**: 
  - `media-server-darwin-arm64` (Apple Silicon)
  - `media-server-darwin-amd64` (Intel)
- **Runner**: `macos-latest`
- **Special Requirements**: Code signing and notarization (see macOS Secrets section below)

### Linux
- **Electron App**: Built as `.AppImage` (universal Linux package)
- **Go Media Server**: `media-server-linux-amd64`
- **Runner**: `ubuntu-latest`
- **Note**: AppImage format works on most Linux distributions including Ubuntu, Fedora, Debian, etc.

## Build Process

When you push to the master branch, the workflow will:

1. Run tests for both Electron and Go components
2. Create a version tag based on `package.json`
3. Build binaries for all platforms in parallel:
   - Windows: Electron app + Go server
   - macOS: Electron app (arm64 + x64) + Go server (arm64 + x64)
   - Linux: Electron app (AppImage) + Go server (amd64)
4. Generate a changelog
5. Create a GitHub release with all binaries attached

## Required GitHub Secrets (macOS Only)

macOS builds require additional secrets for code signing and notarization. Other platforms do not require special secrets.

### Code Signing Certificate

**Secret Name:** `CSC_LINK`  
**Description:** Base64-encoded .p12 certificate file for code signing  
**How to obtain:**
1. Export your Apple Developer ID Application certificate from Keychain Access as a .p12 file
2. Convert it to base64: `base64 -i certificate.p12 | pbcopy`
3. Paste the base64 string as the secret value

**Secret Name:** `CSC_KEY_PASSWORD`  
**Description:** Password for the .p12 certificate file  
**How to obtain:** This is the password you set when exporting the certificate from Keychain Access

### Apple Notarization Credentials

**Secret Name:** `APPLE_ID`  
**Description:** Your Apple ID email address  
**How to obtain:** Use the Apple ID associated with your Apple Developer account

**Secret Name:** `APPLE_ID_PASS`  
**Description:** App-specific password for notarization  
**How to obtain:**
1. Go to https://appleid.apple.com/account/manage
2. Sign in with your Apple ID
3. In the Security section, click "Generate Password" under App-Specific Passwords
4. Generate a password with a label like "Loki Notarization"
5. Copy the generated password and save it as this secret

**Secret Name:** `APPLE_TEAM_ID`  
**Description:** Your Apple Developer Team ID  
**How to obtain:**
1. Go to https://developer.apple.com/account
2. Sign in with your Apple ID
3. Your Team ID is displayed in the membership details section (it's a 10-character alphanumeric string like "XXXXXXXXXX")

## Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each of the five secrets listed above (macOS only):
   - `CSC_LINK`
   - `CSC_KEY_PASSWORD`
   - `APPLE_ID`
   - `APPLE_ID_PASS`
   - `APPLE_TEAM_ID`

**Note:** Windows and Linux builds do not require any additional secrets beyond the default `GITHUB_TOKEN`.

## Linux AppImage Information

The Linux build creates an AppImage, which is a universal Linux package format that:
- Works on most Linux distributions (Ubuntu, Fedora, Debian, Arch, etc.)
- Requires no installation - just make it executable and run
- Includes all dependencies bundled
- Is self-contained and portable

Users can run the AppImage by:
```bash
chmod +x Lowkey-Media-Viewer-*.AppImage
./Lowkey-Media-Viewer-*.AppImage
```

## Troubleshooting

### macOS
- **Notarization fails:** Ensure your Apple ID password is an app-specific password, not your regular Apple ID password
- **Code signing fails:** Verify that your certificate is a "Developer ID Application" certificate, not a "Mac App Distribution" certificate
- **Build timeout:** macOS builds can take longer than Windows builds; this is normal

### Linux
- **AppImage won't run:** Ensure the file is executable (`chmod +x`)
- **Missing dependencies:** AppImage should be self-contained, but very old or minimal Linux systems may need FUSE installed

### Windows
- **Build fails:** Check that Node.js dependencies are properly installed

## References

- [Electron Builder Documentation](https://www.electron.build/)
- [Electron Builder Code Signing (macOS)](https://www.electron.build/code-signing)
- [Apple Notarization Documentation](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [@electron/notarize Documentation](https://github.com/electron/notarize)
- [AppImage Documentation](https://appimage.org/)
