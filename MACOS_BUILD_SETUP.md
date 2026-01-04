# macOS Build Setup Instructions

This document describes the GitHub secrets required to build and notarize the macOS binary for this project.

## Required GitHub Secrets

To enable macOS binary building with code signing and notarization, you need to configure the following secrets in your GitHub repository settings:

### 1. Code Signing Certificate

**Secret Name:** `CSC_LINK`  
**Description:** Base64-encoded .p12 certificate file for code signing  
**How to obtain:**
1. Export your Apple Developer ID Application certificate from Keychain Access as a .p12 file
2. Convert it to base64: `base64 -i certificate.p12 | pbcopy`
3. Paste the base64 string as the secret value

**Secret Name:** `CSC_KEY_PASSWORD`  
**Description:** Password for the .p12 certificate file  
**How to obtain:** This is the password you set when exporting the certificate from Keychain Access

### 2. Apple Notarization Credentials

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
4. Add each of the five secrets listed above:
   - `CSC_LINK`
   - `CSC_KEY_PASSWORD`
   - `APPLE_ID`
   - `APPLE_ID_PASS`
   - `APPLE_TEAM_ID`

## Build Process

Once the secrets are configured, the macOS build will automatically run when you push to the master branch:

1. The `build-electron-macos` job will build the Electron app for both arm64 and x64 architectures
2. The app will be code-signed with your Developer ID certificate
3. The app will be notarized with Apple's notarization service
4. The `build-go-server-macos` job will build the Go media server for both arm64 and x64 architectures
5. All artifacts (.dmg files and binaries) will be uploaded to the GitHub release

## Troubleshooting

- **Notarization fails:** Ensure your Apple ID password is an app-specific password, not your regular Apple ID password
- **Code signing fails:** Verify that your certificate is a "Developer ID Application" certificate, not a "Mac App Distribution" certificate
- **Build timeout:** macOS builds can take longer than Windows builds; this is normal

## References

- [Electron Builder Code Signing Documentation](https://www.electron.build/code-signing)
- [Apple Notarization Documentation](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [@electron/notarize Documentation](https://github.com/electron/notarize)
