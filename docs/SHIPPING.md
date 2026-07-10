# Shipping the Media Server — Manual Steps

Last updated: 2026-07-06. Everything automatable is already in the repo
(release pipeline, installer script, license files, Docker publishing,
headless provisioning). This is the list of things **only you** can do,
plus what to watch on the first release run.

## TL;DR priority order

1. Watch the first release run end-to-end (untested runners: mingw install, macOS smoke, Inno Setup).
2. Make the GHCR image public (one click, after first publish).
3. Set up Windows code signing (Azure Trusted Signing, ~$10/mo) — biggest UX unlock.
4. Decide on GPL source hosting (5-minute task per release, see Licensing).
5. macOS: decide if/when to pay the $99/yr Apple tax.

---

## 1. First release run — what to watch

The next push to `master` touching build paths triggers `release.yml`. New,
never-before-exercised-in-CI pieces:

- [ ] **`build-media-server` (windows)**: the "Ensure C toolchain" step — if
      the runner image already has gcc it's instant; otherwise
      `choco install mingw` runs (~1–2 min). Then the fetch script builds
      `embed.exe`/`onnxtag.exe` with cgo. If this fails, the log will say
      exactly why (compiler missing vs stub detected).
- [ ] **Windows installer step**: Inno Setup 6 is preinstalled on
      `windows-latest`, but the `setup.iss` has never compiled in CI. If
      `ISCC.exe` isn't found at `%ProgramFiles(x86)%\Inno Setup 6\`, check
      the runner image docs and adjust the path.
- [ ] **macOS smoke tests**: `smoke-bundled.sh` boots the server on the
      runner — first time on macOS runners. Gatekeeper shouldn't interfere
      (no quarantine flag on runner-built files), but watch it.
- [ ] **`publish-docker`**: needs `packages: write`, which is set
      job-level. First push creates the GHCR package.
- [ ] Confirm the GitHub Release ends up with: 4 `lowkey-media-server-*`
      archives, `lowkey-media-server-setup-windows-amd64.exe`, and the
      Electron artifacts.

## 2. Docker / GHCR

- [ ] **Make the package public** (one-time): after the first
      `publish-docker` run, go to
      github.com → your profile → Packages → `lowkey-media-server` →
      Package settings → Change visibility → Public. Until then, pulls
      require authentication and the "easy install" pitch doesn't work.
- [ ] **(Optional) link the package to the repo** in the same settings page
      so it shows on the repo sidebar, and add a short README on the
      package page. Suggested user-facing quick start:

      ```bash
      docker run -d --name lowkey \
        -p 10111:10111 \
        -v lowkey-data:/data \
        -v /path/to/your/media:/media \
        -e LOWKEY_ROOT_1=/media:Media \
        ghcr.io/stevecastle/lowkey-media-server:latest
      # then open http://localhost:10111 — the setup wizard takes it from there
      ```

      Fully headless (skips the wizard's account step):

      ```bash
        -e LOWKEY_ADMIN_USER=me -e LOWKEY_ADMIN_PASSWORD=strong-password \
        -e LOWKEY_JWT_SECRET=$(openssl rand -hex 32)
      ```
- [ ] **(Optional) multi-arch**: the image is amd64-only. NAS boxes and Pis
      increasingly want arm64. Add `platforms: linux/amd64,linux/arm64` to
      the `build-push-action` step — but test first; the ONNX runtime stage
      downloads the x64 `.so` and would need an arch-conditional URL.

## 3. Windows users

- [ ] **Code signing — the single biggest UX improvement available.**
      Unsigned installers hit SmartScreen ("Windows protected your PC") and
      a chunk of non-technical users stop there. Cheapest good option:
      **Azure Trusted Signing** (~$9.99/mo, individual accounts supported
      with 3+ years of verifiable identity):
      1. Azure account → create a "Trusted Signing" resource (Basic tier).
      2. Identity validation (individual or org).
      3. Add the `azure/trusted-signing-action` step to `release.yml` after
         the installer build, signing both `lowkeymediaserver.exe` and the
         installer.
      Alternative: a classic OV cert (Sectigo/SSL.com, ~$100–400/yr) —
      note OV certs build SmartScreen reputation gradually; Trusted
      Signing is generally immediate.
- [ ] **Test the installer once on a real/VM Windows box**: install →
      wizard opens → finish → reboot → confirm the tray icon returns (the
      "start with Windows" task) → uninstall cleanly.
- [ ] **(Optional, later) winget manifest** (`microsoft/winget-pkgs` PR) so
      `winget install lowkey-media-server` works. Requires a signed,
      versioned installer URL — do after signing.
- [ ] **(Optional) trim the payload ~55%**: switch `bundled-versions.json`
      to BtbN `-shared` builds (ffmpeg/ffprobe/ffplay share DLLs instead of
      three ~130 MB static binaries). Needs extract-list changes (many DLLs,
      not 3 exes) and a re-pin of hashes — mechanical but fiddly.

## 4. Licensing

Already handled in-repo: `licenses/THIRD-PARTY-LICENSES.md` + full GPL-3.0
text ship inside every archive and the Docker image; models are downloaded
by the user (never redistributed); the ffmpeg builds must stay **GPL**
variants because HLS transcodes with libx264 (`tasks/hls.go`).

What only you can decide/do:

- [ ] **GPL corresponding-source hosting (recommended, 5 min per release).**
      The notice file currently points at the upstream source (exact FFmpeg
      commit + BtbN build scripts + evermeet). The bulletproof reading of
      GPL §6 wants *you* to provide the source. Easy compliance: when a
      release goes out, download the FFmpeg source tarball for the pinned
      commit and attach it to the same GitHub Release (or keep a
      `sources/` branch). Decide if you care about the strict reading; most
      small projects live with upstream links until someone asks.
- [ ] **Honor source requests**: the notice file promises source "upon
      request" — if anyone ever emails asking, send the tarball.
- [ ] **When bumping ffmpeg/exiftool versions**, update the pinned URLs,
      hashes (`fetch-bundled-deps -Mode update` prints them), **and** the
      commit/version references in `licenses/THIRD-PARTY-LICENSES.md`.
- [ ] **Model licenses**: CCIP is OpenRAIL (use restrictions) — it's
      user-downloaded so you're clear, but if you ever pre-bundle models,
      recheck each license first. SigLIP2/DINOv2/WD-tagger/YuNet/SFace are
      permissive (Apache/MIT).

## 5. macOS

**Signing + notarization are wired into `release.yml`** using the same
secrets the Electron build already notarizes with (`CSC_LINK`,
`CSC_KEY_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`,
`APPLE_TEAM_ID`). The darwin jobs sign every Mach-O in the payload with
Developer ID + hardened runtime (workers get a library-validation
entitlement because they dlopen the ONNX runtime), smoke test the signed
binaries, then `notarytool submit --wait`. If the secrets are ever absent
the steps skip and the release ships unsigned rather than failing.

**The Mac-native artifact is now a DMG**: each darwin job assembles
`Lowkey Media Server.app` (menu-bar agent app — `LSUIElement`, no Dock
icon, tray with Open Web UI/Quit from `tray_darwin.go`; bundled deps live
under `Contents/MacOS/bin` so the executable-relative resolver works
unchanged), signs the bundle, wraps it in a drag-to-Applications DMG,
notarizes the DMG and staples the ticket. The tar.gz still ships for CLI
users — its binaries carry the same signatures, and notarization tickets
are per-signature, so they pass Gatekeeper's online check without a second
submission. The macOS *server* binary now builds with `CGO_ENABLED=1`
(Cocoa for the tray); cross-compiled/dev builds without cgo fall back to
headless mode automatically (`tray_darwin_nocgo.go`).

What to verify / know on the first release:

- [ ] **`CSC_LINK` must be a base64 `.p12` with a "Developer ID
      Application" cert** (electron-builder's convention). Wrong cert type
      → notarization rejects; the signing step logs the identity it found.
- [ ] **`tray_darwin.go` has never been compiled** (needs a Mac) — it
      mirrors the Windows tray code, but watch the darwin build step.
- [ ] **Menu-bar icon rendering**: the tray reuses
      `media-server/assets/logo.png` as a *template* icon (monochrome from
      alpha). If it renders as a blob or wrong size on a real Mac, that's a
      5-minute icon swap in `tray_darwin.go`.
- [ ] Install → launch flow to sanity-check on a real Mac: open DMG → drag
      to Applications → launch → menu-bar icon appears → browser opens to
      the setup wizard → Quit from the menu shuts the server down cleanly.
- [ ] **Notarization adds ~1–5 min** per darwin job (`--wait`). If Apple's
      service hiccups the job fails — rerun the workflow.

## 6. Loose ends worth knowing about

- [ ] **`/api/deps/*` endpoints have no auth** (CORS-only, relied on by the
      Electron renderer cross-origin and the wizard). On a LAN that means
      anyone can trigger/delete model downloads. Fine for home use; gate it
      before recommending exposure beyond localhost/LAN.
- [ ] **The server binds all interfaces** and the first-boot wizard is
      intentionally open pre-account (same model as every NAS). Don't
      port-forward it to the internet before completing setup.
- [ ] **README**: there's no user-facing download/install section pointing
      at the release artifacts (zip vs installer vs docker vs tar.gz).
      Worth writing once the first release with these artifacts exists.
- [ ] **Auto-update for the server** doesn't exist (the Electron app has
      one). Lowest-effort v1: a "new version available" banner in the web
      UI comparing against the GitHub releases API.
