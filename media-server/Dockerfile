# =============================================================================
# Lowkey Media Server — Container Build
# =============================================================================
# Multi-stage:
#   1. node-builder   — webpack-builds the React SPA into media-server/loki-static
#   2. go-builder     — builds lowkeymediaserver + the onnxtag CLI with the SPA embedded
#   3. ort-fetcher    — downloads the ONNX Runtime shared library
#   4. runtime        — Debian slim with ffmpeg/exiftool/yt-dlp/gallery-dl pre-installed,
#                       all "bundled" binaries arranged under /opt/lowkey/bin/ so
#                       deps.MustBundled() resolves them.
#
# BUILD CONTEXT IS THE REPO ROOT (not media-server/) because we need the SPA source.
#
#   docker build -f media-server/Dockerfile -t lowkey-media-server .
#
# RUN (mapping host port 18090 → container 8090 to avoid colliding with a
# local dev instance on 8090):
#
#   docker run --rm -p 18090:8090 \
#     -v lowkey-data:/data \
#     -v /path/to/your/media:/media \
#     -e LOWKEY_ROOT_1=/media:Local \
#     lowkey-media-server
#
# Then visit http://localhost:18090
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — build the React renderer
# -----------------------------------------------------------------------------
FROM node:20-bookworm-slim AS node-builder

WORKDIR /src
# Install deps first (cached layer when only sources change)
COPY package.json package-lock.json* yarn.lock* ./
RUN if [ -f yarn.lock ]; then \
      corepack enable && yarn install --frozen-lockfile --network-timeout 600000; \
    else \
      npm ci --no-audit --no-fund; \
    fi

# Copy the source needed for the renderer build
COPY .erb ./.erb
COPY src ./src
COPY scripts ./scripts
COPY tsconfig.json ./
COPY assets ./assets
COPY release/app/package.json ./release/app/package.json

# build:web runs webpack + copies the output into media-server/loki-static
RUN npm run build:web

# -----------------------------------------------------------------------------
# Stage 2 — compile the Go server (with SPA embedded) and the onnxtag CLI
# -----------------------------------------------------------------------------
FROM golang:1.24-bookworm AS go-builder

WORKDIR /src/media-server
COPY media-server/go.mod media-server/go.sum ./
RUN go mod download

# Copy server source
COPY media-server/ ./

# Overwrite the (gitignored) loki-static with the freshly built SPA from stage 1
COPY --from=node-builder /src/media-server/loki-static ./loki-static

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64
RUN go build -ldflags="-s -w" -o /out/lowkeymediaserver . && \
    go build -ldflags="-s -w" -o /out/onnxtag ./cmd/onnxtag

# -----------------------------------------------------------------------------
# Stage 3 — fetch ONNX Runtime shared library (matches bundled-versions.json)
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS ort-fetcher

ARG ONNXRUNTIME_VERSION=1.21.0
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -fsSL -o ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" && \
    tar -xzf ort.tgz && \
    mkdir -p /out && \
    # Newest libonnxruntime.so* file copied as onnxruntime.so (no extension)
    # so it matches what bundled/manifest.go expects on Linux.
    cp -L "$(ls onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so."*[0-9] | sort -V | tail -1)" /out/onnxruntime.so

# -----------------------------------------------------------------------------
# Stage 4 — runtime image
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim

# Native bundled deps installed via apt
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libimage-exiftool-perl \
        ca-certificates \
        python3 \
        python3-venv \
        pipx \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -s /bin/bash lowkey && \
    mkdir -p /data /opt/lowkey/bin /media && \
    chown -R lowkey:lowkey /data /opt/lowkey /media

# ---- Bundled binaries arranged at /opt/lowkey/bin/<id> ----
# bundled.Manifest expects: ffmpeg, ffprobe, ffplay, exiftool, onnxtag,
# onnxruntime (shared lib). We symlink the apt-installed ones and copy the
# Go-built / downloaded ones.
RUN ln -sf /usr/bin/ffmpeg     /opt/lowkey/bin/ffmpeg && \
    ln -sf /usr/bin/ffprobe    /opt/lowkey/bin/ffprobe && \
    ln -sf /usr/bin/ffplay     /opt/lowkey/bin/ffplay && \
    ln -sf /usr/bin/exiftool   /opt/lowkey/bin/exiftool

COPY --from=go-builder    /out/lowkeymediaserver /opt/lowkey/lowkeymediaserver
COPY --from=go-builder    /out/onnxtag           /opt/lowkey/bin/onnxtag
COPY --from=ort-fetcher   /out/onnxruntime.so    /opt/lowkey/bin/onnxruntime.so
RUN chmod +x /opt/lowkey/lowkeymediaserver /opt/lowkey/bin/onnxtag && \
    chown -R lowkey:lowkey /opt/lowkey

# ---- Optional tools — installed for the lowkey user via pipx ----
USER lowkey
ENV PATH=/home/lowkey/.local/bin:$PATH
RUN pipx install yt-dlp && pipx install gallery-dl
USER root

# -----------------------------------------------------------------------------
# Configuration via environment
# -----------------------------------------------------------------------------
# Where the server keeps its state (db, models, onboarding flag, etc.). The
# data dir is also influenced by XDG_DATA_HOME on Linux (platform.GetDataDir).
ENV LOWKEY_DB_PATH=/data/db/media.db \
    LOWKEY_DOWNLOAD_PATH=/data/downloads \
    XDG_DATA_HOME=/data \
    XDG_CACHE_HOME=/data/cache

# Ollama — defaults to the host's instance (works on Docker Desktop). Override
# to point at a remote Ollama with -e LOWKEY_OLLAMA_BASE_URL=http://...:11434
ENV LOWKEY_OLLAMA_BASE_URL=http://host.docker.internal:11434

# -----------------------------------------------------------------------------
# Storage roots — at least one must be provided at run time. Two formats:
# -----------------------------------------------------------------------------
#
# 1) SIMPLE LOCAL PATHS (numbered, optional :label suffix)
#    docker run ... \
#      -v /host/photos:/media/photos \
#      -v /host/videos:/media/videos \
#      -e LOWKEY_ROOT_1=/media/photos:Photos \
#      -e LOWKEY_ROOT_2=/media/videos:Videos \
#      -e LOWKEY_DEFAULT_ROOT=1
#
# 2) FULL JSON (S3 + local, supports per-root S3 credentials)
#    docker run ... \
#      -e 'LOWKEY_ROOTS=[
#            {"type":"local","label":"Local","path":"/media","default":true},
#            {"type":"s3","label":"Cold storage","bucket":"my-media",
#             "endpoint":"https://s3.us-east-1.amazonaws.com",
#             "region":"us-east-1","accessKey":"AKIA...","secretKey":"...",
#             "prefix":"library/"}
#          ]'
#
# LOWKEY_DEFAULT_ROOT picks the default destination for uploads/downloads,
# either a 1-based index or a label string. Only honored for LOWKEY_ROOT_<N>;
# LOWKEY_ROOTS JSON should set "default":true directly on the entry.
# -----------------------------------------------------------------------------

# Other recognized env vars (no default — set at run time if you want them):
#   LOWKEY_JWT_SECRET            — auth signing secret (generate a strong one)
#   LOWKEY_OLLAMA_MODEL          — default vision model, e.g. llama3.2-vision
#   LOWKEY_DISCORD_TOKEN         — required by the Discord ingest task
#   LOWKEY_FASTER_WHISPER_PATH   — path to faster-whisper if you want transcription

EXPOSE 8090

VOLUME ["/data", "/media"]

WORKDIR /opt/lowkey
USER lowkey

# tini handles signals + zombie reaping cleanly (matters because pipx tools
# may spawn child processes for yt-dlp etc.).
ENTRYPOINT ["/usr/bin/tini", "--", "./lowkeymediaserver"]
