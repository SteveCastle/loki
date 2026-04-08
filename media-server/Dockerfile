# =============================================================================
# Lowkey Media Server — Container Build
# =============================================================================
# Multi-stage build: compile Go binary, then run in a minimal image with ffmpeg.
#
# Build:  docker build -t lowkey-media-server .
# Run:    docker run -p 8090:8090 -v lowkey-data:/data lowkey-media-server
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build the Go binary
# ---------------------------------------------------------------------------
FROM golang:1.24-bookworm AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /lowkeymediaserver .

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash lowkey

# Directories for data, config, and media — all under /data for simple volume mounts
RUN mkdir -p /data/db /data/config /data/media /data/packages && \
    chown -R lowkey:lowkey /data

COPY --from=builder /lowkeymediaserver /usr/local/bin/lowkeymediaserver

# Default environment — all configurable at runtime via -e flags
ENV LOWKEY_DB_PATH=/data/db/media.db \
    LOWKEY_DOWNLOAD_PATH=/data/media \
    LOWKEY_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    XDG_DATA_HOME=/data/config

EXPOSE 8090

USER lowkey
ENTRYPOINT ["lowkeymediaserver"]
