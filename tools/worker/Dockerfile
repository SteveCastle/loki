# syntax=docker/dockerfile:1.7
# =============================================================================
# loki-worker — the GPU engines as a resident worker node
# =============================================================================
# One image, every engine: loki-retouch, loki-reshoot, loki-dream and
# loki-segment (built from tools/), run by loki-worker (media-server/cmd/
# loki-worker), which keeps each engine's models on the GPU between jobs and
# serves them to the Lowkey Media Server:
#
#   * persistent node (RunPod pod, EC2/GCE VM, any Docker host): an HTTP API
#     on $PORT (8080) behind a bearer token (LOKI_WORKER_TOKEN);
#   * RunPod queue-based serverless: picked automatically when RunPod's
#     worker environment is present (no port needed).
#
# Models are NOT in the image (17-42 GB per engine): they are fetched on first
# use (or at start with LOKI_WORKER_PRELOAD) into /models — mount a volume
# there (a RunPod network volume is used automatically when present).
#
# BUILD CONTEXT IS THE REPO ROOT; tools/worker/Dockerfile.dockerignore keeps
# it to the sources this build needs:
#
#   docker build -f tools/worker/Dockerfile -t loki-worker .
#   docker run --gpus all -p 8080:8080 -e LOKI_WORKER_TOKEN=secret \
#     -v loki-models:/models loki-worker
#
# GPUs: loki-retouch, loki-reshoot and loki-dream use FP8 tensor cores (Ada
# sm_89, Hopper sm_90, Blackwell through PTX): RTX 4090 / 5090, L4, L40S,
# RTX 6000 Ada, H100, H200, B200. loki-segment also runs on Ampere (A100,
# A6000, A40, RTX 3090). 24 GB of VRAM is the floor; with 48 GB or
# more several engines keep their models loaded side by side.
# =============================================================================

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04
# Where the engines come from: "source" compiles them here (a local build),
# "prebuilt" packages the Linux binaries CI already built
# (tools/worker/prebuilt/<engine>, from the build-ai-tool artifacts), so CI
# never compiles the CUDA engines twice.
ARG ENGINES_STAGE=source

# -----------------------------------------------------------------------------
# Stage 1 — the CUDA engines (Rust + nvcc)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS engines-source
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
WORKDIR /src
COPY tools/ tools/
# SASS per engine (retouch, reshoot and dream use FP8 tensor cores: sm_89+;
# segment also runs on Ampere), plus compute_90 PTX the driver JITs on newer
# GPUs. ENGINES limits the build (e.g. "loki-segment").
ARG ENGINES="loki-retouch loki-reshoot loki-dream loki-segment"
ARG FP8_ARCHS=89,90
ARG ARCHS=80,86,89,90
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target \
    set -eux; mkdir -p /out; \
    for e in $ENGINES; do \
      case "$e" in loki-segment) a="$ARCHS" ;; *) a="$FP8_ARCHS" ;; esac; \
      LOKI_CUDA_ARCHS="$a" LOKI_CUDA_PTX=90 cargo build --release --locked \
        --manifest-path "tools/$e/Cargo.toml" --target-dir "/src/target/$e" --bin "$e"; \
      cp "/src/target/$e/release/$e" /out/; \
    done; \
    ls -la /out

FROM scratch AS engines-prebuilt
COPY tools/worker/prebuilt/ /out/

FROM engines-${ENGINES_STAGE} AS engines

# -----------------------------------------------------------------------------
# Stage 2 — loki-worker (Go)
# -----------------------------------------------------------------------------
FROM golang:1.25-bookworm AS worker
WORKDIR /src/media-server
COPY media-server/go.mod media-server/go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY media-server/workerapi/ workerapi/
COPY media-server/cmd/loki-worker/ cmd/loki-worker/
ARG VERSION=dev
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/loki-worker ./cmd/loki-worker

# -----------------------------------------------------------------------------
# Stage 3 — ffmpeg (loki-reshoot decodes references and muxes its output)
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS ffmpeg
# A static GPL build from BtbN: the newest release branch, found through the
# GitHub API (asset names follow the branch), else the master build. Pass
# FFMPEG_URL to pin one.
ARG FFMPEG_URL=
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && rm -rf /var/lib/apt/lists/* \
 && base=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest \
 && url="$FFMPEG_URL" \
 && if [ -z "$url" ]; then \
      name=$(curl -fsSL https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest \
        | grep -o '"name": *"ffmpeg-n[0-9.]*-latest-linux64-gpl-[0-9.]*\.tar\.xz"' \
        | sed 's/.*"\(ffmpeg[^"]*\)"/\1/' | sort -V | tail -1 || true); \
      url="$base/${name:-ffmpeg-master-latest-linux64-gpl.tar.xz}"; \
    fi \
 && echo "ffmpeg: $url" && mkdir -p /out \
 && curl -fsSL "$url" | tar -xJ --strip-components=2 -C /out --wildcards '*/bin/ffmpeg' '*/bin/ffprobe'

# -----------------------------------------------------------------------------
# Stage 4 — runtime: the CUDA runtime libraries (the NVIDIA driver comes from
# the host through the container toolkit)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tini \
 && rm -rf /var/lib/apt/lists/*
COPY --from=engines --chmod=755 /out/ /usr/local/bin/
COPY --from=worker /out/loki-worker /usr/local/bin/loki-worker
COPY --from=ffmpeg /out/ /usr/local/bin/
ARG VERSION=dev
LABEL org.opencontainers.image.title="loki-worker" \
      org.opencontainers.image.description="Lowkey GPU engines (loki-retouch, loki-reshoot, loki-dream, loki-segment) as a resident worker node" \
      org.opencontainers.image.source="https://github.com/SteveCastle/loki" \
      org.opencontainers.image.version="${VERSION}"
ENV LOKI_MODELS=/models \
    LOKI_WORKER_DATA=/tmp/loki-worker \
    PORT=8080 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME ["/models"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD [ -n "$RUNPOD_WEBHOOK_GET_JOB" ] || curl -fsS "http://127.0.0.1:${PORT}/ping" >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "loki-worker"]
