# loki-worker: GPU worker pools for the Lowkey Media Server

`loki-worker` runs the loki-* GPU engines — **loki-retouch** (image edit/upscale/restore), **loki-reshoot** (video with
sound), **loki-dream** (text → image) and **loki-segment** (selection and roto) — on a GPU machine and serves them to
the Lowkey Media Server. Each engine runs as a resident process (`--worker`, see [`../CONVENTIONS.md`](../CONVENTIONS.md))
that keeps its models in VRAM between jobs, so only the first job on a worker pays the minutes of model loading.

A **pool** is a set of interchangeable workers. The media server sends each Retouch, Reshoot, Dream, Scene-track and
selection job to a worker of the pool (preferring one whose engine already has the models loaded), uploads the job's
files with it, streams the engine's log and progress into the job, and writes the results back to the paths the job
asked for, exactly as if the engine had run locally. Running everything on this machine's GPU (the default) keeps
working unchanged.

```
 Lowkey Media Server ──(HTTPS, bearer token)──▶ loki-worker node ──stdin/stdout──▶ loki-retouch --worker  (models resident)
   engine dispatcher                              queue, staging,               ▶ loki-reshoot --worker
   (local GPU = one more target)                  one slot per GPU               ▶ loki-segment --worker (+ sessions)
                     └──(RunPod API + S3 presigned URLs)──▶ RunPod serverless endpoint running the same image
```

## The image

`ghcr.io/stevecastle/loki-worker` (published by the release pipeline from [`Dockerfile`](Dockerfile), packaging the engines its
tool jobs built; tags `latest`, the app version, `sha-<commit>`). One image holds all four engines,
loki-worker, and ffmpeg. Models are not in the image: they download on first use (17 GB retouch, 42 GB reshoot, 27 GB
dream, 1.8 GB segment) into `/models` — **mount a volume there** so they survive restarts.

| GPU | retouch / reshoot / dream | segment |
|---|---|---|
| RTX 4090, RTX 5090, L4 (24 GB), L40S, RTX 6000 Ada, H100, H200, B200 | ✓ | ✓ |
| A100, A6000, A40, RTX 3090 (Ampere) | ✗ (FP8 tensor cores) | ✓ |

24 GB is the floor (a 24 GB card swaps the large models in and out between phases; the VAE and small models stay).
48 GB and up keep an engine's whole pipeline loaded, and several engines side by side (`LOKI_WORKER_COEXIST=auto`).
A multi-GPU machine is one node with one job slot per GPU.

### Settings (environment)

| variable | default | |
|---|---|---|
| `LOKI_WORKER_TOKEN` | — | bearer token callers must send. **Required** unless listening on loopback (or `LOKI_WORKER_NO_TOKEN=1` on a private network). |
| `PORT` / `LOKI_WORKER_ADDR` | `8080` / `:$PORT` | HTTP listen address. |
| `LOKI_MODELS` | `/models` | model folder(s). On RunPod serverless a network volume is used automatically (`/runpod-volume/models`). |
| `LOKI_WORKER_PRELOAD` | — | engines to download and load at start (`retouch,segment`); `/ping` answers 204 until done. |
| `LOKI_WORKER_TOOLS` | all found | limit the engines served. |
| `LOKI_WORKER_GPUS` | `auto` | GPU indices to use (`0,1`). |
| `LOKI_WORKER_COEXIST` | `auto` | keep several engines' models on one GPU: `auto` (≥ 60 GB), `always`, `never`. |
| `LOKI_WORKER_MODE` | `auto` | `http`, or `runpod` (queue-based serverless; picked automatically from RunPod's environment). |
| `LOKI_WORKER_NAME` | hostname / pod id | name shown on the Config page's GPU Engines tab. |
| `LOKI_WORKER_RESERVE_MB` | per engine | VRAM kept free for activations next to resident models. |

## Quick start: any Docker host with an NVIDIA GPU

```bash
docker run -d --name loki-worker --gpus all --restart unless-stopped \
  -p 8080:8080 -e LOKI_WORKER_TOKEN=$(openssl rand -hex 24) \
  -e LOKI_WORKER_PRELOAD=retouch,segment \
  -v loki-models:/models ghcr.io/stevecastle/loki-worker:latest
```

(or `docker compose -f tools/worker/docker-compose.yml up -d`). Then add it to the server — on the **Workers** page
(`/workers`: Add a worker node → Test → Add), or:

```bash
lokictl pool add-node gpu http://gpu-box:8080 --token <token>
lokictl -o table pool status
```

Adding the first node switches the engines strategy to **pool**. Use a reverse proxy with TLS (or a private network /
VPN such as Tailscale) when the worker is reachable from the internet: the token travels with every request.

## RunPod

Set `RUNPOD_API_KEY` (runpod.io → Settings → API Keys) and let lokictl create the workers:

### Persistent pods (lowest latency, billed while running)

```bash
lokictl pool create runpod-pods gpu --count 2 --preload retouch,segment          # RTX 4090 / L40S / 6000 Ada ...
lokictl pool create runpod-pods big --gpu "NVIDIA H100 80GB HBM3" --network-volume <volume-id>
lokictl pool scale gpu --count 4
lokictl pool destroy gpu                                                         # deletes the pods, removes the pool
```

Each pod gets its own token, a 150 GB pod volume at `/models` (or the network volume you pass, which every pod shares —
models are then downloaded once), and is reached through RunPod's HTTPS proxy (`https://<pod>-8080.proxy.runpod.net`).
Pods need a few minutes to pull the image and download models; the Config page's GPU Engines tab shows them as starting until ready.

By hand: deploy a pod from the image with an HTTP port `8080`, `LOKI_WORKER_TOKEN` set and a volume at `/models`, then
`lokictl pool add-node gpu https://<pod-id>-8080.proxy.runpod.net --token <token>`.

### Serverless (scale to zero, billed per second of work)

```bash
lokictl pool create runpod-serverless burst --max-workers 4 --network-volume <volume-id> \
  --bucket my-transfer-bucket --region us-east-1 --access-key ... --secret-key ...
```

This creates a template for the image and a **queue-based** endpoint (`flashboot` on, idle timeout 120 s so a warm
worker keeps its models for the next job). loki-worker detects RunPod's worker environment and pulls jobs itself — no
port, no handler script. Serverless workers cannot be addressed, so files travel through an S3-compatible bucket with
presigned URLs (AWS S3, Cloudflare R2, Backblaze B2, a public MinIO; `--transfer-root <label>` reuses a configured S3
storage root). Without a bucket, files up to 7 MB go inline in the request (stills work; video references do not).
Always give the endpoint a **network volume**: otherwise every cold worker downloads the models again. Interactive
selection needs a persistent node (a session lives in one process); route `segment` to a node pool or keep it local.

Load-balancing serverless endpoints also work: the image serves `/ping` (204 while preloading, 200 ready) on `PORT`;
add the endpoint URL (`https://<endpoint>.api.runpod.ai`) as a node with the worker token. Keep jobs short there
(RunPod may route a status poll to another replica); queue-based endpoints are the reliable serverless choice.

## AWS

**EC2** (persistent): [`aws/launch-ec2.sh`](aws/launch-ec2.sh) starts a GPU instance (g6e.xlarge: L40S 48 GB; g6.xlarge:
L4 24 GB) from the Deep Learning Base AMI, whose user-data runs the container with a token and a gp3 volume for the
models, and prints the `lokictl pool add-node` line. ECS/EKS: use [`k8s/loki-worker.yaml`](k8s/loki-worker.yaml) (EKS
with the NVIDIA device plugin) or the same container definition (`resourceRequirements: GPU 1`, port 8080, an EFS or EBS
volume at `/models`). Security group: open 8080 only to the media server (or put an ALB with TLS in front).

## Kubernetes, GCP, Lambda Labs, Vast.ai, a second PC

Anything that runs an NVIDIA container works the same way: run the image with a GPU, a token and a volume at `/models`,
expose port 8080, and add the URL as a node. [`k8s/loki-worker.yaml`](k8s/loki-worker.yaml) is a Deployment (one GPU per
replica), Service and PVC; each replica is a node (add each pod's address, or front each with its own Service).

## Choosing where engines run (the media server)

| strategy | |
|---|---|
| `local` (default) | every engine on this machine's GPU, as before; pools are ignored. |
| `pool` | engines go to the pools; this machine needs no GPU. `--fallback=true` runs here when no worker is reachable. |
| `hybrid` | this machine's GPU is one more worker: a free local GPU is used first, then warm workers, then the rest. |

Per-engine routes override it: `lokictl pool route segment local` keeps clicks on the local GPU while generation goes to
the pool; `pool:<name>` pins an engine to one pool. The Config page's GPU Engines tab shows the same controls, every worker's health, GPU,
loaded models and queue, and can preload models on a pool ahead of use (`lokictl pool warm gpu retouch reshoot`).

The queue runs as many GPU jobs at once as the pools have slots (one per node GPU, `maxJobs` per serverless endpoint),
plus this machine's one when it is a target. Splat training always runs locally and takes the local GPU slot.

Config (`config.json`, `engines`), also settable whole through `LOWKEY_ENGINES` (JSON) or `LOWKEY_WORKER_NODES`
(`https://a:8080|token,https://b:8080|token`) for container deployments of the server:

```json
"engines": {
  "strategy": "pool",
  "routes": {"loki-segment": "local"},
  "fallback": false,
  "pools": [
    {"name": "gpu", "kind": "nodes", "nodes": [{"name": "pod1", "url": "https://abc-8080.proxy.runpod.net", "token": "lw_..."}]},
    {"name": "burst", "kind": "runpod", "endpointId": "xyz", "apiKey": "rpa_...", "maxJobs": 4,
     "transfer": {"bucket": "my-transfer-bucket", "region": "us-east-1", "accessKey": "...", "secretKey": "..."}}
  ]
}
```

## API (HTTP mode)

All routes but `/ping` need `Authorization: Bearer <token>`. Paths inside jobs use `$JOB` for the job's folder on the
worker (`in/<k>/<file>` for uploads, `out/<k>/...` for outputs); the client in
[`media-server/workerapi`](../../media-server/workerapi) maps them to and from local paths.

| route | |
|---|---|
| `GET /ping` | 200 ready, 204 starting/preloading (load balancers) |
| `GET /v1/info[?help=1]` | worker, GPUs (free memory, busy job), engines (versions, per-GPU state, resident models), queue |
| `POST /v1/jobs` | multipart: `spec` (JSON `JobSpec`: `tool`, `args`, `inputs`, `dirs`, `light`) then one part per input; or a JSON spec with inputs by `url`/`data` and `outputs` (`putUrls` / `inline`) |
| `GET /v1/jobs/{id}?logs=N` | state, log lines from N, step i/n, stdout, outputs |
| `GET /v1/jobs/{id}/outputs.tar` | the outputs; `GET /v1/jobs/{id}/files/{path}` one file |
| `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | stop; drop with its files (finished jobs expire after an hour) |
| `POST /v1/warm` | `{"tools": [...], "model": "fast"}` — download and load models in the background |
| `POST /v1/sessions`, `PUT /v1/sessions/{id}/files/{path}`, `POST /v1/sessions/{id}/call`, `GET /v1/sessions/{id}/files.tar?prefix=`, `DELETE /v1/sessions/{id}` | interactive loki-segment sessions (`$SESSION` paths) |

`light` jobs (`--show-prompt`, `--help`) run as their own short process and never queue behind GPU work.

Serverless (RunPod queue) jobs take `{"input": {"spec": JobSpec}}` (or `{"warm": {...}}`, `{"info": true}`), stream
`{"logs", "step", "steps"}` chunks while running, and return the final `JobStatus` as the output.

## Building and testing

```bash
docker build -f tools/worker/Dockerfile -t loki-worker .           # repo root as context; ~30 min cold (CUDA kernels)
# CI packages engines it already built instead: put them in tools/worker/prebuilt/ and add --build-arg ENGINES_STAGE=prebuilt
docker build -f tools/worker/Dockerfile --build-arg ENGINES="loki-segment loki-dream" -t loki-worker:light .
cd media-server && go test ./workerapi/ ./cmd/loki-worker/          # fake engines, no GPU
# against a real worker with models (a GPU):
LOKI_WORKER_E2E=http://127.0.0.1:8080 LOKI_WORKER_E2E_TOKEN=... go test ./workerapi -run E2E -v
```

Run a worker from a source checkout without Docker: build the engines (`cargo build --release` in each `tools/loki-*`),
put them next to `loki-worker` (`go build ./cmd/loki-worker` in `media-server/`) or on PATH, and start it with
`LOKI_WORKER_TOKEN=... LOKI_MODELS=~/bin/models loki-worker --addr :8080`.
