# Spec: deploy Lowkey GPU workers on RunPod

**Audience:** an agent with shell access on the machine that runs the Lowkey Media Server (or one that can reach it), plus the
`lokictl` CLI. **Goal:** every GPU job the server runs — Retouch (edit/upscale/restore), Reshoot (video with sound, H3 and
FastH3), Dream (text→image), Scene-editor selection and roto tracking — runs on RunPod workers, with models stored once on
RunPod storage and files exchanged through a transfer bucket. Read `tools/worker/README.md` for background; this file is the
procedure.

## 0. Inputs you need from the user (ask for anything missing, never guess)

| input | used for |
|---|---|
| `RUNPOD_API_KEY` | creating volumes, pods, templates, endpoints (runpod.io → Settings → API Keys, read/write) |
| media server URL + a lokictl credential | `lokictl --server <url>` with `lokictl key create`/`auth` already set up, or `--token <lk_...>` |
| transfer bucket credentials (S3-compatible) | serverless file exchange — see §2.2; ask whether to use an existing S3 storage root of the server, or create a bucket |
| budget / scale | how many persistent pods (default 1) and the serverless max workers (default 3) |
| datacenter preference | default: pick one (§1.3) |

Never print or log the RunPod key, bucket secrets or worker tokens. They end up in the server's config (the API redacts them).

## 1. Preconditions — verify, stop if any fails

1. **Image is public**: `docker manifest inspect ghcr.io/stevecastle/loki-worker:latest` (or
   `curl -s "https://ghcr.io/token?scope=repository:stevecastle/loki-worker:pull"` returns a token and the manifest fetch with it
   is 200). If not public, tell the user: repository → Packages → loki-worker → Package settings → Change visibility → Public.
   Pin the version for production: use `ghcr.io/stevecastle/loki-worker:<app version>` (the tag of the newest release) rather
   than `latest`, and pass it as `--image` everywhere below.
2. **Server is new enough**: `lokictl pool status` must work (the `/api/engines` API). If it 404s the server predates worker
   pools — the user must update the server first.
3. **Dream**: release builds include loki-dream only when the repository variable `LOKI_DREAM` was `1` at build time. Check
   `lokictl task list` for `dream`. If absent, Dream jobs are simply unavailable on this server; continue with the rest and
   tell the user.
4. **GPU choice** (all four engines; retouch/reshoot/dream need FP8 tensor cores, sm_89+):
   - persistent pods: `NVIDIA L40S` (48 GB, engines stay loaded side by side) preferred, else `NVIDIA RTX 6000 Ada Generation`
     (48 GB), else `NVIDIA GeForce RTX 4090` (24 GB, models swap per job), `NVIDIA H100 80GB HBM3` for heavy Reshoot use.
   - serverless: list several (RunPod rents whatever is free): `NVIDIA L40S`, `NVIDIA RTX 6000 Ada Generation`,
     `NVIDIA GeForce RTX 4090`, `NVIDIA H100 80GB HBM3`. **Never** Ampere (A100, A6000, A40, 3090) for these pools.
   - Check the exact GPU type ids exist: `GET https://rest.runpod.io/v1/gputypes` is not guaranteed; if a create call returns
     400 for a GPU id, list types with the RunPod GraphQL `gpuTypes { id displayName memoryInGb }` and adjust.
5. **Datacenter**: network volumes are per datacenter, and only pods/endpoints in that datacenter can mount them. Choose ONE
   datacenter that offers the chosen GPUs on Secure Cloud and supports network volumes (e.g. `EU-RO-1`, `US-KS-2`, `CA-MTL-1`);
   use it for the volume, every pod (`--datacenter`) and the endpoint.

## 2. Storage

### 2.1 Models: one RunPod network volume

- Size: model files total ≈ 17 GB (retouch) + 42 GB (reshoot H3) + 22 GB (reshoot FastH3) + 27 GB (dream) + 2 GB (segment)
  ≈ **110 GB**; create **200 GB** (headroom for updates; downloads write `.part` files beside the finished ones).
- Create it (REST): `POST https://rest.runpod.io/v1/networkvolumes` with `{"name": "loki-models", "size": 200, "dataCenterId": "<dc>"}`
  (Bearer `RUNPOD_API_KEY`). Record its `id`.
- **Layout contract:** models live in the `models/` folder at the root of the volume. Serverless workers mount the volume at
  `/runpod-volume` and loki-worker reads `/runpod-volume/models`; pods created with `lokictl pool create runpod-pods
  --network-volume` mount it at `/runpod-volume` with `LOKI_MODELS=/runpod-volume/models` too (master `6618c2e` and later —
  build lokictl from master: `cd media-server && go build -o lokictl ./cmd/lokictl`; an older lokictl mounts it at `/models`,
  where serverless workers would not find the models). Pods made by hand must use the same two values.
  Several workers downloading at once is safe: each file is fetched by one process under a `<file>.lock`.

### 2.2 Transfer bucket (serverless only)

Serverless workers are not addressable, so inputs and outputs travel through an S3-compatible bucket via presigned URLs
(6-hour TTL). Without one, only payloads ≤ 7 MB work — stills, but not Reshoot's video references or long outputs. Persistent
pods do NOT need it (the server uploads to them directly).

- Use Cloudflare R2 (no egress fees), AWS S3, or Backblaze B2. Create a dedicated bucket, e.g. `lowkey-worker-transfer`,
  private (no public access needed; presigned URLs carry auth).
- Credentials: a key limited to that bucket with `PutObject`, `GetObject`, `DeleteObject` (and `ListBucket` for the check).
- Lifecycle rule: expire objects under prefix `loki-worker/` after 1 day (the server deletes them after each job; this catches
  crashes).
- Endpoint values for `lokictl`: R2 → `--s3-endpoint https://<account>.r2.cloudflarestorage.com --region auto`; S3 → leave
  `--s3-endpoint` empty, `--region <bucket region>`; B2 → `--s3-endpoint https://s3.<region>.backblazeb2.com --region <region>`.
- Alternative: if the server already has an S3 storage root reachable from the internet (not a LAN MinIO), pass
  `--transfer-root <its label>` instead of bucket flags.

## 3. Deploy

Use the same `--image`, `--datacenter`, `--network-volume` for everything.

### 3.1 Seed the models (once)

One persistent pod downloads everything into the volume before any serverless worker starts (otherwise the first jobs each
pay a 100+ GB download):

```bash
lokictl pool create runpod-pods gpu --count 1 --gpu "NVIDIA L40S" --gpu "NVIDIA RTX 6000 Ada Generation" \
  --network-volume <volume-id> --datacenter <dc> --image ghcr.io/stevecastle/loki-worker:<version> \
  --preload retouch,reshoot,dream,segment
lokictl -o table pool status                         # repeat until the gpu/<pod> row is healthy (first start: 20-40 min)
lokictl pool warm gpu reshoot --model fast           # also fetch FastH3 Omni
```

By hand instead (only if lokictl cannot be used): `POST https://rest.runpod.io/v1/pods` with
`{"name":"loki-gpu-1","imageName":"ghcr.io/stevecastle/loki-worker:<version>","gpuTypeIds":["NVIDIA L40S","NVIDIA RTX 6000 Ada Generation"],
"cloudType":"SECURE","containerDiskInGb":30,"networkVolumeId":"<volume-id>","volumeMountPath":"/runpod-volume","dataCenterIds":["<dc>"],
"ports":["8080/http"],"env":{"LOKI_WORKER_TOKEN":"<random 48 hex>","LOKI_MODELS":"/runpod-volume/models","LOKI_WORKER_PRELOAD":"loki-retouch,loki-reshoot,loki-dream,loki-segment"}}`,
then `lokictl pool add-node gpu https://<pod-id>-8080.proxy.runpod.net --token <token> --name <pod-id> --no-test` and wait as
above. Record the pod id: `pool destroy` only deletes pods that `pool create` made.

Watch progress: the pod's logs (RunPod console) show `downloading <file>: N%` lines; `/ping` answers 204 until preload ends.

### 3.2 Serverless endpoint (burst capacity, scales to zero)

```bash
lokictl pool create runpod-serverless burst --max-workers 3 --min-workers 0 --idle 120 --max-job-secs 3600 \
  --gpu "NVIDIA L40S" --gpu "NVIDIA RTX 6000 Ada Generation" --gpu "NVIDIA GeForce RTX 4090" --gpu "NVIDIA H100 80GB HBM3" \
  --network-volume <volume-id> --datacenter <dc> --image ghcr.io/stevecastle/loki-worker:<version> \
  --bucket lowkey-worker-transfer --s3-endpoint <endpoint> --region <region> --access-key <key> --secret-key <secret>
```

(`--datacenter` only applies to pods; for the endpoint it is ignored — the network volume pins the endpoint to its datacenter.) Keep `--max-workers` within the account's serverless quota. Flashboot is on, the idle timeout keeps a worker
(and its loaded models) for 120 s after a job.

### 3.3 Placement

```bash
lokictl pool strategy pool --fallback=false     # or hybrid if the server machine has its own suitable GPU
lokictl pool route segment pool:gpu              # selection sessions need a persistent node, never serverless
lokictl pool status
```

Retouch/Reshoot/Dream/scene-track then use both pools (warm persistent pod first, serverless when it is busy).

## 4. Verify (all must pass; report timings)

Use a small still and a 3-5 s clip from the library (paths from `lokictl media query` or the user).

1. `lokictl -o table pool status`: local + `gpu/<pod>` healthy, `burst` healthy with workers listed.
2. Retouch: `lokictl job run retouch "<image>" --field prompt="make it night" --field steps=8 --wait` → output beside the image.
   Run twice; the second should be seconds (warm).
3. Reshoot: `lokictl job run reshoot "<image>" --field prompt="slow push in" --field duration=2 --wait` (H3), then with
   `--field model=fast`.
4. Dream (if available): `lokictl job run dream --field prompt="a red fox in the snow" --field count=1 --wait`.
5. Serverless path: with the pod busy (start a long Reshoot), submit a Retouch and confirm the job log says
   `running on GPU worker burst`; and a Reshoot with a video reference (exercises the bucket).
6. Scene editor: open a scene on an image (`POST /api/scenes {source}`), `POST /api/scenes/{id}/segment {"points":[[x,y,1]]}`
   twice — the second answers in tens of ms.
7. Cancel: start a Reshoot, `lokictl job cancel <id>`; the pod's engine stays up (`pool status` still healthy, next job warm).

## 5. Hand-off notes for the user

- Costs: persistent pods bill while running (`lokictl pool scale gpu --count 0` stops them, keeping the pool; `pool destroy gpu`
  deletes them). The network volume bills monthly until deleted in the RunPod console. Serverless bills per second of work.
- Rollback: `lokictl pool strategy local` (everything back on the server's GPU), pools stay configured.
- Updating: new image tag → `pool destroy` + `pool create` for pods (models stay on the volume), and update the endpoint's
  template image in the RunPod console (or recreate the endpoint).
- Troubleshooting: worker logs in the RunPod console; `lokictl pool check`; a pod stuck "unhealthy" with `starting
  (preloading models)` is still downloading; `401` = token mismatch (re-add the node); serverless jobs `IN_QUEUE` for minutes =
  no GPU of the listed types free (add types) or a cold start pulling the 3 GB image.
