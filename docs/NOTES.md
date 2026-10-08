# loki-reshoot — MiniMax H3 ref2va inference in Rust/CUDA (sibling of 4kify)

Goal: a self-contained CLI (no Python/PyTorch) reproducing ComfyUI's
`video_minimax_h3_r2v` workflow (MiniMaxH3ReferenceToVideo -> BasicGuider (cfg 1, ONE DiT pass per step)
-> SamplerCustomAdvanced res_multistep / `simple` scheduler / 20 steps -> video VAE decode + audio VAE decode -> mp4).
Hardware target: RTX 4090 (24 GB, sm_89), CUDA 12.6 nvcc, Windows. EFFICIENCY IS A REQUIREMENT:
int8 tensor-core GEMMs, int8/fp8 (Sage-style) attention, fused epilogues, no needless copies,
row-chunked MLP for long sequences, models loaded in phases (text encoder -> evicted -> DiT -> evicted -> VAEs).

## Reference implementation (read it, it is the spec)
ComfyUI root: `C:\Users\steph\Downloads\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\ComfyUI`
Embedded python (has torch+comfy deps): `...\ComfyUI_windows_portable\python_embeded\python.exe`; run reference
scripts with cwd = the ComfyUI root and `sys.path.insert(0, os.getcwd())` (see `reference_4kify/` + 4kify's
`../4kify/ref/ref_dit.py`, `ref_te.py` for the pattern: dump .bin f32 tensors, compare in Rust).
- `comfy/ldm/minimax/model.py`      DiT (PackedLayout, rope, adaLN, blocks, final layer)
- `comfy/ldm/minimax/vae.py`        video VAE (3D causal conv encoder, ViT3D decoder, tiling, temporal chunking)
- `comfy/ldm/minimax/audio_vae.py`  audio VAE (DAC encoder + BigVGAN decoder)
- `comfy/text_encoders/minimax.py`, `qwen3vl.py`, `llama.py` (Qwen3VL_32BConfig)  text/vision conditioning
- `comfy_extras/nodes_minimax_h3.py` MiniMaxH3ReferenceToVideo (ref preprocessing / latent shapes)
- `comfy/model_base.py` class MiniMaxH3 (extra_conds, audio carry scale), `comfy/model_sampling.py` ModelSamplingAV
- `comfy/k_diffusion/sampling.py` res_multistep, `comfy/samplers.py` simple_scheduler
- workflow: `ComfyUI/user/default/workflows/video_minimax_h3_r2v.json`; explainer `../MiniMax-H3-Explained.md`
- comfy_kitchen (nvfp4 / int8 kernels, eager reference): `python_embeded/Lib/site-packages/comfy_kitchen/backends/eager/`

## Models (test copies, read-only): `C:\Users\steph\dev\loki-reshoot\models\`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (20.97 GB) DiT. int8 ConvRot (per-row int8, 256-wide Hadamard
  on activations, same format as 4kify's Qwen DiT), 50 blocks, hidden 5376, 56 heads x128, ffn 14336 (fc1 -> 28672 = [gate|up]),
  "adaln curve" form: `adaln_t_table [1025,8]` lerp'd, adaln linears are 8 -> 6*5376*3 (f16 weights, no silu), token_refiner (2 bf16 blocks),
  condition_proj 5120->5376, video_patch_proj 96->5376 (f32), audio_patch_proj 32->5376 (f32), final_layer with f32 heads.
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.7 GB) Qwen3-VL-32B truncated to 50 layers, NO final norm / lm_head.
  Linear weights are NVFP4 (u8 packed 2xE2M1, hi nibble = even element; fp8-e4m3 block scale per 16 along K in cuBLAS
  *swizzled* layout, f32 `weight_scale_2` per tensor), `comfy_quant` says full_precision_matrix_mult => dequantize to bf16 then matmul.
  `o_proj` and `down_proj` additionally have AWQ `pre_quant_scale` (bf16 per input channel) multiplied into the input first.
  embed_tokens is int8 tensorwise (+ weight_scale). Vision tower is bf16 (27 blocks, 1152, deepstack at 8/16/24, out 5120).
- `minimax_h3_video_vae_fp16.safetensors` (5.2 GB), `minimax_h3_audio_vae_fp32.safetensors` (0.6 GB).
Download source: https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/{diffusion_models,text_encoders,vae}/<file>

## GPU etiquette (one 4090 shared by all workers!)
Run EVERY GPU job (rust tests/bins and python reference dumps) through `bash tools/gpu.sh <cmd>` (mkdir-lock, serializes).
Keep each job short (<= ~10 min). ComfyUI is stopped; do not start it. Never leave GPU processes running.
Clips in tests must be tiny (~1 s: use 22 frames => latent_t 7, and ~0.2-0.4 MP) so nothing takes long.

## Shared engine (copied from 4kify, already compiling)
`cuda.rs` (Device, launch, Tensor buffers), `ops.rs` (int8 GEMM w/ fused epilogues, bf16 GEMM, quant_rows with
ConvRot Hadamard + fused RMSNorm/SwiGLU, sage attention, flash attention, rope), `weights.rs` (qlinear loaders),
`tensor.rs`, `safetensors.rs`, `tokenizer.rs` (Qwen BPE), `image.rs` (lanczos). `reference_4kify/` holds the Qwen-Image
dit/vae/text_encoder/llm/vision/pipeline sources for inspiration (NOT compiled).
build.rs compiles every `kernels/*.cu` to a fatbin; cuda.rs `FATBINS` lists them. Each worker OWNS exactly the files
named in its task (a .cu + a .rs); do not edit other shared files unless the task says so (add new helper fns in your own module).
If you need a change in shared code (ops.rs/cuda.rs/common.cuh), keep it additive and mention it in your final report.

## Cross-module data conventions (all device tensors unless noted)
- Video latent (normalized, as the DiT sees it): f32 channel-first `[24, T, H, W]` contiguous (= torch [1,24,T,H,W]); H=pixels/16.
- Audio latent: f32 `[32, 2, T]` (channel, stereo, time) (= torch [1,32,2,T]); 40 latent frames / s @32 kHz (800 samples each).
- Raw text conditioning: Qwen hidden state after layer 50: `[L, 5120]` (bf16 or f32) + per-token tags (u8: 0 = vision block incl. its
  <|vision_start|>/<|vision_end|>, 1 = text).
- Pixel frames: host `Vec<u8>` RGB `[T, H, W, 3]`.
