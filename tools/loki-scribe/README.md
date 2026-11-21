# loki-scribe

Image → text and text → text with Qwen3.5 / Qwen3.8 GGUF models, on a standalone CUDA engine (Rust + CUDA, no
Python, no llama.cpp). It powers the Lowkey Media Server's descriptions and every other text-generation job, and is
just as usable from a shell or an agent. It follows the toolkit [conventions](../CONVENTIONS.md) (streams, flags,
`--worker`).

```
loki-scribe photo.jpg                                          describe a picture
loki-scribe -p "Transcribe all text in the image." --json *.png      a batch: one load, one JSON line per image
loki-scribe -p "Summarize in 3 bullet points." notes.txt             text -> text
loki-scribe -p "What changed between these?" --together a.jpg b.jpg  several images in one request
loki-scribe --think high -p "Is this chart misleading? Why?" chart.png
producer | loki-scribe --batch - --json > results.jsonl              a long-running queue on stdin
```

## Models

Architecture `qwen35` (Qwen3.5 / Qwen3.8 dense: Gated DeltaNet linear attention with gated full attention every
fourth layer, interleaved M-RoPE) with its Qwen3-VL style vision projector (`mmproj*.gguf`). Weight types Q4_K,
Q5_K, Q6_K, Q8_0, F16, BF16, F32. The default is `lmstudio-community/Qwen3.8-27B-GGUF` (Q4_K_M, 16.8 GB, with the
BF16 projector, 0.9 GB), downloaded on first use.

`--model` takes a `.gguf` path, a Hugging Face `publisher/repo[/file.gguf]` (LM Studio's folder layout; downloaded
when missing, resumable), or a name matched against the model folders: next to the binary, its `models/`,
`$LOKI_MODELS`, `./models`, `~/.lmstudio/models` and `~/.cache/lm-studio/models`. So a model already in LM Studio
is used where it is (`--model qwen3.8-27b-uncensored`). The projector is the `mmproj*.gguf` beside the model (F16
preferred) unless `--mmproj` says otherwise; `--mmproj none` runs text only.

A 27B Q4_K_M needs ~17 GB of VRAM for the weights plus ~0.3 GB per concurrent request (DeltaNet state + KV cache
for its context): 24 GB cards run 8 requests at once.

## Batching

Every input image is its own request. Requests are scheduled continuously: up to `--parallel` (default 8) are
active at once; a new one is admitted (vision encode, chunked prefill on the tensor cores) whenever one finishes,
and every active request advances one token per decode step in a single batched pass over the quantized weights.
Decoding is bandwidth-bound, so 8 requests cost little more per step than one (RTX 4090, 27B Q4_K_M: ~27 ms/step
for 1, ~36 ms/step for 8, i.e. ~220 tokens/s in aggregate; prefill ~900-1300 tokens/s).

`--batch FILE` (`-` = stdin) reads one JSON request per line, consumed as they arrive, so a caller can keep
queueing work into one loaded process: `{"id", "prompt", "images": [...], "system", "max_tokens", "think",
"temperature", "seed", "text"}` (missing fields take the command-line values). Positional results come back in
input order, batch results as they finish. `--worker` keeps the model loaded between whole command lines (the
media server and loki-worker use it).

## Output

Plain text, or with `--json` one object per result: `{"index", "id", "input", "text", "finish", "prompt_tokens",
"tokens", "secs", "model"}` (+ `"thinking"` with `--show-thinking`, `"error"` when a request failed; with `--json`
failures do not change the exit status). `-o DIR` writes `<stem>.txt` per input. `--show-prompt` prints the
final prompt (the GGUF's chat template, images as their token counts) without loading the model.

## Thinking and sampling

Thinking is off by default (`<think></think>` is pre-filled, as the template does with `enable_thinking=false`);
`--think low|medium|high` turns it on, and the reasoning is returned separately. Sampling defaults: temperature
0.7, top-k 20, top-p 0.8 (1.0 / 20 / 0.95 when thinking); `--temp 0` is greedy.

## Verification

`src/bin/scribe_check.rs` (developer checks, needs a GPU and a model): `tokenize` (ids, compared against
`llama-tokenize`), `quant` (GEMV / dequant / GEMM of one weight against an f64 CPU reference), `bench` (GEMV
bandwidth), `logits` (prefill + greedy decode). Against llama.cpp on the same GGUF the top next-token
probabilities agree to ~0.002 and greedy continuations (text and vision) match token for token.
`cargo test --test tokenizer` runs llama.cpp's qwen35 tokenizer fixtures (GPU-free).

Environment: `LOKI_SCRIBE_MODEL` (default model), `LOKI_MODELS`, `LOKI_MODELS_DOWNLOAD`, `LOKI_PROFILE=1`
(per-category GPU timing), `LOKI_SCRIBE_DEBUG_TOP=N` (first-token probabilities on stderr).
