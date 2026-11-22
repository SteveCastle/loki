//! loki-scribe: image -> text and text -> text with Qwen3.5/3.8-family GGUF models on a standalone CUDA engine.
use anyhow::{bail, ensure, Context, Result};
use clap::{CommandFactory, Parser};
use loki_scribe::chat::{self, Think};
use loki_scribe::engine::{self, Engine, Job, Output, Params, Sampling};
use loki_scribe::{models, worker};
use serde_json::json;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

const AFTER_HELP: &str = "\
INPUTS
  Each image is its own request (a batch: the model loads once and up to --parallel requests decode together).
  A text file's contents are appended to the prompt (text -> text). No inputs: one request from the prompt alone.
  --together puts every image into one request instead (compare, combine, read a multi-page document).
  --batch FILE reads one JSON request per line (`-` = stdin, read as it arrives, so a caller can keep queueing):
    {\"id\": \"a1\", \"prompt\": \"...\", \"images\": [\"x.jpg\"], \"system\": \"...\", \"max_tokens\": 300, \"think\": \"off\",
     \"temperature\": 0.7, \"seed\": 1}   (missing fields take the command-line values; \"text\" is appended to the prompt)

OUTPUT
  stdout: the answer text. Several results print as `### <input>` sections; with --json one JSON object per result:
    {\"index\": 0, \"id\": null, \"input\": \"cat.jpg\", \"text\": \"...\", \"finish\": \"stop\", \"prompt_tokens\": 812,
     \"tokens\": 96, \"secs\": 3.1, \"model\": \"...\"}     (+ \"thinking\" with --show-thinking, \"error\" when it failed)
  Positional inputs are reported in input order, --batch requests as they finish.
  -o DIR writes <input stem>.txt per result into DIR and prints the paths; -o FILE writes everything to FILE.
  stderr: progress. Exit status 0 ok, 1 a request failed (with --json failures are reported in their objects instead).

MODELS
  --model takes a .gguf path, a Hugging Face publisher/repo[/file.gguf] (LM Studio's folder layout; downloaded when
  missing), or a name matched against the model folders (binary dir, models/, $LOKI_MODELS, ~/.lmstudio/models).
  Default: $LOKI_SCRIBE_MODEL, else lmstudio-community/Qwen3.8-27B-GGUF (Q4_K_M, ~17 GB). The vision projector is
  the mmproj*.gguf beside the model unless --mmproj says otherwise. Architecture: qwen35 (Qwen3.5 / Qwen3.8 dense);
  weights Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16. Needs an NVIDIA GPU with room for the weights (24 GB for 27B Q4_K_M).

EXAMPLES
  loki-scribe photo.jpg                                       describe a picture
  loki-scribe -p \"Transcribe all text in the image.\" --json *.png     a batch, one JSON line each
  loki-scribe -p \"Summarize in 3 bullet points.\" notes.txt          text -> text
  loki-scribe -p \"What changed between these?\" --together a.jpg b.jpg
  loki-scribe --think high -p \"Is this chart misleading? Why?\" chart.png
  generate-requests | loki-scribe --batch - --json > results.jsonl     a long-running queue
  loki-scribe --model JonathanColetti/Qwen3.8-27B-Uncensored-GGUF -p hi   another qwen35 GGUF (downloaded if missing)";

#[derive(Parser, Clone, Debug)]
#[command(name = "loki-scribe", version, about = "Image -> text and text -> text with Qwen3.5/3.8 GGUF models (standalone CUDA engine)", after_help = AFTER_HELP)]
struct Cli {
    /// Images (one request each) or text files (appended to the prompt); `-` reads one input from stdin
    inputs: Vec<String>,
    /// The instruction
    #[arg(short = 'p', long)]
    prompt: Option<String>,
    /// Read the instruction from a file (`-` = stdin)
    #[arg(short = 'P', long)]
    prompt_file: Option<String>,
    /// System message
    #[arg(short = 's', long)]
    system: Option<String>,
    /// Read the system message from a file
    #[arg(long)]
    system_file: Option<PathBuf>,
    /// All images in one request instead of one request each
    #[arg(long)]
    together: bool,
    /// JSON-lines requests (`-` = stdin, consumed as they arrive)
    #[arg(long)]
    batch: Option<String>,
    /// Output: `-` (stdout, default), a directory (one .txt per input) or a file
    #[arg(short = 'o', long)]
    out: Option<String>,
    /// One JSON object per result on stdout
    #[arg(long)]
    json: bool,
    /// Maximum tokens to generate per request
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Think before answering: off, low, medium, high
    #[arg(long, default_value = "off")]
    think: String,
    /// Include the thinking in the result (text output: before the answer, between <think> tags)
    #[arg(long)]
    show_thinking: bool,
    /// Sampling temperature (0 = greedy). Default 0.7, or 1.0 when thinking
    #[arg(long)]
    temp: Option<f32>,
    /// Top-k (0 = off)
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    /// Top-p. Default 0.8, or 0.95 when thinking
    #[arg(long)]
    top_p: Option<f32>,
    /// Min-p
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// Presence penalty (subtracted from the logit of every token already generated)
    #[arg(long, default_value_t = 0.0)]
    presence_penalty: f32,
    /// Repetition penalty (1 = off)
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    /// DRY loop guard: penalty scale for a token that would extend a repeat of the answer's
    /// own earlier text (0 = off; 0.8 is typical). Leaves short recurring pieces alone
    #[arg(long, default_value_t = 0.0)]
    dry_multiplier: f32,
    /// DRY: growth of the penalty per token of repeat past the allowed length
    #[arg(long, default_value_t = 1.75)]
    dry_base: f32,
    /// DRY: repeats up to this many tokens are free
    #[arg(long, default_value_t = 2)]
    dry_allowed_length: usize,
    /// Random seed
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Requests decoded together (1-8); fewer when GPU memory runs short
    #[arg(long, default_value_t = 8)]
    parallel: usize,
    /// Most vision tokens per image (one per 32x32 pixels; larger images are scaled down)
    #[arg(long, default_value_t = 2048)]
    max_image_tokens: usize,
    /// Fewest vision tokens per image (smaller images are scaled up)
    #[arg(long, default_value_t = 16)]
    min_image_tokens: usize,
    /// Model: .gguf path, Hugging Face publisher/repo[/file.gguf], or a name in the model folders
    #[arg(long)]
    model: Option<String>,
    /// Vision projector (.gguf); `none` for text only
    #[arg(long)]
    mmproj: Option<String>,
    /// Context limit per request (prompt + answer)
    #[arg(long, default_value_t = 16384)]
    ctx: usize,
    /// Print the final prompt (images as their token counts) and exit
    #[arg(long)]
    show_prompt: bool,
    /// Serve JSON-lines requests on stdin, keeping the model loaded (see CONVENTIONS.md)
    #[arg(long)]
    worker: bool,
    /// No progress on stderr
    #[arg(short = 'q', long)]
    quiet: bool,
}

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"];

fn is_image(p: &str) -> bool {
    Path::new(p).extension().map(|e| IMAGE_EXTS.contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false)
}

fn read_text(spec: &str) -> Result<String> {
    if spec == "-" {
        worker::forbid_stdin()?;
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        return Ok(s);
    }
    std::fs::read_to_string(spec).with_context(|| format!("reading {spec}"))
}

fn params(cli: &Cli) -> Result<Params> {
    let think = Think::parse(&cli.think).with_context(|| format!("--think {}: want off, low, medium or high", cli.think))?;
    let thinking = think != Think::Off;
    ensure!(cli.max_tokens >= 1, "--max-tokens must be at least 1");
    ensure!(cli.max_image_tokens >= cli.min_image_tokens.max(4), "--max-image-tokens must be >= --min-image-tokens");
    Ok(Params {
        max_tokens: cli.max_tokens,
        think,
        sampling: Sampling {
            temperature: cli.temp.unwrap_or(if thinking { 1.0 } else { 0.7 }),
            top_k: cli.top_k,
            top_p: cli.top_p.unwrap_or(if thinking { 0.95 } else { 0.8 }),
            min_p: cli.min_p,
            presence_penalty: cli.presence_penalty,
            repeat_penalty: cli.repeat_penalty,
            dry_multiplier: cli.dry_multiplier,
            dry_base: cli.dry_base,
            dry_allowed_length: cli.dry_allowed_length.max(1),
            seed: cli.seed,
        },
        min_image_tokens: cli.min_image_tokens,
        max_image_tokens: cli.max_image_tokens,
    })
}

/// A --batch line -> Job (missing fields from the command line).
fn batch_job(line: &str, index: usize, base: &Job) -> Result<Job> {
    let v: serde_json::Value = serde_json::from_str(line).with_context(|| format!("batch line {}: not JSON", index + 1))?;
    let mut j = base.clone();
    j.index = index;
    j.id = v.get("id").map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()));
    if let Some(p) = v.get("prompt").and_then(|x| x.as_str()) {
        j.prompt = p.to_string();
    }
    if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
        j.prompt = format!("{}\n\n{}", j.prompt.trim(), t);
    }
    if let Some(s) = v.get("system").and_then(|x| x.as_str()) {
        j.system = Some(s.to_string());
    }
    let mut imgs: Vec<PathBuf> = Vec::new();
    if let Some(i) = v.get("image").and_then(|x| x.as_str()) {
        imgs.push(PathBuf::from(i));
    }
    if let Some(a) = v.get("images").and_then(|x| x.as_array()) {
        imgs.extend(a.iter().filter_map(|x| x.as_str()).map(PathBuf::from));
    }
    j.input = imgs.first().map(|p| p.display().to_string());
    j.images = imgs;
    if let Some(n) = v.get("max_tokens").and_then(|x| x.as_u64()) {
        j.params.max_tokens = n.max(1) as usize;
    }
    if let Some(t) = v.get("think").and_then(|x| x.as_str()) {
        j.params.think = Think::parse(t).with_context(|| format!("batch line {}: think {t:?}", index + 1))?;
    }
    if let Some(t) = v.get("temperature").and_then(|x| x.as_f64()) {
        j.params.sampling.temperature = t as f32;
    }
    if let Some(s) = v.get("seed").and_then(|x| x.as_u64()) {
        j.params.sampling.seed = s;
    }
    ensure!(!j.prompt.trim().is_empty() || !j.images.is_empty(), "batch line {}: no prompt and no images", index + 1);
    if j.prompt.trim().is_empty() {
        j.prompt = DEFAULT_IMAGE_PROMPT.to_string();
    }
    Ok(j)
}

const DEFAULT_IMAGE_PROMPT: &str = "Describe this image in detail.";

fn main() {
    if let Err(e) = run(std::env::args().collect()) {
        let msg = format!("{e:#}");
        eprintln!("loki-scribe: error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications (LM Studio or Ollama keep models in VRAM), or lower --parallel / --max-image-tokens / --ctx.");
        }
        std::process::exit(1);
    }
}

fn model_key(model: &Path, mmproj: Option<&Path>, ctx: usize) -> String {
    format!("scribe:{}|{}|{ctx}", model.display(), mmproj.map(|p| p.display().to_string()).unwrap_or_default())
}

fn run(argv: Vec<String>) -> Result<()> {
    let Some(cli) = worker::parse_args::<Cli>(Cli::command(), argv)? else { return Ok(()) };
    if cli.worker {
        ensure!(!worker::active(), "--worker inside a worker request");
        loki_scribe::log::set_quiet(cli.quiet);
        let base = cli.clone();
        return worker::serve(&worker::Tool {
            name: "loki-scribe",
            version: env!("CARGO_PKG_VERSION"),
            model: "Qwen3.8",
            run: &|argv| run(argv),
            // {"cmd": "warm", "model": NAME} loads a model (default: the command line's)
            warm: &|req| {
                let spec = req.get("model").and_then(|m| m.as_str()).map(str::to_string).or(base.model.clone());
                let model = models::resolve_model(spec.as_deref())?;
                let mmproj = models::resolve_mmproj(base.mmproj.as_deref(), &model)?;
                let key = model_key(&model, mmproj.as_deref(), base.ctx);
                worker::load(&key, &model, || Engine::load(&model, mmproj.as_deref(), base.ctx)).map(|_| ())
            },
            custom: None,
        });
    }
    if !worker::active() {
        loki_scribe::log::set_quiet(cli.quiet);
    }
    let params = params(&cli)?;
    let mut prompt = match (&cli.prompt, &cli.prompt_file) {
        (Some(_), Some(_)) => bail!("give --prompt or --prompt-file, not both"),
        (Some(p), None) => p.clone(),
        (None, Some(f)) => read_text(f)?,
        (None, None) => String::new(),
    };
    let system = match (&cli.system, &cli.system_file) {
        (Some(s), _) => Some(s.clone()),
        (None, Some(f)) => Some(std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?),
        _ => None,
    };
    // inputs: images, text files
    let mut images: Vec<String> = Vec::new();
    for inp in &cli.inputs {
        if inp != "-" && is_image(inp) {
            ensure!(Path::new(inp).is_file(), "{inp}: no such file");
            images.push(inp.clone());
        } else if inp == "-" {
            worker::forbid_stdin()?;
            let mut b = Vec::new();
            std::io::stdin().read_to_end(&mut b)?;
            if image::guess_format(&b).is_ok() {
                let tmp = std::env::temp_dir().join(format!("loki-scribe-stdin-{}.img", std::process::id()));
                std::fs::write(&tmp, &b)?;
                images.push(tmp.display().to_string());
            } else {
                prompt = format!("{}\n\n{}", prompt.trim(), String::from_utf8_lossy(&b));
            }
        } else {
            prompt = format!("{}\n\n{}", prompt.trim(), read_text(inp)?);
        }
    }
    if prompt.trim().is_empty() {
        ensure!(!images.is_empty() || cli.batch.is_some(), "nothing to do: give a prompt (-p), inputs or --batch (see --help)");
        if !images.is_empty() {
            prompt = DEFAULT_IMAGE_PROMPT.to_string();
        }
    }
    let base = Job { index: 0, id: None, system: system.clone(), prompt: prompt.trim().to_string(), images: Vec::new(), input: None, params: params.clone(), error: None };
    let model = models::resolve_model(cli.model.as_deref())?;
    let mmproj = models::resolve_mmproj(cli.mmproj.as_deref(), &model)?;

    // the jobs
    let positional = cli.batch.is_none();
    let jobs: Box<dyn Iterator<Item = Job> + Send> = if let Some(b) = &cli.batch {
        ensure!(cli.inputs.is_empty(), "--batch takes no positional inputs");
        let reader: Box<dyn BufRead + Send> = if b == "-" {
            worker::forbid_stdin()?;
            Box::new(std::io::BufReader::new(std::io::stdin()))
        } else {
            Box::new(std::io::BufReader::new(std::fs::File::open(b).with_context(|| format!("opening {b}"))?))
        };
        let base = base.clone();
        let mut idx = 0usize;
        Box::new(reader.lines().map_while(|l| l.ok()).filter(|l| !l.trim().is_empty()).map(move |l| {
            let i = idx;
            idx += 1;
            batch_job(&l, i, &base).unwrap_or_else(|e| {
                let mut j = base.clone();
                j.index = i;
                j.id = Some(format!("line {}", i + 1));
                j.error = Some(format!("{e:#}"));
                j
            })
        }))
    } else if cli.together || images.is_empty() {
        let mut j = base.clone();
        j.images = images.iter().map(PathBuf::from).collect();
        j.input = if images.len() == 1 { Some(images[0].clone()) } else { None };
        Box::new(std::iter::once(j))
    } else {
        let list: Vec<Job> = images
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut j = base.clone();
                j.index = i;
                j.images = vec![PathBuf::from(p)];
                j.input = Some(p.clone());
                j
            })
            .collect();
        Box::new(list.into_iter())
    };
    let n_known = if positional { if cli.together || images.is_empty() { 1 } else { images.len() } } else { usize::MAX };

    if cli.show_prompt {
        let g = loki_scribe::gguf::Gguf::open(&model)?;
        let tok = loki_scribe::tokenizer::Tokenizer::from_gguf(&g)?;
        for j in jobs {
            let imgs = j.images.iter().map(|p| engine::load_image(p, &j.params)).collect::<Result<Vec<_>>>()?;
            let counts: Vec<usize> = imgs.iter().map(|i| i.tokens()).collect();
            let pieces = chat::prompt(&tok, j.system.as_deref(), imgs.len(), &j.prompt, j.params.think)?;
            worker::out_line(&chat::render(&tok, &pieces, &counts));
        }
        return Ok(());
    }

    let key = model_key(&model, mmproj.as_deref(), cli.ctx);
    let eng = worker::load(&key, &model, || Engine::load(&model, mmproj.as_deref(), cli.ctx))?;
    let model_name = if eng.name.is_empty() { model.file_stem().unwrap_or_default().to_string_lossy().into_owned() } else { eng.name.clone() };

    // output
    let out_dir = cli.out.as_deref().filter(|o| *o != "-").map(PathBuf::from).filter(|p| p.is_dir() || (p.extension().is_none() && n_known > 1));
    let mut out_file = match cli.out.as_deref() {
        Some(o) if o != "-" && out_dir.is_none() => Some(std::fs::File::create(o).with_context(|| format!("creating {o}"))?),
        _ => None,
    };
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d)?;
    }
    let multi = n_known > 1;
    let mut failed = 0usize;
    let mut buffered: std::collections::BTreeMap<usize, Output> = Default::default();
    let mut next = 0usize;
    let show_thinking = cli.show_thinking;
    let json_out = cli.json;
    let json_line = |o: &Output| -> String {
        let mut v = json!({"index": o.index, "id": o.id, "input": o.input, "text": o.text, "finish": o.finish, "prompt_tokens": o.prompt_tokens, "tokens": o.tokens, "secs": (o.secs * 100.0).round() / 100.0, "model": model_name});
        if show_thinking {
            v["thinking"] = json!(o.thinking);
        }
        if let Some(e) = &o.error {
            v["error"] = json!(e);
        }
        v.to_string()
    };
    let mut emit = |o: Output, out_file: &mut Option<std::fs::File>| -> Result<()> {
        use std::io::Write;
        if o.error.is_some() {
            failed += 1;
        }
        let text = if show_thinking && o.thinking.is_some() { format!("<think>\n{}\n</think>\n\n{}", o.thinking.as_deref().unwrap_or(""), o.text) } else { o.text.clone() };
        if let Some(d) = &out_dir {
            let stem = o.input.as_deref().map(|i| Path::new(i).file_stem().unwrap_or_default().to_string_lossy().into_owned()).or(o.id.clone()).unwrap_or_else(|| format!("result-{}", o.index));
            let p = d.join(format!("{stem}.txt"));
            if let Some(e) = &o.error {
                eprintln!("loki-scribe: error: {}: {e}", o.input.as_deref().unwrap_or("request"));
                return Ok(());
            }
            std::fs::write(&p, &text)?;
            worker::out_line(&p.display().to_string());
            return Ok(());
        }
        let line = if json_out {
            json_line(&o)
        } else {
            if let Some(e) = &o.error {
                eprintln!("loki-scribe: error: {}: {e}", o.input.as_deref().or(o.id.as_deref()).unwrap_or("request"));
                return Ok(());
            }
            if multi {
                format!("### {}\n{}\n", o.input.as_deref().or(o.id.as_deref()).unwrap_or("result"), text)
            } else {
                text
            }
        };
        match out_file {
            Some(f) => writeln!(f, "{line}")?,
            None => worker::out_line(&line),
        }
        Ok(())
    };
    // A worker supervisor gets each --json result as soon as its sequence ends (batched sequences finish at different
    // times); the ordered lines below still follow at the end of the run.
    let stream_partials = json_out && out_dir.is_none() && out_file.is_none() && worker::active();
    let res = eng.run(jobs, cli.parallel, &mut |o: Output| {
        if stream_partials {
            worker::partial_line(&json_line(&o));
        }
        if positional {
            buffered.insert(o.index, o);
            while let Some(o) = buffered.remove(&next) {
                emit(o, &mut out_file)?;
                next += 1;
            }
            Ok(())
        } else {
            emit(o, &mut out_file)
        }
    });
    for (_, o) in std::mem::take(&mut buffered) {
        emit(o, &mut out_file)?;
    }
    eng.model.dev.trim_pool();
    res?;
    if failed > 0 && !cli.json {
        bail!("{failed} request(s) failed");
    }
    Ok(())
}
