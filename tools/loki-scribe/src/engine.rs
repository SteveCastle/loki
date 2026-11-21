//! Request scheduling: continuous batching over the resident model.
//!
//! Requests arrive from an iterator (positional inputs, a JSONL batch, a worker request); a prefetch thread decodes
//! and resizes their images on the CPU while the GPU works. Up to `parallel` sequences are active at once: a new one
//! is admitted (vision encode + chunked prefill) whenever a slot is free, then every active sequence advances by one
//! token per decode step, all in one batched forward. Each finishes on its own (stop token or token limit) and is
//! reported through `on_done` at once.
use crate::chat::{self, Piece, Think};
use crate::gguf::Gguf;
use crate::model::{SeqState, TextModel, MAX_BATCH, PREFILL_CHUNK};
use crate::tokenizer::Tokenizer;
use crate::vision::{self, Prepared, Vision};
use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Prompts up to this many rows prefill through the GEMV path (8 rows at a time).
const SMALL_PROMPT: usize = 40;

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub max_tokens: usize,
    pub think: Think,
    pub sampling: Sampling,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
}

/// One request as given.
#[derive(Clone, Debug)]
pub struct Job {
    pub index: usize,
    pub id: Option<String>,
    pub system: Option<String>,
    pub prompt: String,
    pub images: Vec<PathBuf>,
    /// the input the result is reported against (a file name), if any
    pub input: Option<String>,
    pub params: Params,
    /// a request that could not be read (reported as a failed result)
    pub error: Option<String>,
}

/// A finished request.
#[derive(Clone, Debug)]
pub struct Output {
    pub index: usize,
    pub id: Option<String>,
    pub input: Option<String>,
    pub text: String,
    pub thinking: Option<String>,
    pub prompt_tokens: usize,
    pub tokens: usize,
    /// "stop" | "length" | "error"
    pub finish: &'static str,
    pub error: Option<String>,
    pub secs: f64,
}

struct Ready {
    job: Job,
    images: Result<Vec<Prepared>>,
}

struct Active {
    job: Job,
    state: SeqState,
    next_pos: i64,
    last: u32,
    generated: Vec<u32>,
    seen: std::collections::HashSet<u32>,
    prompt_tokens: usize,
    rng: u64,
    started: Instant,
}

/// Where a run's time went (reported on stderr at the end).
#[derive(Default)]
struct Stats {
    vision: f64,
    prefill: f64,
    prefill_rows: usize,
    decode: f64,
    steps: usize,
    tokens: usize,
}

pub struct Engine {
    stats: std::cell::RefCell<Stats>,
    pub model: TextModel,
    pub vision: Option<Vision>,
    pub tok: Tokenizer,
    pub name: String,
    stops: Vec<u32>,
    think_end: u32,
}

impl Engine {
    pub fn load(model_path: &Path, mmproj: Option<&Path>, max_ctx: usize) -> Result<Engine> {
        let dev = crate::worker::device()?;
        crate::ops::init(&dev)?;
        let t0 = Instant::now();
        let g = Arc::new(Gguf::open(model_path)?);
        let tok = Tokenizer::from_gguf(&g)?;
        let name = g.s("general.name").unwrap_or("").to_string();
        crate::info!("loading {} ({:.1} GB) ...", model_path.display(), std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0) as f64 / 1e9);
        let last = std::cell::Cell::new(0usize);
        let model = TextModel::load(dev.clone(), g.clone(), max_ctx, &|i, n| {
            let pct = i * 100 / n;
            if pct / 25 != last.get() / 25 && !crate::log::quiet() {
                eprintln!("  layers {i}/{n}");
            }
            last.set(pct);
        })?;
        let vision = match mmproj {
            Some(p) => {
                let vg = Gguf::open(p)?;
                Some(Vision::load(dev.clone(), &vg).with_context(|| format!("loading the vision projector {}", p.display()))?)
            }
            None => None,
        };
        dev.sync()?;
        crate::info!("loaded in {:.1}s ({} layers, {:.1} GB on the GPU{})", t0.elapsed().as_secs_f64(), model.cfg.n_layer, (model.vram_bytes + vision.as_ref().map(|v| v.vram_bytes).unwrap_or(0)) as f64 / 1e9, if vision.is_some() { ", vision on" } else { ", text only" });
        let mut stops = vec![tok.eos];
        for s in ["<|im_end|>", "<|endoftext|>"] {
            if let Ok(id) = tok.special(s) {
                if !stops.contains(&id) {
                    stops.push(id);
                }
            }
        }
        let think_end = tok.special("</think>")?;
        Ok(Engine { stats: Default::default(), model, vision, tok, name, stops, think_end })
    }

    /// The prompt pieces and per-image token counts of a job (images already prepared).
    pub fn build(&self, job: &Job, images: &[Prepared]) -> Result<Vec<Piece>> {
        if !images.is_empty() && self.vision.is_none() {
            bail!("this model has no vision projector (mmproj); it cannot read images");
        }
        chat::prompt(&self.tok, job.system.as_deref(), images.len(), &job.prompt, job.params.think)
    }

    /// Run jobs with up to `parallel` sequences at once; `on_done` gets each result as it finishes.
    pub fn run(&self, jobs: Box<dyn Iterator<Item = Job> + Send>, parallel: usize, on_done: &mut dyn FnMut(Output) -> Result<()>) -> Result<()> {
        let parallel = parallel.clamp(1, MAX_BATCH);
        // prefetch: decode + resize images ahead of the GPU
        let (tx, rx) = std::sync::mpsc::sync_channel::<Ready>(parallel * 2);
        std::thread::spawn(move || {
            for job in jobs {
                let images = match &job.error {
                    Some(e) => Err(anyhow::anyhow!("{e}")),
                    None => job.images.iter().map(|p| load_image(p, &job.params)).collect::<Result<Vec<_>>>(),
                };
                if tx.send(Ready { job, images }).is_err() {
                    break;
                }
            }
        });
        let mut pending: VecDeque<Ready> = VecDeque::new();
        let mut active: Vec<Active> = Vec::new();
        let mut inputs_done = false;
        let mut cap_limit = parallel;
        *self.stats.borrow_mut() = Stats::default();
        let t_run = Instant::now();
        loop {
            crate::worker::check_cancel()?;
            // collect what the prefetcher has; block only when there is nothing else to do
            while !inputs_done && pending.len() < parallel {
                let got = if active.is_empty() && pending.is_empty() {
                    rx.recv().ok()
                } else {
                    match rx.try_recv() {
                        Ok(x) => Some(x),
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
                    }
                };
                match got {
                    Some(x) => pending.push_back(x),
                    None => inputs_done = true,
                }
            }
            // admit
            while active.len() < cap_limit {
                let Some(r) = pending.pop_front() else { break };
                let index = r.job.index;
                match self.admit(r) {
                    Ok(Admit::Started(a)) => active.push(a),
                    Ok(Admit::Finished(o)) => on_done(o)?,
                    Ok(Admit::NoMemory(r)) => {
                        if active.is_empty() {
                            on_done(fail(&r.job, anyhow::anyhow!("out of GPU memory for this request's context (lower --max-tokens / --max-image-tokens)"), Instant::now()))?;
                        } else {
                            cap_limit = active.len();
                            crate::info!("GPU memory holds {cap_limit} sequence(s) at once; the rest wait");
                            pending.push_front(r);
                        }
                        break;
                    }
                    Err(e) => {
                        crate::info!("request {index}: {e:#}");
                        return Err(e);
                    }
                }
            }
            if active.is_empty() {
                if inputs_done && pending.is_empty() {
                    break;
                }
                continue;
            }
            // one decode step for every active sequence
            self.step(&mut active)?;
            let mut i = 0;
            while i < active.len() {
                if let Some(o) = self.finished(&active[i]) {
                    active.swap_remove(i);
                    on_done(o)?;
                } else {
                    i += 1;
                }
            }
        }
        crate::model::PROF.report();
        let st = self.stats.borrow();
        if st.steps > 0 {
            crate::info!(
                "done in {:.1}s: vision {:.1}s, prefill {:.1}s ({} tokens, {:.0}/s), decode {:.1}s ({} steps, {:.1} ms/step, {} tokens, {:.1} tokens/s)",
                t_run.elapsed().as_secs_f64(), st.vision, st.prefill, st.prefill_rows, st.prefill_rows as f64 / st.prefill.max(1e-9), st.decode, st.steps, st.decode * 1000.0 / st.steps as f64, st.tokens, st.tokens as f64 / st.decode.max(1e-9)
            );
        }
        Ok(())
    }

    fn admit(&self, r: Ready) -> Result<Admit> {
        let started = Instant::now();
        let job = r.job;
        let images = match r.images {
            Ok(i) => i,
            Err(e) => return Ok(Admit::Finished(fail(&job, e, started))),
        };
        let pieces = match self.build(&job, &images) {
            Ok(p) => p,
            Err(e) => return Ok(Admit::Finished(fail(&job, e, started))),
        };
        let e = self.model.cfg.n_embd;
        // embedding rows and M-RoPE positions
        let mut embeds: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        if let Some(v) = &self.vision {
            let t = Instant::now();
            for img in &images {
                embeds.push(v.encode(img)?);
            }
            self.stats.borrow_mut().vision += t.elapsed().as_secs_f64();
        }
        let n_rows: usize = pieces.iter().map(|p| if let Piece::Image(i) = p { images[*i].tokens() } else { 1 }).sum();
        let cap = n_rows + job.params.max_tokens + 1;
        if cap > self.model.max_ctx {
            return Ok(Admit::Finished(fail(&job, anyhow::anyhow!("prompt of {n_rows} tokens + {} new exceeds the context of {} (raise --ctx or lower --max-image-tokens)", job.params.max_tokens, self.model.max_ctx), started)));
        }
        let mut state = match SeqState::new(&self.model.dev, &self.model.cfg, cap) {
            Ok(s) => s,
            Err(_) => {
                self.model.dev.trim_pool();
                return Ok(Admit::NoMemory(Ready { job, images: Ok(images) }));
            }
        };
        let mut x = vec![0f32; n_rows * e];
        let mut pos: Vec<[i64; 3]> = Vec::with_capacity(n_rows);
        let mut p: i64 = 0;
        let mut row = 0;
        for piece in &pieces {
            match piece {
                Piece::Tok(t) => {
                    self.model.embed(*t, &mut x[row * e..(row + 1) * e])?;
                    pos.push([p, p, p]);
                    p += 1;
                    row += 1;
                }
                Piece::Image(i) => {
                    let (gy, gx) = images[*i].grid();
                    let emb = &embeds[*i];
                    x[row * e..(row + gy * gx) * e].copy_from_slice(&emb[..gy * gx * e]);
                    for yy in 0..gy {
                        for xx in 0..gx {
                            pos.push([p, p + yy as i64, p + xx as i64]);
                        }
                    }
                    row += gy * gx;
                    p += gy.max(gx) as i64;
                }
            }
        }
        // chunked prefill; the last chunk yields the first token's logits
        let mut logits = None;
        let mut r0 = 0;
        let t_pre = Instant::now();
        while r0 < n_rows {
            crate::worker::check_cancel()?;
            // a short prompt is cheaper as a few 8-row passes over the quantized weights than one dequantized GEMM pass
            let chunk = if n_rows <= SMALL_PROMPT { MAX_BATCH } else { PREFILL_CHUNK };
            let r1 = (r0 + chunk).min(n_rows);
            let last = r1 == n_rows;
            logits = self.model.forward(&x[r0 * e..r1 * e], &pos[r0..r1], &mut [&mut state], true, last)?;
            r0 = r1;
        }
        let logits = logits.context("prefill produced no logits")?;
        {
            let mut st = self.stats.borrow_mut();
            st.prefill += t_pre.elapsed().as_secs_f64();
            st.prefill_rows += n_rows;
        }
        if let Some(k) = std::env::var("LOKI_SCRIBE_DEBUG_TOP").ok().and_then(|v| v.parse::<usize>().ok()) {
            let lg = &logits[..self.model.output_n()];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = lg.iter().map(|v| ((v - mx) as f64).exp()).sum();
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap_or(std::cmp::Ordering::Equal));
            for &i in idx.iter().take(k) {
                eprintln!("  first token {:>7} p={:.4} {:?}", i, ((lg[i] - mx) as f64).exp() / z, self.tok.decode(&[i as u32]));
            }
        }
        let mut a = Active {
            rng: job.params.sampling.seed ^ (job.index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            job,
            state,
            next_pos: p,
            last: 0,
            generated: Vec::new(),
            seen: Default::default(),
            prompt_tokens: n_rows,
            started,
        };
        let t = sample(&logits[..self.model.output_n()], &a.job.params.sampling, &a.seen, &mut a.rng);
        a.push(t);
        crate::info!("[{}] prompt {} tokens{} -> generating", a.job.input.as_deref().or(a.job.id.as_deref()).unwrap_or("request"), n_rows, if images.is_empty() { String::new() } else { format!(" ({} image(s))", images.len()) });
        Ok(Admit::Started(a))
    }

    fn step(&self, active: &mut [Active]) -> Result<()> {
        let t_step = Instant::now();
        let e = self.model.cfg.n_embd;
        let live: Vec<usize> = (0..active.len()).filter(|&i| !self.is_done(&active[i])).collect();
        if live.is_empty() {
            return Ok(());
        }
        let mut x = vec![0f32; live.len() * e];
        let mut pos = Vec::with_capacity(live.len());
        for (r, &i) in live.iter().enumerate() {
            self.model.embed(active[i].last, &mut x[r * e..(r + 1) * e])?;
            let p = active[i].next_pos;
            pos.push([p, p, p]);
        }
        let logits = {
            let mut states: Vec<&mut SeqState> = Vec::with_capacity(live.len());
            // split borrows: collect &mut to each live sequence's state
            let mut rest: &mut [Active] = active;
            let mut base = 0;
            for &i in &live {
                let (head, tail) = rest.split_at_mut(i - base + 1);
                states.push(&mut head[i - base].state);
                rest = tail;
                base = i + 1;
            }
            self.model.forward(&x, &pos, &mut states, false, true)?.context("decode produced no logits")?
        };
        let v = self.model.output_n();
        for (r, &i) in live.iter().enumerate() {
            let a = &mut active[i];
            a.next_pos += 1;
            let t = sample(&logits[r * v..(r + 1) * v], &a.job.params.sampling, &a.seen, &mut a.rng);
            a.push(t);
        }
        let mut st = self.stats.borrow_mut();
        st.decode += t_step.elapsed().as_secs_f64();
        st.steps += 1;
        st.tokens += live.len();
        Ok(())
    }

    fn is_done(&self, a: &Active) -> bool {
        a.generated.last().map(|t| self.stops.contains(t)).unwrap_or(false) || a.generated.len() >= a.job.params.max_tokens || a.state.len + 1 >= a.state.cap
    }

    fn finished(&self, a: &Active) -> Option<Output> {
        if !self.is_done(a) {
            return None;
        }
        let stopped = a.generated.last().map(|t| self.stops.contains(t)).unwrap_or(false);
        let toks: Vec<u32> = a.generated.iter().copied().filter(|t| !self.stops.contains(t)).collect();
        let (thinking, answer) = match toks.iter().position(|t| *t == self.think_end) {
            Some(i) => (Some(self.tok.decode(&toks[..i]).trim().to_string()), &toks[i + 1..]),
            None if a.job.params.think != Think::Off => (Some(self.tok.decode(&toks).trim().to_string()), &toks[toks.len()..]),
            None => (None, &toks[..]),
        };
        let secs = a.started.elapsed().as_secs_f64();
        crate::info!("[{}] {} tokens in {:.1}s ({})", a.job.input.as_deref().or(a.job.id.as_deref()).unwrap_or("request"), a.generated.len(), secs, if stopped { "done" } else { "token limit" });
        Some(Output {
            index: a.job.index,
            id: a.job.id.clone(),
            input: a.job.input.clone(),
            text: self.tok.decode(answer).trim().to_string(),
            thinking: thinking.filter(|t| !t.is_empty()),
            prompt_tokens: a.prompt_tokens,
            tokens: a.generated.len(),
            finish: if stopped { "stop" } else { "length" },
            error: None,
            secs,
        })
    }
}

enum Admit {
    Started(Active),
    Finished(Output),
    NoMemory(Ready),
}

impl Active {
    fn push(&mut self, t: u32) {
        self.generated.push(t);
        self.seen.insert(t);
        self.last = t;
    }
}

impl TextModel {
    pub fn output_n(&self) -> usize {
        self.cfg.vocab
    }
}

fn fail(job: &Job, e: anyhow::Error, started: Instant) -> Output {
    Output { index: job.index, id: job.id.clone(), input: job.input.clone(), text: String::new(), thinking: None, prompt_tokens: 0, tokens: 0, finish: "error", error: Some(format!("{e:#}")), secs: started.elapsed().as_secs_f64() }
}

/// Load and prepare one image for the vision tower.
pub fn load_image(p: &Path, params: &Params) -> Result<Prepared> {
    let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    let img = image::load_from_memory(&bytes).with_context(|| format!("decoding {}", p.display()))?;
    Ok(vision::prepare(&img, params.min_image_tokens, params.max_image_tokens))
}

// ---------------------------------------------------------------- sampling

fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pick the next token: penalties, then greedy (temperature 0) or top-k / top-p / min-p sampling.
pub fn sample(logits: &[f32], s: &Sampling, seen: &std::collections::HashSet<u32>, rng: &mut u64) -> u32 {
    // penalties touch only the tokens generated so far
    let mut owned;
    let logits: &[f32] = if seen.is_empty() || (s.repeat_penalty == 1.0 && s.presence_penalty == 0.0) {
        logits
    } else {
        owned = logits.to_vec();
        for &t in seen {
            let l = &mut owned[t as usize];
            if s.repeat_penalty != 1.0 {
                *l = if *l > 0.0 { *l / s.repeat_penalty } else { *l * s.repeat_penalty };
            }
            *l -= s.presence_penalty;
        }
        &owned
    };
    let adj = |i: usize| logits[i];
    if s.temperature <= 0.0 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..logits.len() {
            let l = adj(i);
            if l > best.1 {
                best = (i, l);
            }
        }
        return best.0 as u32;
    }
    let k = if s.top_k == 0 { 256 } else { s.top_k.min(1024) };
    // top-k by partial selection
    let mut cand: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    let mut min_kept = f32::NEG_INFINITY;
    for i in 0..logits.len() {
        let l = adj(i);
        if cand.len() < k {
            cand.push((l, i as u32));
            if cand.len() == k {
                cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                min_kept = cand[k - 1].0;
            }
        } else if l > min_kept {
            let at = cand.partition_point(|c| c.0 >= l);
            cand.insert(at, (l, i as u32));
            cand.pop();
            min_kept = cand[k - 1].0;
        }
    }
    if cand.len() < k {
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mx = cand[0].0;
    let mut probs: Vec<f32> = cand.iter().map(|c| ((c.0 - mx) / s.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    // min-p relative to the best, then top-p
    let mut n = probs.len();
    if s.min_p > 0.0 {
        let thr = probs[0] * s.min_p;
        n = probs.iter().take_while(|p| **p >= thr).count().max(1);
    }
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, p) in probs.iter().take(n).enumerate() {
            acc += p;
            if acc >= s.top_p {
                n = i + 1;
                break;
            }
        }
    }
    let total: f32 = probs[..n].iter().sum();
    let r = (next_u64(rng) >> 11) as f64 / (1u64 << 53) as f64 * total as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += probs[i] as f64;
        if r < acc {
            return cand[i].1;
        }
    }
    cand[n - 1].1
}
