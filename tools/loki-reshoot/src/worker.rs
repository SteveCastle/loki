//! `--worker`: the engine as a resident process (CONVENTIONS.md, "Worker mode").
//!
//! Requests arrive as JSON lines on stdin, each one a command line for this tool. Models stay on the GPU between
//! requests, so a warm request skips the minutes a cold run spends loading; when a load would not fit next to the
//! resident models, the least recently used ones that the current request is not holding are evicted first. stdout
//! carries protocol events only (a run's own result lines become `stdout` events); stderr carries the progress log
//! exactly as on the command line and ends every request with a sentinel line, so a supervisor can attribute it.
//! Every crate of the toolkit carries this same file; only `RESERVE_MB` differs.
use crate::cuda::Device;
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::any::Any;
use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// VRAM left free next to the resident models for a run's own activations (`LOKI_WORKER_RESERVE_MB` overrides).
/// A run that still runs out of memory is retried once with every idle model evicted.
const RESERVE_MB: usize = 5000;

/// Version of the stdio protocol (bumped only on incompatible changes).
pub const PROTOCOL: u32 = 1;

/// The stderr line that closes each request's log: `<SENTINEL> <id>`.
pub const SENTINEL: &str = "\u{1e}loki-worker done";

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CANCEL: AtomicBool = AtomicBool::new(false);
static CURRENT: Mutex<String> = Mutex::new(String::new());
/// The current run's result lines, sent when it ends (a retried attempt's lines are dropped).
static OUT: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether this process runs as a worker (`--worker`).
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Fails once the supervisor cancelled the current request; sampling loops call it every step.
pub fn check_cancel() -> Result<()> {
    if CANCEL.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    Ok(())
}

/// stdin carries the requests in worker mode, so no input can come from it.
pub fn forbid_stdin() -> Result<()> {
    if active() {
        bail!("stdin input (`-`) is not available in --worker mode; pass a file");
    }
    Ok(())
}

/// A result line: printed on stdout from the command line, a `stdout` event in worker mode.
pub fn out_line(s: &str) {
    if active() {
        if let Ok(mut o) = OUT.lock() {
            o.push(s.to_string());
        }
    } else {
        println!("{s}");
    }
}

/// Parses a command line. On the command line clap prints help or usage and exits as usual; in worker mode help and
/// version become result lines and a usage error becomes the request's error. `None`: nothing left to run.
pub fn parse_args<C: clap::FromArgMatches>(cmd: clap::Command, argv: Vec<String>) -> Result<Option<C>> {
    if !active() {
        return Ok(Some(C::from_arg_matches(&cmd.get_matches_from(argv))?));
    }
    match cmd.try_get_matches_from(argv) {
        Ok(m) => Ok(Some(C::from_arg_matches(&m)?)),
        Err(e) if matches!(e.kind(), clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion) => {
            for l in e.to_string().lines() {
                out_line(l);
            }
            Ok(None)
        }
        Err(e) => bail!("{}", e.to_string().trim().trim_start_matches("error: ")),
    }
}

fn emit(v: &Value) {
    let mut so = std::io::stdout().lock();
    let _ = writeln!(so, "{v}");
    let _ = so.flush();
}

// ---- residency

struct Entry {
    key: String,
    bytes: usize,
    used: u64,
    val: Rc<dyn Any>,
}

#[derive(Default)]
struct Cache {
    dev: Option<Arc<Device>>,
    entries: Vec<Entry>,
    tick: u64,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

/// The GPU: a fresh context on the command line, one context for the life of a worker.
pub fn device() -> Result<Arc<Device>> {
    if !active() {
        return Device::new(0);
    }
    if let Some(d) = CACHE.with(|c| c.borrow().dev.clone()) {
        return Ok(d);
    }
    let d = Device::new(0)?;
    CACHE.with(|c| c.borrow_mut().dev = Some(d.clone()));
    Ok(d)
}

fn reserve() -> usize {
    std::env::var("LOKI_WORKER_RESERVE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(RESERVE_MB) << 20
}

fn is_oom(e: &anyhow::Error) -> bool {
    let m = format!("{e:#}");
    m.contains("OUT_OF_MEMORY") || m.contains("out of memory") || m.contains("allocating")
}

/// Loads a model through the residency cache. `key` names the model and everything its loaded form depends on;
/// `file` is the weights file (its size estimates the VRAM the load needs). On the command line this is just `load()`:
/// the caller owns the model and dropping it frees it, as before.
pub fn load<T: 'static>(key: &str, file: &Path, mut load: impl FnMut() -> Result<T>) -> Result<Rc<T>> {
    if !active() {
        return Ok(Rc::new(load()?));
    }
    let hit = CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.tick += 1;
        let t = c.tick;
        c.entries.iter_mut().find(|e| e.key == key).map(|e| {
            e.used = t;
            e.val.clone()
        })
    });
    if let Some(v) = hit {
        if let Ok(v) = v.downcast::<T>() {
            crate::info!("  {key}: resident");
            return Ok(v);
        }
    }
    let need = std::fs::metadata(file).map(|m| m.len() as usize).unwrap_or(0);
    make_room(need + reserve())?;
    let dev = device()?;
    let before = dev.free_mem().unwrap_or(0);
    let v = match load() {
        Ok(v) => v,
        Err(e) if is_oom(&e) && evict_unused() > 0 => load()?,
        Err(e) => return Err(e),
    };
    dev.sync()?;
    dev.trim_pool();
    let after = dev.free_mem().unwrap_or(0);
    let rc = Rc::new(v);
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let used = c.tick;
        c.entries.push(Entry { key: key.to_string(), bytes: before.saturating_sub(after).max(need), used, val: rc.clone() });
    });
    Ok(rc)
}

/// Evicts least recently used models nobody holds until `want` bytes are free (or nothing more can go).
fn make_room(want: usize) -> Result<()> {
    let dev = device()?;
    loop {
        dev.trim_pool();
        if dev.free_mem()? >= want {
            return Ok(());
        }
        let victim = CACHE.with(|c| {
            let mut c = c.borrow_mut();
            let i = c.entries.iter().enumerate().filter(|(_, e)| Rc::strong_count(&e.val) == 1).min_by_key(|(_, e)| e.used).map(|(i, _)| i);
            i.map(|i| c.entries.remove(i))
        });
        match victim {
            Some(e) => {
                crate::info!("  evicting {} ({} MB) to make room", e.key, e.bytes >> 20);
                drop(e);
                dev.sync()?;
            }
            None => return Ok(()),
        }
    }
}

/// Drops every resident model nobody holds; returns how many went.
pub fn evict_unused() -> usize {
    let gone: Vec<Entry> = CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let (keep, gone): (Vec<Entry>, Vec<Entry>) = c.entries.drain(..).partition(|e| Rc::strong_count(&e.val) > 1);
        c.entries = keep;
        gone
    });
    let n = gone.len();
    drop(gone);
    if let Some(d) = CACHE.with(|c| c.borrow().dev.clone()) {
        let _ = d.sync();
        d.trim_pool();
    }
    n
}

fn resident() -> Value {
    CACHE.with(|c| Value::Array(c.borrow().entries.iter().map(|e| json!({"key": e.key, "mb": e.bytes >> 20})).collect()))
}

fn gpu() -> Value {
    match CACHE.with(|c| c.borrow().dev.clone()) {
        Some(d) => {
            d.trim_pool();
            json!({"total_mb": d.total_mem >> 20, "free_mb": d.free_mem().unwrap_or(0) >> 20})
        }
        None => Value::Null,
    }
}

// ---- the request loop

/// Hooks of one tool: `run` executes a command line (argv[0] is the tool), `warm` fetches and loads its models.
pub struct Tool<'a> {
    pub name: &'a str,
    pub version: &'a str,
    pub model: &'a str,
    pub run: &'a dyn Fn(Vec<String>) -> Result<()>,
    pub warm: &'a dyn Fn(&Value) -> Result<()>,
    /// Tool-specific commands (`cmd` other than run/warm/evict/status); the value returned is the done event's "reply".
    pub custom: Option<&'a dyn Fn(&str, &Value) -> Result<Value>>,
}

/// Serves requests until stdin closes.
pub fn serve(tool: &Tool) -> Result<()> {
    ACTIVE.store(true, Ordering::Relaxed);
    device()?;
    emit(&json!({"event": "ready", "tool": tool.name, "version": tool.version, "model": tool.model, "protocol": PROTOCOL, "gpu": gpu()}));
    let (tx, rx) = std::sync::mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(v) => {
                    // a cancel acts at once, while the request it names is still running
                    if let Some(id) = v.get("cancel").and_then(|c| c.as_str()) {
                        if CURRENT.lock().map(|c| *c == id).unwrap_or(false) {
                            CANCEL.store(true, Ordering::Relaxed);
                        }
                        continue;
                    }
                    if tx.send(v).is_err() {
                        break;
                    }
                }
                Err(e) => emit(&json!({"event": "error", "error": format!("bad JSON: {e}")})),
            }
        }
    });
    for req in rx {
        let id = req.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let t = Instant::now();
        let mut reply = Value::Null;
        let res: Result<()> = match req.get("cmd").and_then(|v| v.as_str()) {
            None | Some("run") => {
                let args = req.get("args").and_then(|a| a.as_array()).map(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect::<Option<Vec<String>>>());
                match args {
                    Some(Some(args)) => {
                        if let Ok(mut c) = CURRENT.lock() {
                            *c = id.clone();
                        }
                        CANCEL.store(false, Ordering::Relaxed);
                        let argv: Vec<String> = std::iter::once(tool.name.to_string()).chain(args).collect();
                        let mut r = (tool.run)(argv.clone());
                        if matches!(&r, Err(e) if is_oom(e)) && !CANCEL.load(Ordering::Relaxed) && evict_unused() > 0 {
                            crate::info!("{}: out of GPU memory next to the resident models; evicted them, retrying", tool.name);
                            if let Ok(mut o) = OUT.lock() {
                                o.clear();
                            }
                            r = (tool.run)(argv);
                        }
                        r
                    }
                    _ => Err(anyhow!("a run needs \"args\": [string, ...]")),
                }
            }
            Some("warm") => {
                if let Ok(mut c) = CURRENT.lock() {
                    *c = id.clone();
                }
                (tool.warm)(&req)
            }
            Some("evict") => {
                let n = evict_unused();
                crate::info!("evicted {n} model(s)");
                Ok(())
            }
            Some("status") => Ok(()),
            Some(other) => match tool.custom {
                Some(f) => {
                    if let Ok(mut c) = CURRENT.lock() {
                        *c = id.clone();
                    }
                    f(other, &req).map(|v| reply = v)
                }
                None => Err(anyhow!("unknown cmd {other:?} (run, warm, evict, status)")),
            },
        };
        let cancelled = CANCEL.swap(false, Ordering::Relaxed);
        for line in OUT.lock().map(|mut o| std::mem::take(&mut *o)).unwrap_or_default() {
            emit(&json!({"id": id, "event": "stdout", "line": line}));
        }
        let (ok, error, oom) = match &res {
            Ok(()) => (true, Value::Null, false),
            Err(e) => {
                eprintln!("{}: error: {e:#}", tool.name);
                (false, Value::from(format!("{e:#}")), is_oom(e))
            }
        };
        // a failed kernel poisons the CUDA context for good: say so, then exit for the supervisor to restart us
        let healthy = CACHE.with(|c| c.borrow().dev.clone()).map(|d| d.sync().is_ok()).unwrap_or(true);
        eprintln!("{SENTINEL} {id}");
        let _ = std::io::stderr().flush();
        emit(&json!({"id": id, "event": "done", "ok": ok, "error": error, "oom": oom, "cancelled": cancelled, "reply": reply, "secs": (t.elapsed().as_secs_f64() * 100.0).round() / 100.0, "resident": resident(), "gpu": gpu()}));
        if let Ok(mut c) = CURRENT.lock() {
            c.clear();
        }
        if !healthy {
            eprintln!("{}: the GPU context failed; exiting so the supervisor restarts the engine", tool.name);
            std::process::exit(3);
        }
    }
    Ok(())
}
