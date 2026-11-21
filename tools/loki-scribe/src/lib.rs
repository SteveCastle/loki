//! loki-scribe: Qwen3.5 / Qwen3.8 ("qwen35" GGUF) multimodal text generation on a standalone CUDA engine.
pub mod chat;
pub mod cuda;
pub mod engine;
pub mod gguf;
pub mod model;
pub mod models;
pub mod ops;
pub mod quant;
pub mod tensor;
pub mod tokenizer;
pub mod vision;
pub mod worker;

/// Progress/diagnostic output (stderr), silenced by `--quiet`. Results go to stdout, never through this.
pub mod log {
    use std::sync::atomic::{AtomicBool, Ordering};
    static QUIET: AtomicBool = AtomicBool::new(false);
    pub fn set_quiet(q: bool) {
        QUIET.store(q, Ordering::Relaxed);
    }
    pub fn quiet() -> bool {
        QUIET.load(Ordering::Relaxed)
    }
}

#[macro_export]
macro_rules! info {
    ($($t:tt)*) => {
        if !$crate::log::quiet() {
            eprintln!($($t)*);
        }
    };
}
