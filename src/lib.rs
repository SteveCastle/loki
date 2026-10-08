pub mod cuda;
pub mod image;
pub mod ops;
pub mod safetensors;
pub mod tensor;
pub mod tokenizer;
pub mod weights;

// H3 engine components (one workstream each)
pub mod dit_h3;
pub mod fit;
pub mod media;
pub mod models;
pub mod pipeline;
pub mod presets;
pub mod sampler;
pub mod te_h3;
pub mod vae_audio;
pub mod vae_video;

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

/// CPU reference helpers shared by tests.
pub mod reference {
    /// Normalized regular Hadamard H256 = kron^4(h4) / 16, applied to each 256-group of a row.
    pub fn hadamard256_rows(x: &[f32], k: usize) -> Vec<f32> {
        let h4 = [[1.0f32, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0]];
        // build H256 explicitly
        let mut h = vec![vec![1.0f32]];
        let mut size = 1;
        while size < 256 {
            let mut nh = vec![vec![0.0f32; size * 4]; size * 4];
            for i in 0..size * 4 {
                for j in 0..size * 4 {
                    nh[i][j] = h[i / 4][j / 4] * h4[i % 4][j % 4];
                }
            }
            h = nh;
            size *= 4;
        }
        let rows = x.len() / k;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            for g in 0..k / 256 {
                let base = r * k + g * 256;
                for j in 0..256 {
                    let mut s = 0.0f64;
                    for i in 0..256 {
                        s += (x[base + i] as f64) * (h[i][j] as f64);
                    }
                    out[base + j] = (s / 16.0) as f32;
                }
            }
        }
        out
    }
    pub fn quant_rows_i8(x: &[f32], k: usize) -> (Vec<i8>, Vec<f32>) {
        let rows = x.len() / k;
        let mut q = vec![0i8; x.len()];
        let mut s = vec![0f32; rows];
        for r in 0..rows {
            let amax = x[r * k..(r + 1) * k].iter().fold(0f32, |a, v| a.max(v.abs()));
            let sc = amax / 127.0;
            s[r] = if sc > 0.0 { sc } else { f32::MIN_POSITIVE };
            for i in 0..k {
                let v = if sc > 0.0 { (x[r * k + i] / sc).round_ties_even() } else { 0.0 };
                q[r * k + i] = v.clamp(-128.0, 127.0) as i8;
            }
        }
        (q, s)
    }
}
