pub mod cli;
pub mod cuda;
pub mod detector;
pub mod genmeta;
pub mod everything;
pub mod image;
pub mod interactive;
pub mod media;
pub mod model;
pub mod models;
pub mod ops;
pub mod safetensors;
pub mod serve;
pub mod tensor;
pub mod text;
pub mod tracker;
pub mod video;
pub mod vit;
pub mod weights;
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

/// Reference-dump I/O shared by the check tools: `<dir>/index.json` {name: shape} + `<dir>/<name>.bin` (f32 LE).
pub mod dump {
    use anyhow::{Context, Result};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    pub struct Dump {
        pub dir: PathBuf,
        pub index: BTreeMap<String, Vec<usize>>,
    }
    impl Dump {
        pub fn open(dir: &Path) -> Result<Dump> {
            let s = std::fs::read_to_string(dir.join("index.json")).with_context(|| format!("reading {}/index.json", dir.display()))?;
            let index: BTreeMap<String, Vec<usize>> = serde_json::from_str(&s)?;
            Ok(Dump { dir: dir.to_path_buf(), index })
        }
        pub fn create(dir: &Path) -> Result<Dump> {
            std::fs::create_dir_all(dir)?;
            Ok(Dump { dir: dir.to_path_buf(), index: BTreeMap::new() })
        }
        pub fn read(&self, name: &str) -> Result<Vec<f32>> {
            let b = std::fs::read(self.dir.join(format!("{name}.bin"))).with_context(|| format!("reading dump {name}"))?;
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }
        pub fn write(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(self.dir.join(format!("{name}.bin")), bytes)?;
            self.index.insert(name.to_string(), shape.to_vec());
            std::fs::write(self.dir.join("index.json"), serde_json::to_string_pretty(&self.index)?)?;
            Ok(())
        }
    }

    /// (cosine, relative L2 error, max abs diff)
    pub fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
        let (mut ab, mut aa, mut bb, mut dd, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            ab += x * y;
            aa += x * x;
            bb += y * y;
            dd += (x - y) * (x - y);
            mx = mx.max((x - y).abs());
        }
        (ab / (aa.sqrt() * bb.sqrt() + 1e-30), dd.sqrt() / (aa.sqrt() + 1e-30), mx)
    }
}
