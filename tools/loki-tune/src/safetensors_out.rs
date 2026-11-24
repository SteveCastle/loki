//! Writing safetensors files (LoRAs, optimizer states, caches), and reading a file's `__metadata__` alone.
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

#[derive(Default)]
pub struct Out {
    entries: Vec<(String, String, Vec<usize>, Vec<u8>)>,
}

impl Out {
    pub fn add(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) {
        self.entries.push((name.to_string(), dtype.to_string(), shape.to_vec(), bytes));
    }
}

/// Write `out` to `path` atomically (a `.part` file renamed into place), tensors sorted by name.
pub fn write(path: &Path, out: &Out, meta: &BTreeMap<String, String>) -> Result<()> {
    let mut idx: Vec<usize> = (0..out.entries.len()).collect();
    idx.sort_by(|&a, &b| out.entries[a].0.cmp(&out.entries[b].0));
    let mut header = serde_json::Map::new();
    if !meta.is_empty() {
        header.insert("__metadata__".into(), serde_json::to_value(meta)?);
    }
    let mut off = 0usize;
    for &i in &idx {
        let (name, dtype, shape, bytes) = &out.entries[i];
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let part = path.with_extension("safetensors.part");
    {
        let f = std::fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;
        let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
        w.write_all(&(h.len() as u64).to_le_bytes())?;
        w.write_all(&h)?;
        for &i in &idx {
            w.write_all(&out.entries[i].3)?;
        }
        w.flush()?;
    }
    std::fs::rename(&part, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// The `__metadata__` of a safetensors file (header only).
pub fn read_metadata(path: &Path) -> Result<BTreeMap<String, String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n) as usize;
    if n > 100 << 20 {
        bail!("{} is not a safetensors file", path.display());
    }
    let mut h = vec![0u8; n];
    f.read_exact(&mut h)?;
    let v: serde_json::Value = serde_json::from_slice(&h).with_context(|| format!("{}: bad safetensors header", path.display()))?;
    let mut out = BTreeMap::new();
    if let Some(m) = v.get("__metadata__").and_then(|m| m.as_object()) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(out)
}
