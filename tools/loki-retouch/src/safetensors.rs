//! Memory-mapped safetensors reader.
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

pub struct SafeTensors {
    mmap: Mmap,
    base: usize,
    pub tensors: HashMap<String, TensorInfo>,
    pub path: String,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors> {
        let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", path.display()))?;
        if mmap.len() < 8 {
            bail!("{} is not a safetensors file", path.display());
        }
        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..8 + n]).context("parsing safetensors header")?;
        let mut tensors = HashMap::new();
        for (k, v) in header {
            if k == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(v).with_context(|| format!("tensor header {k}"))?;
            tensors.insert(k, info);
        }
        Ok(SafeTensors { mmap, base: 8 + n, tensors, path: path.display().to_string() })
    }
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| anyhow!("tensor {name} not found in {}", self.path))
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        let (s, e) = info.data_offsets;
        Ok(&self.mmap[self.base + s..self.base + e])
    }
    /// Read a small tensor as f32 (BF16/F32/F16 supported).
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let b = self.bytes(name)?;
        Ok(match info.dtype.as_str() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "BF16" => b.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32()).collect(),
            "F16" => b.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32()).collect(),
            other => bail!("tensor {name}: unsupported dtype {other}"),
        })
    }
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data_offsets.1 - t.data_offsets.0).sum()
    }
}
