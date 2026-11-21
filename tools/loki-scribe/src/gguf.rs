//! GGUF (v2/v3) reader: metadata key/values and tensor directory over a memory-mapped file.
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A metadata value.
#[derive(Clone, Debug)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// GGML tensor types this engine understands (others are refused with their name).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::BF16,
            o => GgmlType::Other(o),
        }
    }
    /// (elements per block, bytes per block) in the file.
    pub fn block(self) -> Result<(usize, usize)> {
        Ok(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
            GgmlType::Other(id) => bail!("unsupported GGML tensor type {} ({id})", ggml_type_name(id)),
        })
    }
    pub fn name(self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::Other(id) => ggml_type_name(id).into(),
        }
    }
}

fn ggml_type_name(id: u32) -> &'static str {
    match id {
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        _ => "unknown",
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: dims[0] is the innermost (contiguous) dimension.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// absolute byte offset in the file
    pub offset: usize,
    pub bytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    /// (rows, cols) of a 2-D weight in row-major terms: rows = dims[1..], cols = dims[0].
    pub fn rows_cols(&self) -> (usize, usize) {
        (self.dims[1..].iter().product(), self.dims[0])
    }
}

pub struct Gguf {
    pub path: PathBuf,
    map: Mmap,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "truncated GGUF header");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let at = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.val(at)?);
                }
                Value::Arr(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_le_bytes(self.take(8)?.try_into()?)),
            _ => bail!("bad GGUF value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mapping {}", path.display()))?;
        let mut r = Rd { b: &map, p: 0 };
        ensure!(r.take(4)? == b"GGUF", "{} is not a GGUF file", path.display());
        let ver = r.u32()?;
        ensure!(ver == 2 || ver == 3, "{}: GGUF version {ver} unsupported", path.display());
        let nt = r.u64()? as usize;
        let nk = r.u64()? as usize;
        let mut kv = HashMap::new();
        for _ in 0..nk {
            let k = r.str()?;
            let t = r.u32()?;
            let v = r.val(t)?;
            kv.insert(k, v);
        }
        let align = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let mut infos = Vec::with_capacity(nt);
        for _ in 0..nt {
            let name = r.str()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()? as usize);
            }
            let ty = GgmlType::from_id(r.u32()?);
            let off = r.u64()? as usize;
            infos.push((name, dims, ty, off));
        }
        let base = (r.p + align - 1) / align * align;
        let mut tensors = HashMap::new();
        for (name, dims, ty, off) in infos {
            let numel: usize = dims.iter().product();
            let bytes = match ty.block() {
                Ok((be, bb)) => numel / be * bb,
                Err(_) => 0,
            };
            let offset = base + off;
            ensure!(offset + bytes <= map.len(), "{}: tensor {name} runs past the end of the file (truncated download?)", path.display());
            tensors.insert(name.clone(), TensorInfo { name, dims, ty, offset, bytes });
        }
        Ok(Gguf { path: path.to_path_buf(), map, kv, tensors })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }
    pub fn u(&self, key: &str) -> Result<usize> {
        self.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).with_context(|| format!("{}: missing {key}", self.path.display()))
    }
    pub fn f(&self, key: &str) -> Result<f64> {
        self.get(key).and_then(|v| v.as_f64()).with_context(|| format!("{}: missing {key}", self.path.display()))
    }
    pub fn s(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("{}: missing tensor {name}", self.path.display()))
    }
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.map[t.offset..t.offset + t.bytes]
    }
    /// f32 copy of a float tensor (F32 / F16 / BF16).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let d = self.data(t);
        Ok(match t.ty {
            GgmlType::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            GgmlType::F16 => d.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            GgmlType::BF16 => d.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            _ => crate::quant::dequant_cpu(t.ty, d, t.numel())?,
        })
    }
    /// Total bytes of every tensor (a VRAM estimate).
    pub fn tensor_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.bytes).sum()
    }
}
