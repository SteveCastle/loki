//! Minimal device tensor: an owned (or borrowed) device allocation plus shape and dtype.
use crate::cuda::{DevBuf, Device};
use anyhow::Result;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    I8,
    U8,
}
impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
}

/// A view into device memory. `buf` keeps the allocation alive; `ptr` may point inside it.
#[derive(Clone)]
pub struct Tensor {
    pub buf: Arc<DevBuf>,
    pub ptr: u64,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(dev: &Device, dtype: DType, shape: &[usize]) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let buf = dev.alloc(n * dtype.size())?;
        let ptr = buf.ptr();
        Ok(Tensor { buf: Arc::new(buf), ptr, dtype, shape: shape.to_vec() })
    }
    pub fn zeros(dev: &Device, dtype: DType, shape: &[usize]) -> Result<Tensor> {
        let t = Tensor::new(dev, dtype, shape)?;
        dev.memset_at(t.ptr, t.bytes())?;
        Ok(t)
    }
    pub fn from_buf(buf: DevBuf, dtype: DType, shape: &[usize]) -> Tensor {
        let ptr = buf.ptr();
        Tensor { buf: Arc::new(buf), ptr, dtype, shape: shape.to_vec() }
    }
    pub fn from_f32(dev: &Device, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Ok(Tensor::from_buf(dev.upload(data)?, DType::F32, shape))
    }
    pub fn from_bf16(dev: &Device, data: &[u16], shape: &[usize]) -> Result<Tensor> {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Ok(Tensor::from_buf(dev.upload(data)?, DType::BF16, shape))
    }
    pub fn from_i8(dev: &Device, data: &[i8], shape: &[usize]) -> Result<Tensor> {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Ok(Tensor::from_buf(dev.upload(data)?, DType::I8, shape))
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }
    /// Rows [start, start+len) of a tensor whose first dimension is the row index.
    pub fn rows(&self, start: usize, len: usize) -> Tensor {
        let row_elems: usize = self.shape[1..].iter().product();
        assert!(start + len <= self.shape[0], "rows out of range: {}+{} > {}", start, len, self.shape[0]);
        let mut shape = self.shape.clone();
        shape[0] = len;
        Tensor { buf: self.buf.clone(), ptr: self.ptr + (start * row_elems * self.dtype.size()) as u64, dtype: self.dtype, shape }
    }
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(self.numel(), shape.iter().product::<usize>(), "reshape numel mismatch {:?} -> {:?}", self.shape, shape);
        Tensor { buf: self.buf.clone(), ptr: self.ptr, dtype: self.dtype, shape: shape.to_vec() }
    }
    pub fn to_f32_vec(&self, dev: &Device) -> Result<Vec<f32>> {
        match self.dtype {
            DType::F32 => dev.dtoh_at::<f32>(self.ptr, self.numel()),
            DType::BF16 => {
                let v: Vec<u16> = dev.dtoh_at(self.ptr, self.numel())?;
                Ok(v.into_iter().map(|b| half::bf16::from_bits(b).to_f32()).collect())
            }
            DType::I8 => {
                let v: Vec<i8> = dev.dtoh_at(self.ptr, self.numel())?;
                Ok(v.into_iter().map(|b| b as f32).collect())
            }
            DType::U8 => {
                let v: Vec<u8> = dev.dtoh_at(self.ptr, self.numel())?;
                Ok(v.into_iter().map(|b| b as f32).collect())
            }
        }
    }
}

pub fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
pub fn bf16_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, {:?})", self.dtype, self.shape)
    }
}
