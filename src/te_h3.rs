//! te_h3 (stub with the agreed API)
use crate::cuda::Device;
use crate::tensor::Tensor;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
pub enum RefItem { Image { w: usize, h: usize, rgb: Vec<u8> }, Audio, Video { frames: Vec<(usize, usize, Vec<u8>)>, timestamps: Vec<f32> } }
pub struct Conditioning { pub context: Tensor, pub tags: Vec<u8> }
pub struct TextEncoder;
impl TextEncoder {
    pub fn load(_dev: Arc<Device>, _path: &Path) -> Result<Self> { unimplemented!() }
    pub fn encode(&self, _prompt: &str, _items: &[RefItem]) -> Result<Conditioning> { unimplemented!() }
}
