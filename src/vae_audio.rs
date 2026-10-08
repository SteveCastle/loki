//! vae_audio (stub with the agreed API)
use crate::cuda::Device;
use crate::tensor::Tensor;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
pub struct AudioVae;
impl AudioVae {
    pub fn load(_dev: Arc<Device>, _path: &Path) -> Result<AudioVae> { unimplemented!() }
    pub fn encode(&self, _wav: &[f32], _n: usize) -> Result<Tensor> { unimplemented!() }
    pub fn decode(&self, _z: &Tensor, _t: usize) -> Result<Vec<f32>> { unimplemented!() }
}
