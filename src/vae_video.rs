//! vae_video (stub with the agreed API)
use crate::cuda::Device;
use crate::tensor::Tensor;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
pub struct VideoVae;
impl VideoVae {
    pub fn load(_dev: Arc<Device>, _path: &Path) -> Result<VideoVae> { unimplemented!() }
    pub fn encode(&self, _frames: &[u8], _t: usize, _h: usize, _w: usize) -> Result<Tensor> { unimplemented!() }
    pub fn decode(&self, _z: &Tensor, _tl: usize, _hl: usize, _wl: usize, _sink: &mut dyn FnMut(&[u8], usize) -> Result<()>) -> Result<usize> { unimplemented!() }
}
