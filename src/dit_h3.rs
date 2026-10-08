//! dit_h3 (stub with the agreed API; replaced by the DiT workstream)
use crate::cuda::Device;
use crate::tensor::Tensor;
use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
pub enum RefKind { Image, Audio, Video, VideoAudio }
pub struct RefBlock { pub kind: RefKind, pub latent_t: usize, pub latent_h: usize, pub latent_w: usize, pub ref_audio_t: usize, pub video: Option<Tensor>, pub audio: Option<Tensor> }
pub struct DitInputs<'a> { pub text: &'a Tensor, pub text_tags: &'a [u8], pub refs: &'a [RefBlock], pub latent_t: usize, pub latent_h: usize, pub latent_w: usize, pub audio_t: usize, pub seed: u64, pub cond_noise_aug: Option<f32> }
pub struct Run;
pub struct Dit;
impl Dit {
    pub fn load(_dev: Arc<Device>, _path: &Path) -> Result<Dit> { unimplemented!() }
    pub fn prepare(&self, _inp: &DitInputs) -> Result<Run> { unimplemented!() }
    pub fn forward(&self, _run: &mut Run, _xv: &Tensor, _xa: &Tensor, _sigma: f32, _ov: &Tensor, _oa: &Tensor) -> Result<()> { unimplemented!() }
}
