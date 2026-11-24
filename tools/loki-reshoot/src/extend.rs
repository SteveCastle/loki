//! Long videos (past the model's ~15 s): chained chunks, each continuing the clean latent tail of the one before.
//!
//! Chunk k > 1 is a normal run whose first `context` frames are anchored to the last `context` frames generated so far,
//! as a guide (ComfyUI `MiniMaxH3AddGuide` at frame 0: video latent + its soundtrack, never denoised). The model
//! re-generates those frames around the guide and continues; the long latent keeps only what follows them. The tail is
//! cut straight out of the latent (no decode / re-encode): `context` is 17k+5 frames, so it starts on a latent-cycle
//! boundary (the VAE packs every 17 frames into 5 latent frames: 1, 4, 4, 4, 4) and is itself a valid clip latent.
//! The kept latents concatenate into one valid latent of the whole video, decoded once, so there is no pixel seam.
//! (The technique of H3 Continuum / the H3 Extender's motion context, on the core guide path.)
use crate::media::FPS;
use crate::pipeline::{align_frame_count, MAX_FRAMES};
use anyhow::{ensure, Result};

/// Default chunk length when a long video does not set one (10 s; Continuum's authors recommend 10-15 s chunks over 5 s).
pub const DEFAULT_CHUNK_SECONDS: f64 = 10.0;
/// Default context: 22 frames (7 latent frames, ~0.9 s), Continuum's "Balanced".
pub const DEFAULT_CONTEXT_FRAMES: usize = 22;
/// Longest video a run accepts (5 minutes).
pub const MAX_TOTAL_FRAMES: usize = 5 * 60 * FPS + 5;
/// audio latent frames per video frame
const AUDIO_PER_FRAME: f64 = 40.0 / FPS as f64;

#[derive(Clone, Debug)]
pub struct Extend {
    /// longest chunk, frames (17k+5, <= MAX_FRAMES)
    pub chunk_frames: usize,
    /// frames each chunk continues from (5, 22 or 39)
    pub context_frames: usize,
    /// prompts of chunks 2, 3, ... (the last one repeats; empty = the main prompt throughout)
    pub prompts: Vec<String>,
}

impl Extend {
    pub fn new(chunk_seconds: f64, context_frames: usize, prompts: Vec<String>) -> Result<Extend> {
        ensure!([5, 22, 39].contains(&context_frames), "--context-frames must be 5, 22 or 39 (whole latent cycles), got {context_frames}");
        let chunk_frames = align_frame_count(((chunk_seconds * FPS as f64).round() as usize).max(5));
        ensure!(chunk_frames <= MAX_FRAMES, "--chunk-seconds {chunk_seconds} exceeds the model's ~15 s limit");
        ensure!(chunk_frames >= context_frames + 2 * 17, "--chunk-seconds {chunk_seconds} is too short for {context_frames} context frames");
        Ok(Extend { chunk_frames, context_frames, prompts })
    }
    /// Index of chunk `k`'s prompt (0-based) in [main prompt, prompts...].
    pub fn prompt_index(&self, k: usize) -> usize {
        k.min(self.prompts.len())
    }
}

/// Frame counts of the chunks of a `total`-frame video (all 17k+5): as few chunks as `chunk` frames allow, all about
/// the same length (no short tail chunk, which would fall outside the trained 5-15 s range). Chunk 1 adds all its
/// frames, every later one its frames minus `context`.
pub fn plan(total: usize, chunk: usize, context: usize) -> Result<Vec<usize>> {
    ensure!(total % 17 == 5 && chunk % 17 == 5 && context % 17 == 5, "frame counts must be 17k+5");
    ensure!(chunk >= context + 17, "chunk ({chunk}) must be longer than the context ({context})");
    ensure!(total <= MAX_TOTAL_FRAMES, "{total} frames ({:.0} s) is longer than the {} s a run allows", total as f64 / FPS as f64, MAX_TOTAL_FRAMES / FPS);
    if total <= chunk {
        return Ok(vec![total]);
    }
    // in units of 17 frames: chunk 1 = 17a + 5 adds a units, a later chunk 17(u+1) + 5 (= 17u + 22 with 22 context) adds u
    let ctx_units = (context - 5) / 17;
    let units = (total - 5) / 17;
    let cap = (chunk - 5) / 17;
    let n_more = (units - cap).div_ceil(cap - ctx_units);
    let n = n_more + 1;
    // equal lengths: every chunk is w units long (chunk 1 adds w, the others w - ctx_units)
    let w = units + n_more * ctx_units;
    Ok((0..n).map(|i| 17 * (w / n + usize::from(i < w % n)) + 5).collect())
}

/// Audio latent frames of the context's soundtrack.
pub fn context_audio_ticks(context: usize) -> usize {
    (context as f64 * AUDIO_PER_FRAME).round() as usize
}

/// Where a chunk starting at long-video frame `start` sits on the long soundtrack, which has `have` audio latent frames
/// so far and carries the last `ctx_ticks` of them as the guide: (the guide's offset from the chunk's audio origin, in
/// audio latent frames; the first audio latent frame of the chunk that is new).
pub fn audio_join(start: usize, have: usize, ctx_ticks: usize) -> (f64, usize) {
    let origin = start as f64 * AUDIO_PER_FRAME;
    ((have - ctx_ticks) as f64 - origin, (have as f64 - origin).round().max(0.0) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn total_of(chunks: &[usize], ctx: usize) -> usize {
        chunks[0] + chunks[1..].iter().map(|c| c - ctx).sum::<usize>()
    }
    #[test]
    fn plans_cover_the_video_in_equal_chunks() {
        for ctx in [5, 22, 39] {
            for chunk in [124, 243, 362] {
                for total in (5..2000).filter(|f| f % 17 == 5) {
                    let p = plan(total, chunk, ctx).unwrap();
                    assert_eq!(total_of(&p, ctx), total, "total {total} chunk {chunk} ctx {ctx}: {p:?}");
                    assert!(p.iter().all(|&c| c <= chunk && c % 17 == 5 && (p.len() == 1 || c > ctx)), "{p:?}");
                    assert!(p.iter().max().unwrap() - p.iter().min().unwrap() <= 17, "{p:?}");
                    // fewest chunks: one fewer would not fit
                    if p.len() > 1 {
                        assert!(chunk + (p.len() - 2) * (chunk - ctx) < total, "total {total} chunk {chunk} ctx {ctx}: {p:?}");
                    }
                }
            }
        }
        assert_eq!(plan(736, 243, 22).unwrap(), vec![209, 209, 192, 192]);
        assert_eq!(plan(243, 243, 22).unwrap(), vec![243]);
    }
    #[test]
    fn audio_joins_stay_on_the_timeline() {
        // 22 frames of context = 36.67 audio latent frames, carried as 37
        assert_eq!(context_audio_ticks(22), 37);
        // after a 209-frame chunk (348.33 -> 348 audio frames), the next chunk starts at frame 187 (311.67)
        let (off, skip) = audio_join(187, 348, 37);
        assert!((off + 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(skip, 36);
    }
}
