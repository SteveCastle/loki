//! What each step trains on, as a pure function of the seed and the step: the image (shuffled epochs), the noise
//! level, the noise, caption dropout and mirroring. A resumed run therefore continues exactly where the checkpoint
//! left off, and two runs with the same seed see the same data.
use crate::lora::Rng;

/// How noise levels are drawn.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Timesteps {
    /// MOSS-Image's final pretraining schedule (sigma = 1 - sigmoid(N(-1.09, 0.95)), 1% uniform), with `clean_share`
    /// of the steps drawn uniformly from the clean half (sigma < 0.5), where faces and fine detail are learned.
    Moss,
    /// sigma ~ U(0, 1)
    Uniform,
    /// sigma = sigmoid(N(0, 1)) (SD3's logit-normal)
    LogitNormal,
}

/// Noise level for one sample, snapped to the 1000-step training grid (sigma = j / 1000, j = 1..1000).
pub fn sigma(rng: &mut Rng, ts: Timesteps, clean_share: f64) -> f32 {
    let s = match ts {
        Timesteps::Moss => {
            let u = rng.uniform();
            if u < clean_share {
                rng.uniform() * 0.5
            } else if u < clean_share + 0.01 {
                rng.uniform()
            } else {
                let z = -1.09 + 0.95 * rng.normal();
                1.0 - 1.0 / (1.0 + (-z).exp())
            }
        }
        Timesteps::Uniform => rng.uniform(),
        Timesteps::LogitNormal => 1.0 / (1.0 + (-rng.normal()).exp()),
    };
    ((s * 1000.0).round().clamp(1.0, 1000.0) / 1000.0) as f32
}

/// The image a micro-step (step * batch + j) trains on: epochs are shuffled permutations of the set.
pub fn item(seed: u64, micro: usize, n: usize) -> usize {
    let (epoch, pos) = (micro / n, micro % n);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(epoch as u64));
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    perm[pos]
}

/// The per-micro-step stream (noise level, noise, dropout, mirroring).
pub fn micro_rng(seed: u64, micro: usize) -> Rng {
    Rng::new(seed ^ (micro as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) ^ 0x6c6f_6b69)
}

/// Learning rate at optimizer step `step` (0-based): linear warmup, then constant or cosine to 10%.
pub fn lr(base: f32, step: usize, total: usize, warmup: usize, cosine: bool) -> f32 {
    if step < warmup {
        return base * (step + 1) as f32 / warmup as f32;
    }
    if !cosine || total <= warmup {
        return base;
    }
    let t = (step - warmup) as f32 / (total - warmup).max(1) as f32;
    base * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * t.min(1.0)).cos()))
}

/// Default number of optimizer steps for a set of `n` images: about 50 epochs, 1000..3000. A style has to carry over
/// to subjects it never saw, so it trains longer: about 70 epochs, at least 2000.
pub fn default_steps(n: usize, batch: usize, style: bool) -> usize {
    if style {
        return ((n * 70).div_ceil(batch.max(1))).clamp(2000, 3000);
    }
    ((n * 50).div_ceil(batch.max(1))).clamp(1000, 3000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_visit_every_image_once() {
        for n in [1usize, 2, 7, 30] {
            for e in 0..3 {
                let mut v: Vec<usize> = (0..n).map(|p| item(5, e * n + p, n)).collect();
                v.sort();
                assert_eq!(v, (0..n).collect::<Vec<_>>());
            }
        }
        // different epochs, different orders
        let a: Vec<usize> = (0..30).map(|p| item(5, p, 30)).collect();
        let b: Vec<usize> = (0..30).map(|p| item(5, 30 + p, 30)).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn moss_schedule_shape() {
        let mut rng = Rng::new(1);
        let xs: Vec<f32> = (0..20000).map(|_| sigma(&mut rng, Timesteps::Moss, 0.0)).collect();
        let clean = xs.iter().filter(|&&s| s < 0.5).count() as f64 / xs.len() as f64;
        // the MOSS schedule leaves ~12% of steps below sigma 0.5
        assert!((clean - 0.13).abs() < 0.03, "{clean}");
        let mut rng = Rng::new(1);
        let xs: Vec<f32> = (0..20000).map(|_| sigma(&mut rng, Timesteps::Moss, 0.3)).collect();
        let clean = xs.iter().filter(|&&s| s < 0.5).count() as f64 / xs.len() as f64;
        assert!((clean - 0.39).abs() < 0.03, "{clean}");
        assert!(xs.iter().all(|&s| s >= 0.001 && s <= 1.0 && ((s * 1000.0).round() - s * 1000.0).abs() < 1e-3));
    }

    #[test]
    fn warmup_and_cosine() {
        assert_eq!(lr(1.0, 0, 100, 10, false), 0.1);
        assert_eq!(lr(1.0, 50, 100, 10, false), 1.0);
        assert!((lr(1.0, 99, 100, 10, true) - 0.1).abs() < 0.01);
        assert_eq!(default_steps(10, 1, false), 1000);
        assert_eq!(default_steps(40, 1, false), 2000);
        assert_eq!(default_steps(400, 1, false), 3000);
        assert_eq!(default_steps(10, 1, true), 2000);
        assert_eq!(default_steps(35, 1, true), 2450);
    }
}
