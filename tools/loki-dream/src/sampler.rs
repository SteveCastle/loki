//! MOSS-Image sampling schedule: `MOSSImageScheduler.set_timesteps` from pipeline_moss_image.py, in f32 like torch.
//!
//! The official scheduler shifts a linspace of `steps` points (shift 5), then snaps every sigma to the nearest point of
//! the 1000-step training grid (`training_shift` 1, i.e. t/1000 for t = 1..1000) and drops consecutive duplicates.
//! The model sees `timestep = sigma * 1000`; the Euler update is `x += (sigma_next - sigma) * v` with a final sigma of 0.
//!
//! ComfyUI deviates here: its `simple` scheduler over ModelSamplingDiscreteFlow(shift=5) picks sigmas from the
//! *shifted* 1000-point table, so its 50-step schedule is close to but not the official one (see README, "Deviations").

/// torch.linspace(start, end, steps) in float32 (ATen's CPU kernel: from both ends toward the middle).
pub fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps).map(|i| if i < half { start + step * i as f32 } else { end - step * (steps - i - 1) as f32 }).collect()
}

fn grid(count: usize, shift: f32, total: usize) -> Vec<f32> {
    let lower = (shift as f64 / (total as f64 + shift as f64 - 1.0)) as f32;
    linspace_f32(1.0, lower, count).into_iter().map(|v| (shift * v) / (1.0 + (shift - 1.0) * v)).collect()
}

/// Sigmas of the official schedule for `steps` steps (shift 5), ending with the terminal 0. The number of steps actually
/// taken is `len() - 1` (snapping can merge neighbours at high step counts).
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    const TOTAL: usize = 1000;
    let s = grid(steps.max(1), shift, TOTAL);
    let mut training = grid(TOTAL, 1.0, TOTAL);
    training.reverse(); // ascending
    let mut out: Vec<f32> = Vec::with_capacity(s.len() + 1);
    for &v in &s {
        let upper = training.partition_point(|&t| t < v).min(TOTAL - 1);
        let lower = upper.saturating_sub(1);
        let pick = if (v - training[lower]).abs() <= (training[upper] - v).abs() { lower } else { upper };
        let snapped = training[pick];
        if out.last() != Some(&snapped) {
            out.push(snapped);
        }
    }
    out.push(0.0);
    out
}

/// The timestep the transformer sees for a sigma (diffusers: sigmas * num_train_timesteps, in f32).
pub fn timestep(sigma: f32) -> f32 {
    sigma * 1000.0
}

/// img2img (SDEdit) like diffusers' img2img pipelines: with `strength` s, skip the first `steps - round_down(steps*s)`
/// sigmas; sampling starts from `(1 - sigma) x0 + sigma noise` at the first kept sigma.
pub fn img2img_start(n_steps: usize, strength: f32) -> usize {
    let init = ((n_steps as f32 * strength.clamp(0.0, 1.0)) as usize).min(n_steps);
    n_steps - init
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_torch() {
        let v = linspace_f32(1.0, 0.001, 1000);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[999], 0.001);
        assert_eq!(v.len(), 1000);
    }

    #[test]
    fn eight_step_schedule_is_monotone_and_snapped() {
        let s = sigmas(8, 5.0);
        assert_eq!(s.len(), 9);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        for w in s.windows(2) {
            assert!(w[0] > w[1]);
        }
        // every non-terminal sigma sits on the t/1000 training grid
        for &v in &s[..8] {
            let t = v * 1000.0;
            assert!((t - t.round()).abs() < 1e-2, "{v}");
        }
    }

    #[test]
    fn strength_skips_steps() {
        assert_eq!(img2img_start(20, 1.0), 0);
        assert_eq!(img2img_start(20, 0.5), 10);
        assert_eq!(img2img_start(8, 0.6), 4);
    }
}
