//! `torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)`, bit for bit
//! (up to libm's last ulp): MT19937 seeded like ATen's CPU generator, 24-bit uniforms, and ATen's `normal_fill`
//! Box-Muller over blocks of 16 (the last 16 values recomputed from fresh uniforms when the size is not a multiple of 16).
//!
//! MOSS-Image's official pipeline and ComfyUI both draw the initial latent this way on the CPU, so the same seed gives the
//! same starting noise (and, with the same settings, the same image) in loki-dream, diffusers and ComfyUI.

pub struct Mt19937 {
    state: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..624 {
            state[j] = 1812433253u32.wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30)).wrapping_add(j as u32);
        }
        Mt19937 { state, idx: 624 }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for i in 0..624 {
            let y = (self.state[i] & UPPER) | (self.state[(i + 1) % 624] & LOWER);
            let mut v = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= MATRIX_A;
            }
            self.state[i] = v;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.twist();
        }
        let mut y = self.state[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// ATen uniform_real_distribution<float>(0, 1): 24 random bits.
    fn uniform_f32(&mut self) -> f32 {
        let v = self.next_u32() & ((1 << 24) - 1);
        v as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

fn normal_fill_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0f32 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        d[j] = radius * theta.cos();
        d[j + 8] = radius * theta.sin();
    }
}

/// `n` standard normal floats as torch.randn draws them from a freshly seeded CPU generator.
pub fn torch_randn(seed: u64, n: usize) -> Vec<f32> {
    let mut g = Mt19937::new(seed);
    if n < 16 {
        // tiny tensors take another ATen path; never used here (latents have >= 128 values), so only approximately torch's
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u1 = (g.next_u64_f64()).max(f64::MIN_POSITIVE);
            let u2 = g.next_u64_f64();
            out.push(((-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        }
        return out;
    }
    let mut d: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
    let mut i = 0;
    while i + 16 <= n {
        normal_fill_16(&mut d[i..i + 16]);
        i += 16;
    }
    if n % 16 != 0 {
        let tail = n - 16;
        for k in 0..16 {
            d[tail + k] = g.uniform_f32();
        }
        normal_fill_16(&mut d[tail..]);
    }
    d
}

impl Mt19937 {
    /// 53-bit double from two draws (ATen random64 -> uniform_real<double>), only used for tiny tensors.
    fn next_u64_f64(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        let v = ((hi << 32) | lo) & ((1u64 << 53) - 1);
        v as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_sequence() {
        // std::mt19937 seeded with 5489 (the C++ default): the 10000th output is 4123659995
        let mut g = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10000 {
            last = g.next_u32();
        }
        assert_eq!(last, 4123659995);
    }

    #[test]
    fn torch_randn_seed_42() {
        // torch.randn(32, generator=torch.Generator("cpu").manual_seed(42))[:4] = 1.9269, 1.4873, 0.9007, -2.1055
        let v = torch_randn(42, 32);
        let want = [1.9269153f32, 1.4872841, 0.9007172, -2.1055214];
        for (a, b) in v.iter().zip(want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
