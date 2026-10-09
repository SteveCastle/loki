//! res_multistep vs ComfyUI (toy denoiser). Needs ref_out/sampler from ref/ref_sampler.py.
use loki_reshoot::cuda::Device;
use loki_reshoot::sampler;
use loki_reshoot::tensor::{DType, Tensor};

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn res_multistep_matches_comfy() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/ref_out/sampler/");
    if !std::path::Path::new(&format!("{dir}out.bin")).exists() {
        eprintln!("no reference dump, skipping");
        return;
    }
    let (x0, want) = (read_f32(&format!("{dir}x0.bin")), read_f32(&format!("{dir}out.bin")));
    let ref_sig = read_f32(&format!("{dir}sigmas.bin"));
    let sig = sampler::simple_sigmas(20, 12.0);
    for (a, b) in sig.iter().zip(&ref_sig) {
        assert!((*a as f32 - b).abs() < 1e-6, "sigma {a} vs {b}");
    }
    let dev = Device::new(0).unwrap();
    let xv = Tensor::from_f32(&dev, &x0, &[1000]).unwrap();
    let xa = Tensor::from_f32(&dev, &x0[..8], &[8]).unwrap();
    let dummy = Tensor::new(&dev, DType::F32, &[8]).unwrap();
    let _ = dummy;
    let d2 = dev.clone();
    sampler::res_multistep(
        &dev,
        &sig,
        &xv,
        &xa,
        &mut |x, _a, s, ov, oa| {
            let xs = x.to_f32_vec(&d2)?;
            let den: Vec<f32> = xs.iter().map(|v| v / (1.0 + s) + 0.1 * (3.0 * v).cos()).collect();
            let v: Vec<f32> = xs.iter().zip(&den).map(|(x, d)| (x - d) / s).collect();
            d2.htod_at(ov.ptr, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) })?;
            d2.htod_at(oa.ptr, &vec![0u8; 32])?;
            Ok(())
        },
        &|_, _| {},
    )
    .unwrap();
    let got = xv.to_f32_vec(&dev).unwrap();
    let mut maxd = 0f32;
    for (g, w) in got.iter().zip(&want) {
        maxd = maxd.max((g - w).abs());
    }
    eprintln!("max abs diff {maxd}");
    assert!(maxd < 2e-4, "res_multistep differs: {maxd}");
}
