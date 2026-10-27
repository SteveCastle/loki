use std::path::{Path, PathBuf};
use std::process::Command;

fn find_cl() -> Option<PathBuf> {
    let vswhere = Path::new(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let out = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-find", r"VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(PathBuf::from(line).parent()?.to_path_buf())
    }
}

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let kdir = Path::new("kernels");
    println!("cargo:rerun-if-changed=kernels");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let ccbin = find_cl();
    // SASS for every architecture in LOKI_CUDA_ARCHS (default 89, the RTX 40 series), plus PTX for LOKI_CUDA_PTX
    // (default 80) that the driver JITs on newer GPUs. The FP8 tensor-core kernels (retouch, reshoot, dream) need 89+, so the worker
    // image builds LOKI_CUDA_ARCHS=89,90 LOKI_CUDA_PTX=90 (Ada, Hopper, and Blackwell through the PTX).
    println!("cargo:rerun-if-env-changed=LOKI_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=LOKI_CUDA_PTX");
    let ptx = std::env::var("LOKI_CUDA_PTX").unwrap_or_else(|_| "80".to_string());
    anyhow::ensure!(!ptx.is_empty() && ptx.chars().all(|c| c.is_ascii_digit()), "LOKI_CUDA_PTX: {ptx:?} is not an architecture number such as 90");
    let archs = std::env::var("LOKI_CUDA_ARCHS").unwrap_or_else(|_| "89".to_string());
    let mut gencode: Vec<String> = Vec::new();
    for a in archs.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        anyhow::ensure!(a.chars().all(|c| c.is_ascii_digit()), "LOKI_CUDA_ARCHS: {a:?} is not an architecture number such as 89");
        gencode.extend(["-gencode".to_string(), format!("arch=compute_{a},code=sm_{a}")]);
    }
    gencode.extend(["-gencode".to_string(), format!("arch=compute_{ptx},code=compute_{ptx}")]);
    let mut entries: Vec<_> = std::fs::read_dir(kdir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "cu").unwrap_or(false))
        .collect();
    entries.sort();
    for p in std::fs::read_dir(kdir)? {
        println!("cargo:rerun-if-changed={}", p?.path().display());
    }
    for cu in entries {
        let stem = cu.file_stem().unwrap().to_string_lossy().to_string();
        let out = out_dir.join(format!("{stem}.fatbin"));
        let mut cmd = Command::new(&nvcc);
        cmd.arg("-fatbin")
            .arg("-O3")
            .arg("-std=c++17")
            .arg("--expt-relaxed-constexpr")
            .arg("-lineinfo")
            .arg("-Xptxas").arg("-v")
            .args(&gencode)
            .arg("-I").arg(kdir)
            .arg("-o").arg(&out)
            .arg(&cu);
        if let Some(cc) = &ccbin {
            cmd.arg("-ccbin").arg(cc);
        }
        let output = cmd.output().map_err(|e| anyhow::anyhow!("failed to run nvcc: {e}"))?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        for line in stderr.lines() {
            println!("cargo:warning=[nvcc {stem}] {line}");
        }
        if !output.status.success() {
            // cl.exe (and nvcc's own fatal errors) report on stdout, so include both.
            let stdout = String::from_utf8_lossy(&output.stdout);
            anyhow::bail!("nvcc failed on {}:\n{}\n{}", cu.display(), stdout, stderr);
        }
    }
    Ok(())
}
