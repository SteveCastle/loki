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
            .arg("-gencode").arg("arch=compute_89,code=sm_89")
            .arg("-gencode").arg("arch=compute_80,code=compute_80")
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
            anyhow::bail!("nvcc failed on {}:\n{}", cu.display(), stderr);
        }
    }
    Ok(())
}
