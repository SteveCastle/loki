//! Thin CUDA driver layer: context, stream, kernel registry, launches, device buffers.
use anyhow::{anyhow, Context, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const FATBINS: &[(&str, &[u8])] = &[
    ("elementwise", include_bytes!(concat!(env!("OUT_DIR"), "/elementwise.fatbin"))),
    ("quant", include_bytes!(concat!(env!("OUT_DIR"), "/quant.fatbin"))),
    ("gemm_int8", include_bytes!(concat!(env!("OUT_DIR"), "/gemm_int8.fatbin"))),
    ("gemm_bf16", include_bytes!(concat!(env!("OUT_DIR"), "/gemm_bf16.fatbin"))),
    ("attention", include_bytes!(concat!(env!("OUT_DIR"), "/attention.fatbin"))),
    ("h3_dit", include_bytes!(concat!(env!("OUT_DIR"), "/h3_dit.fatbin"))),
    ("nvfp4", include_bytes!(concat!(env!("OUT_DIR"), "/nvfp4.fatbin"))),
    ("vvae", include_bytes!(concat!(env!("OUT_DIR"), "/vvae.fatbin"))),
    ("avae", include_bytes!(concat!(env!("OUT_DIR"), "/avae.fatbin"))),
    ("sampler", include_bytes!(concat!(env!("OUT_DIR"), "/sampler.fatbin"))),
    ("sage_attn", include_bytes!(concat!(env!("OUT_DIR"), "/sage_attn.fatbin"))),
];

/// Kernel argument value. Pointers are passed as raw 64-bit device addresses.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    U32(u32),
    I64(i64),
    F32(f32),
}
impl From<u64> for Arg { fn from(v: u64) -> Self { Arg::Ptr(v) } }
impl From<i32> for Arg { fn from(v: i32) -> Self { Arg::I32(v) } }
impl From<u32> for Arg { fn from(v: u32) -> Self { Arg::U32(v) } }
impl From<i64> for Arg { fn from(v: i64) -> Self { Arg::I64(v) } }
impl From<f32> for Arg { fn from(v: f32) -> Self { Arg::F32(v) } }
impl From<usize> for Arg { fn from(v: usize) -> Self { Arg::I32(v as i32) } }

pub struct Device {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    modules: Vec<Arc<CudaModule>>,
    funcs: Mutex<HashMap<String, CudaFunction>>,
    pub sm_count: i32,
    pub total_mem: usize,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ctx = CudaContext::new(ordinal).context("creating CUDA context (is an NVIDIA driver installed?)")?;
        // Single-stream engine: skip cudarc's per-buffer event bookkeeping.
        unsafe { ctx.disable_event_tracking(); }
        let stream = ctx.default_stream();
        // Keep freed memory in the driver's pool instead of handing it back on every sync.
        unsafe {
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let dev = ordinal as sys::CUdevice;
            if sys::cuDeviceGetDefaultMemPool(&mut pool, dev) == sys::CUresult::CUDA_SUCCESS {
                let mut thr: u64 = u64::MAX;
                let _ = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut thr as *mut u64 as *mut std::ffi::c_void,
                );
            }
        }
        let sm_count = ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let (_free, total) = ctx.mem_get_info()?;
        let mut modules = Vec::new();
        for (name, bin) in FATBINS {
            let m = ctx
                .load_module(Ptx::from_binary(bin.to_vec()))
                .with_context(|| format!("loading CUDA module {name}"))?;
            modules.push(m);
        }
        Ok(Arc::new(Device { ctx, stream, modules, funcs: Mutex::new(HashMap::new()), sm_count, total_mem: total }))
    }

    pub fn func(&self, name: &str) -> Result<CudaFunction> {
        if let Some(f) = self.funcs.lock().unwrap().get(name) {
            return Ok(f.clone());
        }
        for m in &self.modules {
            if let Ok(f) = m.load_function(name) {
                self.funcs.lock().unwrap().insert(name.to_string(), f.clone());
                return Ok(f);
            }
        }
        Err(anyhow!("kernel {name} not found in any module"))
    }

    /// Opt a kernel into more than 48KB of dynamic shared memory.
    pub fn set_max_smem(&self, name: &str, bytes: u32) -> Result<()> {
        let f = self.func(name)?;
        f.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32)?;
        Ok(())
    }

    pub fn launch(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, args: &[Arg]) -> Result<()> {
        let f = self.func(name)?;
        let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: smem };
        // Materialize argument storage so the builder can take stable references.
        let mut p: Vec<u64> = Vec::new();
        let mut i: Vec<i32> = Vec::new();
        let mut u: Vec<u32> = Vec::new();
        let mut l: Vec<i64> = Vec::new();
        let mut fl: Vec<f32> = Vec::new();
        for a in args {
            match a {
                Arg::Ptr(v) => p.push(*v),
                Arg::I32(v) => i.push(*v),
                Arg::U32(v) => u.push(*v),
                Arg::I64(v) => l.push(*v),
                Arg::F32(v) => fl.push(*v),
            }
        }
        let (mut pi, mut ii, mut ui, mut li, mut fi) = (0, 0, 0, 0, 0);
        let mut b = self.stream.launch_builder(&f);
        for a in args {
            match a {
                Arg::Ptr(_) => { b.arg(&p[pi]); pi += 1; }
                Arg::I32(_) => { b.arg(&i[ii]); ii += 1; }
                Arg::U32(_) => { b.arg(&u[ui]); ui += 1; }
                Arg::I64(_) => { b.arg(&l[li]); li += 1; }
                Arg::F32(_) => { b.arg(&fl[fi]); fi += 1; }
            }
        }
        unsafe { b.launch(cfg) }.with_context(|| format!("launching {name} grid={grid:?} block={block:?} smem={smem}"))?;
        Ok(())
    }

    /// 1-D launch helper for `n` elements with 256 threads per block.
    pub fn launch_n(&self, name: &str, n: usize, args: &[Arg]) -> Result<()> {
        let blocks = ((n + 255) / 256).max(1) as u32;
        self.launch(name, (blocks, 1, 1), (256, 1, 1), 0, args)
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn free_mem(&self) -> Result<usize> {
        Ok(self.ctx.mem_get_info()?.0)
    }

    pub fn alloc(&self, bytes: usize) -> Result<DevBuf> {
        let bytes = bytes.max(256);
        let slice = unsafe { self.stream.alloc::<u8>(bytes) }
            .with_context(|| format!("allocating {} MB on device (free {} MB)", bytes >> 20, self.free_mem().unwrap_or(0) >> 20))?;
        let ptr = slice.leak();
        Ok(DevBuf { ptr, len: bytes, stream: self.stream.clone() })
    }
    pub fn alloc_zeros(&self, bytes: usize) -> Result<DevBuf> {
        let b = self.alloc(bytes)?;
        self.memset_at(b.ptr, b.len)?;
        Ok(b)
    }

    pub fn htod<T: Copy>(&self, buf: &DevBuf, data: &[T]) -> Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        assert!(bytes.len() <= buf.len, "htod overflow {} > {}", bytes.len(), buf.len);
        self.htod_at(buf.ptr, bytes)
    }
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<DevBuf> {
        let b = self.alloc(std::mem::size_of_val(data))?;
        self.htod(&b, data)?;
        Ok(b)
    }
    pub fn dtoh<T: Copy + Default>(&self, buf: &DevBuf, count: usize) -> Result<Vec<T>> {
        assert!(count * std::mem::size_of::<T>() <= buf.len, "dtoh overflow");
        self.dtoh_at(buf.ptr, count)
    }
    pub fn dtoh_at<T: Copy + Default>(&self, ptr: u64, count: usize) -> Result<Vec<T>> {
        let mut out = vec![T::default(); count];
        let bytes = count * std::mem::size_of::<T>();
        unsafe {
            let r = sys::cuMemcpyDtoHAsync_v2(out.as_mut_ptr() as *mut _, ptr, bytes, self.stream.cu_stream());
            if r != sys::CUresult::CUDA_SUCCESS { return Err(anyhow!("cuMemcpyDtoH failed: {:?}", r)); }
        }
        self.stream.synchronize()?;
        Ok(out)
    }
    pub fn dtod(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        unsafe {
            let r = sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream.cu_stream());
            if r != sys::CUresult::CUDA_SUCCESS { return Err(anyhow!("cuMemcpyDtoD failed: {:?}", r)); }
        }
        Ok(())
    }
    pub fn htod_at(&self, dst: u64, bytes: &[u8]) -> Result<()> {
        unsafe {
            let r = sys::cuMemcpyHtoD_v2(dst, bytes.as_ptr() as *const _, bytes.len());
            if r != sys::CUresult::CUDA_SUCCESS { return Err(anyhow!("cuMemcpyHtoD failed: {:?}", r)); }
        }
        Ok(())
    }
    pub fn memset_at(&self, dst: u64, bytes: usize) -> Result<()> {
        unsafe {
            let r = sys::cuMemsetD8Async(dst, 0, bytes, self.stream.cu_stream());
            if r != sys::CUresult::CUDA_SUCCESS { return Err(anyhow!("cuMemsetD8 failed: {:?}", r)); }
        }
        Ok(())
    }
}

/// Owned device allocation (raw pointer; freed asynchronously on drop).
pub struct DevBuf {
    ptr: u64,
    pub len: usize,
    stream: Arc<CudaStream>,
}
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}
impl DevBuf {
    #[inline]
    pub fn ptr(&self) -> u64 { self.ptr }
}
impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::free_async(self.ptr, self.stream.cu_stream());
        }
    }
}

// ------------------------------------------------------------------------------------------
// Lightweight profiler (enabled with LOKI_PROFILE=1): accumulates GPU time per category.
pub struct Profiler {
    enabled: bool,
    acc: Mutex<Vec<(String, f64, usize)>>,
}
impl Profiler {
    pub fn new() -> Profiler {
        Profiler { enabled: std::env::var("LOKI_PROFILE").is_ok(), acc: Mutex::new(Vec::new()) }
    }
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    /// Time `f` on the device (synchronizes when enabled).
    pub fn time<F: FnOnce() -> Result<()>>(&self, dev: &Device, name: &str, f: F) -> Result<()> {
        if !self.enabled {
            return f();
        }
        dev.sync()?;
        let t0 = std::time::Instant::now();
        f()?;
        dev.sync()?;
        let dt = t0.elapsed().as_secs_f64();
        let mut acc = self.acc.lock().unwrap();
        if let Some(e) = acc.iter_mut().find(|e| e.0 == name) {
            e.1 += dt;
            e.2 += 1;
        } else {
            acc.push((name.to_string(), dt, 1));
        }
        Ok(())
    }
    pub fn report(&self) {
        if !self.enabled {
            return;
        }
        let acc = self.acc.lock().unwrap();
        let total: f64 = acc.iter().map(|e| e.1).sum();
        crate::info!("---- profile (total {:.2}s) ----", total);
        let mut v: Vec<_> = acc.iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (n, t, c) in v {
            crate::info!("  {:<24} {:>8.3}s  {:>5.1}%  x{}", n, t, 100.0 * t / total, c);
        }
    }
}
