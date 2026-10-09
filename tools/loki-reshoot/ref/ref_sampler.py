"""Reference for res_multistep + simple scheduler (toy denoiser). Run with the Comfy embedded python, cwd = ComfyUI root.
Writes ref_out/sampler/{x0.bin,sigmas.bin,out.bin} (f32)."""
import sys, os
sys.path.insert(0, os.getcwd())
import torch, numpy as np
import comfy.samplers, comfy.k_diffusion.sampling as ks, comfy.model_sampling

out = r"C:\Users\steph\dev\loki\tools\loki-reshoot\ref_out\sampler"
os.makedirs(out, exist_ok=True)
class MS:  # ModelSamplingAV(shift=12) sigmas table
    def __init__(self):
        ms = comfy.model_sampling.ModelSamplingDiscreteFlow()
        ms.set_parameters(shift=12.0)
        self.sigmas = ms.sigmas
        self.noise_scale = 1.0
ms = MS()
sig = comfy.samplers.simple_scheduler(ms, 20)
print("sigmas", sig[:4].tolist(), sig[-3:].tolist())
def f(x, s):
    return x / (1.0 + s) + 0.1 * torch.cos(3.0 * x)
class Inner:
    class P:
        def get_model_object(self, n): return ms
    model_patcher = P()
class M:
    inner_model = Inner()
    def __call__(self, x, sigma, **kw):
        return f(x, sigma.view(-1, *([1] * (x.ndim - 1))))
g = torch.Generator().manual_seed(1)
x0 = torch.randn(1, 1000, generator=g)
res = ks.sample_res_multistep(M(), x0.clone(), sig, disable=True)
x0.numpy().astype(np.float32).tofile(out + r"\x0.bin")
sig.numpy().astype(np.float32).tofile(out + r"\sigmas.bin")
res.numpy().astype(np.float32).tofile(out + r"\out.bin")
print("done", float(res.abs().mean()))
