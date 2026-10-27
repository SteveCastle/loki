"""Shared helpers for the reference dumps: build the OFFICIAL SAM 3.1 stack (facebookresearch/sam3) from the
same fp16 safetensors file the native engine loads, and write tensors as raw little-endian f32 + a JSON index.

Run with the reference venv, isolated:  tools/ref-env/Scripts/python.exe -I ref/<script>.py ...
The sam3 package is installed into that venv from ref_src/sam3 (pip install --no-deps).
"""
import contextlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.environ.get("SAM3_CKPT", os.path.join(ROOT, "models", "sam3.1_multiplex_fp16.safetensors"))


def load_state_dict(path=CKPT):
    from safetensors.torch import load_file

    sd = load_file(path)
    return {k: v.float() for k, v in sd.items()}


def build_predictor():
    """The official SAM 3.1 multiplex video predictor (detector + tracker), fp32 weights, no FA3."""
    import sam3.model_builder as mb
    import sam3.model.decoder as dec

    # the memory attention pins SDPBackend.FLASH_ATTENTION, which Windows builds of PyTorch do not ship;
    # let SDPA pick a backend (same math)
    dec.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()

    real_load = torch.load

    def fake_load(p, *a, **k):
        name = getattr(p, "name", p)
        if str(name).endswith(".safetensors"):
            return load_state_dict(str(name))
        return real_load(p, *a, **k)

    torch.load = fake_load
    try:
        pred = mb.build_sam3_multiplex_video_predictor(
            checkpoint_path=CKPT, use_fa3=False, use_rope_real=False, compile=False, async_loading_frames=False
        )
    finally:
        torch.load = real_load
    # upstream predictor passes a kwarg the multiplex model's init_state does not take
    init = pred.model.init_state
    pred.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k != "offload_state_to_cpu"})
    return pred


def force_fp32():
    """Make the reference a true fp32 run: the official ViT/MLP path always runs fc1+act in bf16
    (perflib.fused.addmm_act); replace it with the plain fp32 computation."""
    import torch.nn.functional as F
    import sam3.model.vitdet as vitdet

    def addmm_act_fp32(activation, linear, mat1):
        y = linear(mat1)
        if activation in [F.relu, torch.nn.ReLU]:
            return F.relu(y)
        return F.gelu(y)

    vitdet.addmm_act = addmm_act_fp32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Dump:
    """Writes name.bin (f32) and index.json {name: shape} into a directory."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.index = {}

    def __call__(self, name, t):
        if isinstance(t, torch.Tensor):
            t = t.detach().float().cpu().numpy()
        a = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
        a.tofile(os.path.join(self.dir, name + ".bin"))
        self.index[name] = list(a.shape)
        with open(os.path.join(self.dir, "index.json"), "w") as f:
            json.dump(self.index, f, indent=1)
        print(f"  dump {name} {list(a.shape)}", file=sys.stderr)
