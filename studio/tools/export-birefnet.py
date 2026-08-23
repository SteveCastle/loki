"""Build the WebGPU-ready BiRefNet used by studio's roto node.

The raw onnx-community BiRefNet exports CANNOT run on onnxruntime-web's
WebGPU EP (measured 2026-08-23, ort-web 1.27, RTX 4090):

  * the decoder holds Concat nodes with up to 1024 inputs and Split nodes
    with 32 outputs — past the GPU's storage-buffers-per-shader limit
    (ort batches wide Concats since 1.27, with an off-by-one; wide Splits
    are simply unsupported);
  * the variadic `Sum` op has no WebGPU kernel, and it sits exactly where
    the unrolled deformable convolutions materialize ~784 MB tensors — the
    CPU fallback walks the 4 GB wasm heap into std::bad_alloc;
  * the fp16 export defeats onnxruntime's constant folding (no CPU fp16
    kernels for the folded chains), leaving 312 ConstantOfShape islands.

This script rebuilds the model so every node runs on the GPU:

  1. offline BASIC-level optimization (onnxruntime) — constant-folds the
     graph from ~16k nodes to ~4k and eliminates every CPU-only shape op;
  2. cascade surgery — wide Concat/Split fanouts become trees capped at 6,
     so any shader needs at most 7 storage buffers (safe even on devices
     with the WebGPU minimum limit of 8), and Sum becomes an Add chain;
  3. float16 conversion with keep_io_types (graph I/O stays float32, so
     the browser feeds/reads plain Float32Arrays).

The result is bit-identical to the source graph at fp32 (max sigmoid-
space deviation after fp16: ~2e-3). Any BiRefNet variant with the same
I/O contract (general / HR / matting / dynamic) should convert the same
way — point --src at its fp32 model.onnx.

Usage:
  pip install onnx onnxruntime onnxconverter-common numpy
  python export-birefnet.py \
      --src https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx \
      --out birefnet_lite_webgpu_fp16.onnx

Host the output anywhere (a GitHub release asset works) and point
localStorage['lowkey-studio.roto-config'] = {"modelUrl": "<url>"} at it —
or update DEFAULTS.modelUrl in studio/roto.js.
"""
import argparse
import os
import sys
import tempfile
import urllib.request

import numpy as np
import onnx
from onnx import helper, numpy_helper

CAP = 6   # max fanout per node: CAP inputs/outputs + data buffer <= 8 bindings


def cascade_patch(model):
    """Rewrite wide Concat/Split and variadic Sum into GPU-safe cascades."""
    g = model.graph
    consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in g.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value':
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    new_inits = []
    uid = [0]

    def uniq(base):
        uid[0] += 1
        return f'{base}_wgpu{uid[0]}'

    def cascade_concat(node):
        axis = next(a.i for a in node.attribute if a.name == 'axis')
        nodes = []
        inputs = list(node.input)
        while len(inputs) > CAP:
            nxt = []
            for i in range(0, len(inputs), CAP):
                chunk = inputs[i:i + CAP]
                if len(chunk) == 1:
                    nxt.append(chunk[0])
                    continue
                out = uniq(node.name + '_cc')
                nodes.append(helper.make_node('Concat', chunk, [out], name=out, axis=axis))
                nxt.append(out)
            inputs = nxt
        nodes.append(helper.make_node('Concat', inputs, list(node.output),
                                      name=node.name + '_wgpu_top', axis=axis))
        return nodes

    def cascade_split(node):
        axis = next(a.i for a in node.attribute if a.name == 'axis')
        sizes = consts[node.input[1]].astype(np.int64)
        assert len(sizes) == len(node.output)
        nodes = []

        def emit(data_in, outs, szs):
            if len(outs) <= CAP:
                sname = uniq(node.name + '_sz')
                new_inits.append(numpy_helper.from_array(np.asarray(szs, np.int64), sname))
                nodes.append(helper.make_node('Split', [data_in, sname], list(outs),
                                              name=uniq(node.name + '_sp'), axis=axis))
                return
            groups = [(outs[i:i + CAP], szs[i:i + CAP]) for i in range(0, len(outs), CAP)]
            gouts = [uniq(node.name + '_grp') for _ in groups]
            gsz = [int(np.sum(s)) for _, s in groups]
            sname = uniq(node.name + '_gsz')
            new_inits.append(numpy_helper.from_array(np.asarray(gsz, np.int64), sname))
            nodes.append(helper.make_node('Split', [data_in, sname], gouts,
                                          name=uniq(node.name + '_gs'), axis=axis))
            for go, (o, s) in zip(gouts, groups):
                emit(go, o, s)

        emit(node.input[0], list(node.output), list(sizes))
        return nodes

    def sum_to_adds(node):
        nodes = []
        acc = node.input[0]
        for i, nxt in enumerate(node.input[1:]):
            last = i == len(node.input) - 2
            out = node.output[0] if last else uniq(node.name + '_add')
            nodes.append(helper.make_node('Add', [acc, nxt], [out], name=uniq(node.name + '_a')))
            acc = out
        return nodes

    out_nodes = []
    stats = {'Concat': 0, 'Split': 0, 'Sum': 0}
    for n in g.node:
        if n.op_type == 'Concat' and len(n.input) > CAP:
            out_nodes.extend(cascade_concat(n))
            stats['Concat'] += 1
        elif n.op_type == 'Split' and len(n.output) > CAP and len(n.input) > 1 and n.input[1] in consts:
            out_nodes.extend(cascade_split(n))
            stats['Split'] += 1
        elif n.op_type == 'Sum' and len(n.input) >= 2:
            out_nodes.extend(sum_to_adds(n))
            stats['Sum'] += 1
        else:
            out_nodes.append(n)

    del g.node[:]
    g.node.extend(out_nodes)
    g.initializer.extend(new_inits)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='fp32 model.onnx: local path or https URL')
    ap.add_argument('--out', required=True, help='output .onnx path')
    ap.add_argument('--fp32', action='store_true', help='skip the float16 conversion')
    ap.add_argument('--no-verify', action='store_true', help='skip the numeric check')
    args = ap.parse_args()

    import onnxruntime as ort

    workdir = tempfile.mkdtemp(prefix='birefnet-export-')
    src = args.src
    if src.startswith('http'):
        local = os.path.join(workdir, 'src.onnx')
        print(f'downloading {src} …')
        urllib.request.urlretrieve(src, local)
        src = local

    # 1. offline constant fold / optimize (EP-agnostic BASIC level only —
    #    EXTENDED introduces CPU-specific fused ops WebGPU can't run).
    print('optimizing (constant fold)…')
    opt = os.path.join(workdir, 'opt.onnx')
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = opt
    ort.InferenceSession(src, so, providers=['CPUExecutionProvider'])

    # 2. cascade surgery
    print('patching wide fanouts…')
    model = onnx.load(opt, load_external_data=False)
    stats = cascade_patch(model)
    print(f'  rewrote {stats["Concat"]} Concat, {stats["Split"]} Split, {stats["Sum"]} Sum nodes')

    # 3. fp16
    if not args.fp32:
        print('converting to float16 (I/O stays float32)…')
        import warnings
        from onnxconverter_common import float16
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, args.out)
    print(f'saved {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)')

    if args.no_verify:
        return
    print('verifying against the source model (CPU, ~2 min)…')
    rng = np.random.default_rng(0)
    x = ((rng.random((1, 3, 1024, 1024)) - 0.45) / 0.225).astype(np.float32)

    def run(path):
        s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        return s.run(None, {s.get_inputs()[0].name: x})[0]

    sig = lambda v: 1 / (1 + np.exp(-v.astype(np.float32)))
    d = np.abs(sig(run(args.out)) - sig(run(src)))
    print(f'sigmoid-space max diff {d.max():.4g} (mean {d.mean():.3g}) — expect <5e-3 for fp16')
    if d.max() > 0.05:
        sys.exit('verification FAILED — output diverges from the source model')


if __name__ == '__main__':
    main()
