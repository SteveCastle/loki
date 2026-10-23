"""Reference dump of the OFFICIAL SAM 3.1 video tracker (multiplex predictor) on a folder of JPEG frames.

Usage (from tools/loki-segment):
    tools/ref-env/Scripts/python.exe -I ref/ref_video.py FRAMES_DIR OUT_DIR --obj 1:X,Y[,L][;X,Y,L...] [--obj 2:...] [--fp32]

Clicks are in video pixels on frame 0, one --obj per object. Dumps (f32, see ref_common.Dump):
  images            [N, 3, 1008, 1008]  the official preprocessed frames (so the native check can use the same input)
  f{t}_lowres       [n_obj, 288, 288]   tracker mask logits per frame (what the memory encoder sees, before video res)
  f{t}_objscore     [n_obj]
  f{t}_objptr       [16, 256]           multiplexed object pointers stored for the frame
  f{t}_maskmem      [5184, 256]         memory features stored for the frame (token-major)
  f{t}_pixmem       [5184, 256]         memory-conditioned features (propagated frames)
and meta.json with the prompts, video size and per-frame object ids.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402


def tok(x):
    if x.dim() == 4:
        x = x[0]
    return x.permute(1, 2, 0).reshape(-1, x.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--obj", action="append", default=[])
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    if a.fp32:
        ref_common.force_fp32()
        ac = torch.amp.autocast_mode.autocast
        ac.__enter__ = lambda self: None
        ac.__exit__ = lambda self, *args: None
    import sam3.model.decoder as decmod
    import sam3.model_builder as mb

    decmod.sdpa_kernel = lambda *a, **k: contextlib.nullcontext()
    with contextlib.redirect_stdout(io.StringIO()):
        trk = mb.build_sam3_multiplex_video_model(
            checkpoint_path=None, load_from_HF=False, multiplex_count=16, use_fa3=False, use_rope_real=False,
            compile=False, strict_state_dict_loading=False)
    sd = ref_common.load_state_dict()
    new = {}
    for k, v in sd.items():
        if k.startswith("tracker.model."):
            new[k[len("tracker.model."):]] = v
        elif k.startswith("detector.backbone.vision_backbone."):
            new["backbone.vision_backbone." + k[len("detector.backbone.vision_backbone."):]] = v
    missing, unexpected = trk.load_state_dict(new, strict=False)
    missing = [k for k in missing if "freqs_cis" not in k]
    print("missing", missing[:8], "unexpected", unexpected[:8], file=sys.stderr)
    trk = trk.cuda().eval()
    dump = ref_common.Dump(a.out)
    rec = {"frame": None}

    orig_track = trk.track_step
    orig_pmc = trk._prepare_memory_conditioned_features

    def track_step(*args, **kw):
        out = orig_track(*args, **kw)
        cur = out[0] if isinstance(out, tuple) else out
        t = kw.get("frame_idx")
        if isinstance(cur, dict) and t is not None:
            if cur.get("pred_masks") is not None:
                dump(f"f{t}_lowres", cur["pred_masks"][:, 0])
            if cur.get("object_score_logits") is not None:
                dump(f"f{t}_objscore", cur["object_score_logits"].flatten())
            if cur.get("obj_ptr") is not None:
                dump(f"f{t}_objptr", cur["obj_ptr"][0])
            if cur.get("maskmem_features") is not None:
                dump(f"f{t}_maskmem", tok(cur["maskmem_features"]))
        return out

    def pmc(*args, **kw):
        out = orig_pmc(*args, **kw)
        dump(f"f{kw.get('frame_idx')}_pixmem", tok(out))
        return out

    trk.track_step = track_step
    # upstream bit rot: forward_image's clone loop assumes every backbone_out entry is a dict, which breaks when
    # the detector (sam3) outputs are requested; the tracker does not use them
    _fi = trk.forward_image
    trk.forward_image = lambda img, **kw: _fi(img, **{**kw, "need_sam3_out": False})
    trk._prepare_memory_conditioned_features = pmc

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32):
        import sam3.model.video_tracking_multiplex_demo as vtmd
        from sam3.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo as Base

        # upstream bit rot: the demo passes loader options the loader no longer takes
        _lvf = vtmd.load_video_frames
        vtmd.load_video_frames = lambda *a, use_torchcodec=False, use_cv2=False, **k: _lvf(*a, **k)

        st = Base.init_state(trk, video_path=a.frames, offload_video_to_cpu=False, offload_state_to_cpu=False)
        imgs = st["images"]
        n = len(imgs) if a.max_frames <= 0 else min(a.max_frames, len(imgs))
        H, W = st["video_height"], st["video_width"]
        dump("images", torch.stack([imgs[i].float() for i in range(n)]))
        meta = {"width": W, "height": H, "frames": n, "objects": {}, "fp32": a.fp32}
        for spec in a.obj:
            oid, pts = spec.split(":")
            coords, labels = [], []
            for p in pts.split(";"):
                v = [float(t) for t in p.split(",")]
                coords.append([v[0] / W, v[1] / H])
                labels.append(int(v[2]) if len(v) > 2 else 1)
            meta["objects"][oid] = {"points": [[c[0] * W, c[1] * H, l] for c, l in zip(coords, labels)]}
            trk.add_new_points(st, frame_idx=0, obj_id=int(oid), points=torch.tensor(coords, dtype=torch.float32),
                               labels=torch.tensor(labels, dtype=torch.int32), clear_old_points=True, rel_coordinates=True)
        Base.propagate_in_video_preflight(trk, st, run_mem_encoder=True)
        for fi, obj_ids, low_res, video_res in Base.propagate_in_video(trk, st, start_frame_idx=0, max_frame_num_to_track=n,
                                                                     reverse=False, tqdm_disable=True):
            dump(f"f{fi}_out_lowres", low_res[:, 0])
            meta.setdefault("out_obj_ids", {})[fi] = [int(i) for i in obj_ids]
            if fi + 1 >= n:
                break
        c0 = st["output_dict"]["cond_frame_outputs"][0]
        dump("c0_lowres", c0["pred_masks"][:, 0])
        dump("c0_objscore", c0["object_score_logits"].flatten())
        dump("c0_objptr", c0["obj_ptr"][0])
        dump("c0_maskmem", tok(c0["maskmem_features"]))
        dump("c0_maskmem_pos", tok(c0["maskmem_pos_enc"][-1]))
        meta["c0_conditioning_objects"] = sorted(int(i) for i in c0.get("conditioning_objects", []))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("ok", a.out)


if __name__ == "__main__":
    main()
