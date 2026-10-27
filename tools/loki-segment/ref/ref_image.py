"""Reference dump of the OFFICIAL SAM 3.1 image path (facebookresearch/sam3) for one image.

Usage (from tools/loki-segment):
    tools/ref-env/Scripts/python.exe -I ref/ref_image.py IMAGE OUT_DIR [--fp32] [--text T ...] [--point X,Y[,L] ...] [--box X,Y,W,H]

Coordinates are input-image pixels (like the CLI). Every tensor is written as raw f32 with a JSON index
(see ref_common.Dump), token-major ([H*W, C]) for feature maps so the native engine can compare directly.
Without --fp32 the model runs under bf16 autocast exactly like the official predictors.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_common  # noqa: E402


def tok(x):
    """[1, C, H, W] or [C, H, W] -> [H*W, C]"""
    if x.dim() == 4:
        x = x[0]
    return x.permute(1, 2, 0).reshape(-1, x.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--point", action="append", default=[])
    ap.add_argument("--box", default=None)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with contextlib.redirect_stdout(io.StringIO()):
        pred = ref_common.build_predictor()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    det = pred.model.detector
    trk = pred.model.tracker.model
    from sam3.model.sam3_image_processor import Sam3Processor

    proc = Sam3Processor(det)
    dump = ref_common.Dump(a.out)
    img = Image.open(a.image).convert("RGB")
    W, H = img.size
    meta = {"width": W, "height": H, "fp32": a.fp32, "text": a.text, "points": a.point, "box": a.box}

    if a.fp32:
        ref_common.force_fp32()
    ctx = torch.autocast("cuda", enabled=not a.fp32, dtype=torch.bfloat16)
    with torch.inference_mode(), ctx:
        # ---- preprocessing (Sam3Processor.transform) and backbone, captured stage by stage
        from torchvision.transforms import v2

        x = v2.functional.to_image(img).cuda()
        x = proc.transform(x).unsqueeze(0)
        dump("image", x[0])
        vb = det.backbone.vision_backbone
        trunk_out = vb.trunk(x)[-1]
        dump("trunk", tok(trunk_out))
        for name, convs in (("sam3", vb.convs), ("interactive", vb.interactive_convs), ("propagation", vb.propagation_convs)):
            for i, c in enumerate(convs):
                dump(f"neck_{name}_{i}", tok(c(trunk_out)))
        st = proc.set_image(img)

        # ---- text + detector
        for ti, t in enumerate(a.text):
            st = proc.set_text_prompt(state=st, prompt=t)
            tok_ids = det.backbone.language_backbone.tokenizer([t], context_length=32)[0]
            meta[f"tokens_{ti}"] = tok_ids.tolist()
            tf = det.backbone.forward_text([t], device="cuda")
            dump(f"text{ti}_features", tf["language_features"][:, 0])  # [32, 256]
            dump(f"text{ti}_mask", tf["language_mask"][0].float())  # 1 = padding
            # rerun the grounding with hooks on the internals
            caps = {}
            hooks = [
                det.geometry_encoder.register_forward_hook(lambda m, i, o: caps.__setitem__("geo", o)),
                det.transformer.encoder.register_forward_hook(lambda m, i, o: caps.__setitem__("enc", o)),
                det.transformer.decoder.register_forward_hook(lambda m, i, o: caps.__setitem__("dec", o)),
                det.segmentation_head.register_forward_hook(lambda m, i, o: caps.__setitem__("seg", o)),
            ]
            out = det.forward_grounding(
                backbone_out=st["backbone_out"], find_input=proc.find_stage,
                geometric_prompt=st["geometric_prompt"], find_target=None)
            for h in hooks:
                h.remove()
            geo_feats, geo_mask = caps["geo"]
            dump(f"text{ti}_geo", geo_feats[:, 0])
            dump(f"text{ti}_memory", caps["enc"]["memory"][:, 0])
            hs, refb, pres, _ = caps["dec"]
            dump(f"text{ti}_hs", hs[-1, :, 0])  # last layer, normed, [200, 256]
            dump(f"text{ti}_hs_all", hs[:, :, 0])
            dump(f"text{ti}_refboxes", refb[:, :, 0])
            dump(f"text{ti}_presence", pres[:, 0])
            dump(f"text{ti}_logits", out["pred_logits"][0, :, 0])  # joint (class x presence) logits
            dump(f"text{ti}_boxes", out["pred_boxes"][0])  # cxcywh normalized
            dump(f"text{ti}_masks", out["pred_masks"][0])  # [200, 288, 288]
            dump(f"text{ti}_semantic", caps["seg"]["semantic_seg"][0, 0])
            meta[f"text{ti}_scores_kept"] = st["scores"].float().cpu().tolist()
            meta[f"text{ti}_boxes_kept_xyxy"] = st["boxes"].float().cpu().tolist()

        # ---- interactive (SAM) head: clicks and/or a box on the image
        if a.point or a.box:
            feats = st["backbone_out"]["interactive"]["backbone_fpn"]
            feats = [getattr(f, "tensors", f) for f in feats]
            dec = trk.interactive_sam_mask_decoder
            s0 = dec.conv_s0(feats[0])
            s1 = dec.conv_s1(feats[1])
            dump("sam_s0", tok(s0))
            dump("sam_s1", tok(s1))
            pix = feats[2] + trk.interactivity_no_mem_embed.view(1, -1, 1, 1)
            coords, labels = [], []
            for p in a.point:
                v = [float(t) for t in p.split(",")]
                coords.append([v[0] / W * 1008, v[1] / H * 1008])
                labels.append(int(v[2]) if len(v) > 2 else 1)
            if a.box:
                bx, by, bw, bh = [float(t) for t in a.box.split(",")]
                coords += [[bx / W * 1008, by / H * 1008], [(bx + bw) / W * 1008, (by + bh) / H * 1008]]
                labels += [2, 3]
            pc = torch.tensor([coords], device="cuda", dtype=torch.float32)
            pl = torch.tensor([labels], device="cuda", dtype=torch.int32)
            n_pts = len(a.point) + (2 if a.box else 0)
            multimask = 0 < n_pts <= 1
            sparse, dense = trk.interactive_sam_prompt_encoder(points=(pc, pl), boxes=None, masks=None)
            dump("sam_sparse", sparse[0])
            lowres, ious, toks, objs = dec(
                image_embeddings=pix, image_pe=trk.interactive_sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=multimask, repeat_image=True, high_res_features=[s0, s1])
            dump("sam_lowres", lowres[0])  # [M, 288, 288]
            dump("sam_ious", ious[0])
            dump("sam_obj", objs[0])
            meta["sam_multimask"] = multimask

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("ok", a.out)


if __name__ == "__main__":
    main()
