#!/usr/bin/env python3
"""
Evaluate YOLO-seg and DINOv3+LoRA on the SAME metrics, on our val set or zero-shot on DENTEX.

WHY
---
The two architectures report incomparable numbers by default: YOLO gives mask mAP50-95 (COCO,
averaged over IoU 0.50:0.05:0.95), DINOv3 gives mIoU (semantic overlap at a single threshold).
mAP is systematically harsher, so "0.810 mIoU vs 0.730 mAP" says nothing about which model is
better. This computes both metrics for both models from the same predictions.

The bridge is the property that makes this whole task tractable: each FDI class occurs at most
once per image, so a semantic prediction converts to exactly one instance per predicted class,
and an instance prediction collapses to a semantic map. Neither direction loses information.
Confidence for a DINOv3 "instance" is the mean softmax probability of that class over its own
predicted region -- the natural analogue of a detector's score.

DENTEX zero-shot: DENTEX labels teeth as (quadrant, position), so FDI = quadrant*10 + position,
which maps onto our 32 classes exactly (verified: all 32 codes present, none outside).

Usage:
  python evaluate_models.py --model yolo  --weights runs/ablate_1280/weights/best.pt
  python evaluate_models.py --model dinov3 --weights runs/dinov3_vitb_lora/best.pt
  python evaluate_models.py --model yolo --weights ... --dataset dentex
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
LABELS_MAP = os.path.join(BASE_DIR, "shared", "labels_map.json")
NUM_CLASSES = 32


def load_fdi_maps():
    with open(LABELS_MAP) as f:
        raw = json.load(f)
    idx_to_fdi = {int(k): int(v["fdi"]) for k, v in raw.items()}
    idx_to_name = {int(k): v["name"] for k, v in raw.items()}
    return idx_to_fdi, {v: k for k, v in idx_to_fdi.items()}, idx_to_name


# --------------------------------------------------------------- ground truth
def gt_from_yolo(data_dir, split="val"):
    """{image_id: (path, {class: mask_fn})} from our YOLO polygon labels."""
    img_dir = os.path.join(data_dir, "images", split)
    lbl_dir = os.path.join(data_dir, "labels", split)
    items = []
    for f in sorted(os.listdir(img_dir)):
        iid = os.path.splitext(f)[0]
        lp = os.path.join(lbl_dir, f"{iid}.txt")
        if not os.path.exists(lp):
            continue
        polys = defaultdict(list)
        for line in open(lp):
            p = line.split()
            if len(p) < 7:
                continue
            polys[int(p[0])].append(np.array([float(x) for x in p[1:]],
                                             dtype=np.float32).reshape(-1, 2))
        items.append((iid, os.path.join(img_dir, f), dict(polys), True))
    return items


def gt_from_dentex(dentex_root):
    """DENTEX quadrant-enumeration -> our class indices. Coordinates are absolute pixels."""
    _, fdi_to_idx, _ = load_fdi_maps()
    jp = os.path.join(dentex_root, "train_quadrant_enumeration.json")
    xr = os.path.join(dentex_root, "xrays")
    d = json.load(open(jp))
    c1 = {c["id"]: int(c["name"]) for c in d["categories_1"]}
    c2 = {c["id"]: int(c["name"]) for c in d["categories_2"]}
    by_img = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        if "category_id_1" not in a or "category_id_2" not in a:
            continue
        fdi = c1[a["category_id_1"]] * 10 + c2[a["category_id_2"]]
        idx = fdi_to_idx.get(fdi)
        seg = a.get("segmentation") or []
        if idx is None or not seg or len(seg[0]) < 6:
            continue
        pts = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
        by_img[a["image_id"]][idx].append(pts)
    items = []
    for im in d["images"]:
        p = os.path.join(xr, im["file_name"])
        if os.path.exists(p) and im["id"] in by_img:
            items.append((os.path.splitext(im["file_name"])[0], p,
                          dict(by_img[im["id"]]), False))
    return items


def image_size(path):
    """Dimensions without decoding pixels — decoding a 2900x1400 JPEG just to read its
    shape was a large part of the I/O load."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return h, w


def rasterise(polys, w, h, normalised):
    out = {}
    for cls, plist in polys.items():
        m = np.zeros((h, w), dtype=np.uint8)
        for pts in plist:
            q = pts.copy()
            if normalised:
                q[:, 0] *= w
                q[:, 1] *= h
            cv2.fillPoly(m, [q.astype(np.int32)], 1)
        if m.any():
            out[cls] = m
    return out


# ---------------------------------------------------------------- predictions
def _encode(mask):
    """Store masks RLE-compressed, never as dense arrays.

    A dense uint8 mask at DENTEX resolution is ~4 MB; holding up to 32 of them for each of
    634 images is ~80 GB and will take the machine down. RLE of a single tooth is a few KB,
    and pycocotools needs RLE anyway, so nothing is wasted converting early.
    """
    from pycocotools import mask as maskutil
    rle = maskutil.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return rle


def _decode(rle):
    from pycocotools import mask as maskutil
    r = dict(rle)
    r["counts"] = r["counts"].encode()
    return maskutil.decode(r)


def predict_yolo(weights, items, imgsz, conf=0.25, postproc="greedy", refiner=None):
    """postproc: how the FDI-uniqueness constraint is resolved.

    greedy -- keep the highest-scoring instance per class, each class decided in isolation.
    assign -- solve the whole image at once as a linear assignment, so which instance keeps a
              contested class is decided by arch geometry as well as confidence, and a losing
              duplicate can be reassigned to a free class its position supports.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    if postproc == "assign":
        from arch_assignment import assign_fdi
    preds = {}
    for iid, path, _, _ in items:
        r = model.predict(path, imgsz=imgsz, conf=conf, verbose=False, retina_masks=True)[0]
        h, w = r.orig_shape
        raw = []
        if r.masks is not None:
            for m, c, s, b in zip(r.masks.data.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.xywhn.cpu().numpy()):
                mm = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
                raw.append({"class_id": int(c), "score": float(s), "mask": mm,
                            "cx": float(b[0]), "cy": float(b[1])})
        if postproc == "assign":
            kept = assign_fdi(raw)
        else:
            best = {}
            for inst in raw:
                c = inst["class_id"]
                if c not in best or inst["score"] > best[c]["score"]:
                    best[c] = inst
            kept = list(best.values())
        if refiner is not None and kept:
            boxes, masks_in = [], []
            for k in kept:
                ys, xs = np.nonzero(k["mask"])
                if len(xs) == 0:
                    boxes.append([0, 0, 1, 1])
                else:
                    boxes.append([float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())])
                masks_in.append(k["mask"])
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            refined = refiner.refine_batch(rgb, boxes, masks_in)
            for k, m in zip(kept, refined):
                k["mask"] = m
        preds[iid] = {int(k["class_id"]): (_encode(k["mask"]), k["score"]) for k in kept}
        del r
    return preds


def predict_dinov3(weights, items, width, height):
    import torch
    from train_dinov3_lora import DINOv3Segmenter
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    model_name = saved.get("model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    model = DINOv3Segmenter(model_name, lora_r=saved.get("lora_r", 16))
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    W = width or saved.get("width", 1024)
    H = height or saved.get("height", 528)
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    preds = {}
    with torch.no_grad():
        for iid, path, _, _ in items:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            oh, ow = im.shape[:2]
            x = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            x = np.repeat(x[None], 3, 0).astype(np.float32) / 255.0
            x = torch.from_numpy((x - mean) / std)[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            prob = logits.float().softmax(1)[0].cpu().numpy()
            lab = prob.argmax(0)
            out = {}
            for c in range(1, NUM_CLASSES + 1):        # 0 is background
                m = (lab == c)
                if m.sum() < 20:                        # ignore speckle
                    continue
                score = float(prob[c][m].mean())        # analogue of a detector score
                mm = cv2.resize(m.astype(np.uint8), (ow, oh),
                                interpolation=cv2.INTER_NEAREST)
                out[c - 1] = (_encode(mm), score)
            preds[iid] = out
    return preds


# -------------------------------------------------------------------- metrics
def predict_mask2former(weights, items):
    """Mask2Former predictions, resized to ORIGINAL image resolution.

    Evaluating on the network's own 1280x672 grid flatters every model: boundary errors that
    span several pixels at full resolution partly vanish when downsampled. YOLO was scored at
    original resolution, so Mask2Former must be too or the comparison is meaningless.

    One instance per FDI class is kept, matching the constraint applied to YOLO, so the
    comparison isolates the architecture rather than the post-processing.
    """
    import torch
    from transformers import Mask2FormerForUniversalSegmentation
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    W = int(saved.get("width", 1280))
    H = int(saved.get("height", 672))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        saved.get("checkpoint", "facebook/mask2former-swin-tiny-coco-instance"),
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    model.load_state_dict(ck["model"])
    model.cuda().eval()

    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    preds = {}
    with torch.no_grad():
        for iid, path, _, _ in items:
            im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            oh, ow = im.shape[:2]
            x = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            x = np.repeat(x[None], 3, 0).astype(np.float32) / 255.0
            x = torch.from_numpy((x - mean) / std)[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(pixel_values=x)
            cls = out.class_queries_logits.float()[0].softmax(-1)[:, :-1]   # drop no-object
            masks = out.masks_queries_logits.float()
            masks = torch.nn.functional.interpolate(
                masks, size=(H, W), mode="bilinear", align_corners=False)[0]
            score, qcls = cls.max(-1)
            best = {}
            for q in range(cls.shape[0]):
                c, sc = int(qcls[q]), float(score[q])
                if sc < 0.5:
                    continue
                if c not in best or sc > best[c][1]:
                    best[c] = (q, sc)
            got = {}
            for c, (q, sc) in best.items():
                m = (masks[q].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                if m.sum() == 0:
                    continue
                m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
                got[c] = (_encode(m), sc)
            preds[iid] = got
    return preds


def compute_miou(items, preds):
    """Dataset-level IoU per class, then mean over classes that actually occur."""
    inter = np.zeros(NUM_CLASSES, np.float64)
    union = np.zeros(NUM_CLASSES, np.float64)
    seen = np.zeros(NUM_CLASSES, bool)
    for iid, path, polys, norm in items:
        h, w = image_size(path)
        gt = rasterise(polys, w, h, norm)
        pr = preds.get(iid, {})
        for c in range(NUM_CLASSES):
            g = gt.get(c)
            rle = pr.get(c, (None, 0))[0]
            if g is None and rle is None:
                continue
            if g is not None:
                seen[c] = True
            g = np.zeros((h, w), np.uint8) if g is None else g
            p = np.zeros((h, w), np.uint8) if rle is None else _decode(rle)
            inter[c] += np.logical_and(g, p).sum()
            union[c] += np.logical_or(g, p).sum()
        del gt
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return float(iou[seen].mean()), {c: float(iou[c]) for c in range(NUM_CLASSES) if seen[c]}


def compute_coco_map(items, preds):
    """COCO segm mAP via pycocotools, built from the same masks used for mIoU."""
    from pycocotools import mask as maskutil
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    aid = 1
    for n, (iid, path, polys, norm) in enumerate(items, 1):
        h, w = image_size(path)
        images.append({"id": n, "file_name": iid, "width": w, "height": h})
        for c, m in rasterise(polys, w, h, norm).items():
            rle = maskutil.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode()
            anns.append({"id": aid, "image_id": n, "category_id": int(c) + 1,
                         "segmentation": rle, "area": float(m.sum()),
                         "bbox": list(maskutil.toBbox(rle)), "iscrowd": 0})
            aid += 1
        for c, (rle, s) in preds.get(iid, {}).items():
            dets.append({"image_id": n, "category_id": int(c) + 1,
                         "segmentation": dict(rle), "score": float(s)})

    gt = {"images": images, "annotations": anns,
          "categories": [{"id": c + 1, "name": str(c)} for c in range(NUM_CLASSES)]}
    tmp = os.path.join(BASE_DIR, "_eval_gt.json")
    with open(tmp, "w") as f:
        json.dump(gt, f)
    coco = COCO(tmp)
    if not dets:
        return 0.0, 0.0
    dt = coco.loadRes(dets)
    e = COCOeval(coco, dt, "segm")
    e.evaluate(); e.accumulate(); e.summarize()
    os.remove(tmp)
    return float(e.stats[0]), float(e.stats[1])       # mAP50-95, mAP50


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["yolo", "dinov3", "mask2former"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=["val", "dentex"], default="val")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--dentex-root",
                    default=os.path.expanduser(
                        "~/opg-teeth/external/dentex/unpacked/training_data/quadrant_enumeration"))
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    ap.add_argument("--width", type=int, default=None, help="DINOv3 width (default: as trained)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--refine", choices=["none", "sam2"], default="none",
                    help="boundary refinement applied to predicted masks")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-min-iou", type=float, default=0.5,
                    help="reject a refined mask that disagrees with the original below this")
    ap.add_argument("--postproc", choices=["greedy", "assign"], default="greedy",
                    help="how to resolve the one-instance-per-FDI-class constraint")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = (gt_from_dentex(args.dentex_root) if args.dataset == "dentex"
             else gt_from_yolo(args.data, "val"))
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("no evaluation items found")
    print(f"{args.model} on {args.dataset}: {len(items)} images")

    refiner = None
    if args.refine == "sam2":
        from sam2_refine import Sam2Refiner
        print(f"loading {args.sam2_model} for boundary refinement")
        refiner = Sam2Refiner(args.sam2_model, min_iou=args.sam2_min_iou)

    if args.model == "yolo":
        preds = predict_yolo(args.weights, items, args.imgsz, postproc=args.postproc,
                             refiner=refiner)
    elif args.model == "mask2former":
        preds = predict_mask2former(args.weights, items)
    else:
        preds = predict_dinov3(args.weights, items, args.width, args.height)

    if refiner is not None:
        print("  " + refiner.report())

    miou, per_class = compute_miou(items, preds)
    map5095, map50 = compute_coco_map(items, preds)

    _, _, names = load_fdi_maps()
    print("\n" + "=" * 64)
    print(f"  {args.model.upper()}  |  {args.dataset}  |  {len(items)} images  |  "
          f"postproc={args.postproc}  refine={args.refine}")
    print(f"    mIoU          : {miou:.4f}")
    print(f"    mask mAP50-95 : {map5095:.4f}")
    print(f"    mask mAP50    : {map50:.4f}")
    print("=" * 64)
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:6]
    print("  lowest-IoU classes:")
    for c, v in worst:
        print(f"    {names.get(c, c):34s} {v:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "dataset": args.dataset,
                       "n_images": len(items), "mIoU": miou,
                       "mAP50_95": map5095, "mAP50": map50,
                       "per_class_iou": {names.get(c, str(c)): v
                                         for c, v in per_class.items()}}, f, indent=2)
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
