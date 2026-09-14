"""Dump per-(image, class) evaluation records so metrics can be recomputed on resamples.

The aggregate harness in evaluate_models.py collapses everything to three numbers, which is
enough to report but not enough to test. Bootstrapping mAP needs the ability to recompute it
on an arbitrary resample of images, and re-running inference per resample is not affordable.

FDI uniqueness makes the cheap route exact. Each class occurs at most once per image, and the
post-processing enforces at most one detection per class per image, so every (image, class)
cell holds at most one ground truth and at most one detection. COCO's greedy IoU matching has
nothing to disambiguate, and the whole evaluation reduces to a table of independent cells:

    (image, class) -> was there a GT, was there a detection, at what score, at what IoU

From that table both mAP (via exact 101-point COCO interpolation) and dataset-level mIoU are
recomputable for any subset or resample of images, which is what the bootstrap needs.

Writes one JSON per (model, dataset). Consumed by stats_tests.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_models as em


def build_records(items, preds):
    """One record per (image, class) cell that holds a GT, a detection, or both."""
    recs = []
    for i, (iid, path, polys, norm) in enumerate(items):
        h, w = em.image_size(path)
        gt = em.rasterise(polys, w, h, norm)
        pr = preds.get(iid, {})
        for c in set(gt) | set(pr):
            g = gt.get(c)
            entry = pr.get(c)
            rle, score = (entry if entry is not None else (None, None))

            if g is not None and rle is not None:
                p = em._decode(rle)
                inter = int(np.logical_and(g, p).sum())
                union = int(np.logical_or(g, p).sum())
                iou = inter / union if union else 0.0
            elif g is not None:
                inter, union, iou = 0, int(g.sum()), 0.0
            else:
                p = em._decode(rle)
                inter, union, iou = 0, int(p.sum()), 0.0

            recs.append({
                "i": i,
                "c": int(c),
                "gt": int(g is not None),
                "score": (float(score) if score is not None else None),
                "iou": float(iou),
                "inter": inter,
                "union": union,
            })
        del gt
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["yolo", "dinov3", "mask2former"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", choices=["val", "dentex"], default="val")
    ap.add_argument("--data", default=em.DEFAULT_DATA)
    ap.add_argument("--dentex-root", default=os.path.expanduser(
        "~/opg-teeth/external/dentex/unpacked/training_data/quadrant_enumeration"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--postproc", choices=["greedy", "assign"], default="greedy")
    ap.add_argument("--refine", choices=["none", "sam2"], default="none")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam2-min-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True, help="name for the output file")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/opg-teeth/records"))
    args = ap.parse_args()

    items = (em.gt_from_dentex(args.dentex_root) if args.dataset == "dentex"
             else em.gt_from_yolo(args.data, "val"))
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("no evaluation items found")
    print(f"{args.model} on {args.dataset}: {len(items)} images", flush=True)

    refiner = None
    if args.refine == "sam2":
        from sam2_refine import Sam2Refiner
        print(f"loading {args.sam2_model} for boundary refinement", flush=True)
        refiner = Sam2Refiner(args.sam2_model, min_iou=args.sam2_min_iou)

    if args.model == "yolo":
        preds = em.predict_yolo(args.weights, items, args.imgsz, postproc=args.postproc,
                                refiner=refiner)
    elif args.model == "mask2former":
        preds = em.predict_mask2former(args.weights, items)
    else:
        preds = em.predict_dinov3(args.weights, items, args.width, args.height)

    recs = build_records(items, preds)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "tag": args.tag,
            "postproc": args.postproc,
            "images": [iid for iid, _, _, _ in items],
            "n_images": len(items),
            "records": recs,
        }, f)
    print(f"  {len(recs)} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
