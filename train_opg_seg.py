#!/usr/bin/env python3
"""
Train a YOLO instance-segmentation model for 32-class FDI tooth segmentation on OPGs.

This is both the AI Dentify production model and the baseline row of the paper, so the
defaults below are chosen for the dataset rather than copied from the YOLO examples. Three
of them differ from the stock settings and matter a great deal:

  fliplr=0.0   MANDATORY. A horizontal flip mirrors the arch, which turns tooth 18 (upper
               RIGHT third molar) into tooth 28 (upper LEFT third molar). Ultralytics
               defaults this to 0.5, which would silently corrupt half of every epoch's
               labels. There is no "flip the image and remap the classes" option in the
               stock loader, so the augmentation is simply off. This is the single most
               destructive default for FDI-numbered data.

  imgsz=1024   The images are 1615x840. At the stock 640 a third molar spans very few
               pixels, and those are exactly the classes with the fewest instances
               (~830 vs ~1400 for central incisors) and the worst AP.

  rect=True    Aspect ratio is 1.92:1. Square letterboxing wastes ~half the frame on
               padding; rectangular batches keep the resolution where the anatomy is.

Photometric augmentation is pushed above stock because the deployment failure mode is
domain shift across scanners and exposure settings, not geometry. Geometric augmentation
stays conservative: panoramic acquisition geometry is highly standardised, so large
rotations and shears produce images unlike anything the model will ever see.

Usage:
  python train_opg_seg.py                          # full run, ~3-4 h on an A10G
  python train_opg_seg.py --smoke                  # 3 epochs, verifies the pipeline
  python train_opg_seg.py --model yolo11l-seg.pt --epochs 150
  python train_opg_seg.py --folds 5                # 5-fold CV for the paper
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset", "data.yaml")
SEED = 42


def check_environment():
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\n  pip install ultralytics torch torchvision")
    if torch.cuda.is_available():
        name = torch.backends.cuda and torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: cuda ({name}, {mem:.0f} GB)")
        return "0"
    if torch.backends.mps.is_available():
        print("Device: mps (Apple Silicon) — expect roughly 4-6x an A10G's wall clock")
        return "mps"
    print("Device: cpu — this will be unusably slow for training")
    return "cpu"


def verify_labels(data_yaml):
    """Fail loudly before burning GPU hours on a broken dataset."""
    root = os.path.dirname(os.path.abspath(data_yaml))
    problems, counts = [], Counter()
    for split in ("train", "val"):
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(img_dir):
            problems.append(f"missing {img_dir}")
            continue
        imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
        if imgs - lbls:
            problems.append(f"{split}: {len(imgs - lbls)} image(s) without a label")
        if lbls - imgs:
            problems.append(f"{split}: {len(lbls - imgs)} label(s) without an image")
        for name in lbls:
            with open(os.path.join(lbl_dir, f"{name}.txt")) as f:
                for n, line in enumerate(f, 1):
                    parts = line.split()
                    if not parts:
                        continue
                    cls, coords = int(parts[0]), [float(x) for x in parts[1:]]
                    counts[cls] += 1
                    if not 0 <= cls <= 31:
                        problems.append(f"{split}/{name}.txt:{n} class {cls} outside 0-31")
                    if len(coords) < 6 or len(coords) % 2:
                        problems.append(f"{split}/{name}.txt:{n} {len(coords)} coords")
                    if any(c < 0 or c > 1 for c in coords):
                        problems.append(f"{split}/{name}.txt:{n} coord outside [0,1]")
    if problems:
        print("Dataset problems found:")
        for p in problems[:20]:
            print(f"  {p}")
        sys.exit(f"\n{len(problems)} problem(s). Fix before training.")
    missing = [c for c in range(32) if counts[c] == 0]
    if missing:
        print(f"  WARNING: {len(missing)} class(es) have no instances at all: {missing}")
    print(f"Dataset OK: {sum(counts.values())} polygons across {len(counts)} classes")


def train_args(args, device, name):
    return dict(
        data=args.data, epochs=3 if args.smoke else args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=device, project=args.project, name=name,
        seed=SEED, deterministic=True, val=True, plots=True, patience=args.patience,
        rect=True,          # 1.92:1 images; square letterboxing wastes half the frame
        # --- geometry: conservative, panoramic acquisition is standardised ---
        fliplr=0.0,         # NEVER raise this: mirroring swaps left/right FDI labels
        flipud=0.0,
        degrees=3.0, translate=0.05, scale=0.25, shear=1.0, perspective=0.0,
        mosaic=args.mosaic, close_mosaic=10, copy_paste=0.0, mixup=0.0,
        # --- photometry: pushed up, this is where real domain shift lives ---
        hsv_h=0.0,          # radiographs are greyscale; hue rotation is meaningless
        hsv_s=0.0,
        hsv_v=0.5,          # exposure/brightness varies most across scanners
        # --- optimisation ---
        optimizer="AdamW", lr0=0.001, lrf=0.01, warmup_epochs=3.0,
        cos_lr=True, overlap_mask=False,  # teeth are adjacent but must not merge
    )


def summarise(metrics, labels):
    """Print per-class mask AP so third-molar performance stays visible."""
    try:
        maps = metrics.seg.maps
        print(f"\n  mask mAP50-95: {metrics.seg.map:.4f}   mAP50: {metrics.seg.map50:.4f}")
        ranked = sorted(enumerate(maps), key=lambda kv: kv[1])
        print("  weakest 6 classes:")
        for idx, ap in ranked[:6]:
            print(f"    {labels.get(idx, idx):32s} {ap:.4f}")
    except Exception as e:
        print(f"  (per-class summary unavailable: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default="yolo11m-seg.pt",
                    help="yolo11n/s/m/l/x-seg.pt (default m: best accuracy/latency trade)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8, help="lower to 4 if CUDA OOMs at 1024")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--mosaic", type=float, default=0.4,
                    help="mosaic hurts when every image has the same global layout")
    ap.add_argument("--project", default=os.path.join(BASE_DIR, "runs"))
    ap.add_argument("--name", default="opg_seg")
    ap.add_argument("--folds", type=int, default=0, help="k-fold CV instead of the fixed split")
    ap.add_argument("--smoke", action="store_true", help="3 epochs to prove the pipeline")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"No data.yaml at {args.data}. Run download_merged_opg_dataset.py first.")

    device = check_environment()
    if not args.skip_verify:
        verify_labels(args.data)

    import yaml
    from ultralytics import YOLO
    with open(args.data) as f:
        labels = yaml.safe_load(f).get("names", {})

    if args.folds:
        sys.exit("--folds is not implemented yet; run the fixed split first, then add CV "
                 "once the baseline number is known.")

    name = f"{args.name}_smoke" if args.smoke else args.name
    print(f"\nTraining {args.model} at imgsz={args.imgsz}, batch={args.batch}, "
          f"{'3 (smoke)' if args.smoke else args.epochs} epochs")
    print("  fliplr=0.0 — horizontal flip disabled, it would invert every FDI label\n")

    model = YOLO(args.model)
    results = model.train(**train_args(args, device, name))

    out = os.path.join(args.project, name)
    print(f"\nDone. Weights: {out}/weights/best.pt")
    summarise(results, labels)

    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)
    print(f"Config written to {out}/run_config.json")


if __name__ == "__main__":
    main()
