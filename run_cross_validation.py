#!/usr/bin/env python3
"""
5-fold cross-validation for the OPG tooth-segmentation model.

WHY
---
Every number reported so far comes from a single 144-image validation split. At this dataset
size that carries visible variance -- the resolution ablation moved mask mAP50-95 by 0.019
between 1024 and 1280, which is the same order as split noise might be. Cross-validation
turns each headline figure into a mean with a spread, which is what a reviewer will expect
before believing a 0.019 difference means anything.

TWO THINGS THIS DOES THAT A NAIVE SPLIT WOULD NOT
-------------------------------------------------
1. GROUPED folds. Exhaustive pixel correlation over the dataset found HOPG_261/HOPG_271 at
   r=0.962 with a 7px offset and no monotone intensity relation -- almost certainly the same
   patient imaged twice, not a duplicated file. Splitting such a pair across folds leaks:
   the model would be scored on a patient it trained on. Images are therefore grouped by
   correlation before folding, and a group never straddles folds.

2. STRATIFIED folds. The set is ~82% regular OPGs and ~18% hard cases. A purely random fold
   can under-represent the hard subset badly enough to move its metrics on its own, so folds
   are balanced by track.

Usage:
  python run_cross_validation.py --build-only     # inspect the fold assignment first
  python run_cross_validation.py                  # build folds and train all 5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
EMB_H, EMB_W = 64, 128
GROUP_THRESHOLD = 0.95      # below the 0.99 duplicate bar: catches same-patient repeats too
SEED = 42


def embed(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    v = cv2.resize(im, (EMB_W, EMB_H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def collect(data_dir):
    items = []
    for split in ("train", "val"):
        d = os.path.join(data_dir, "images", split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                iid = os.path.splitext(f)[0]
                items.append({
                    "id": iid,
                    "img": os.path.join(d, f),
                    "lbl": os.path.join(data_dir, "labels", split, f"{iid}.txt"),
                    "track": "hard" if iid.startswith("HOPG") else "regular",
                })
    return [i for i in items if os.path.exists(i["lbl"])]


def group_related(items):
    """Union-find over correlation, so same-patient repeats share a fold."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        embs = list(ex.map(embed, [i["img"] for i in items]))
    keep = [k for k, e in enumerate(embs) if e is not None]
    A = np.stack([embs[k] for k in keep])
    C = A @ A.T
    np.fill_diagonal(C, -1.0)

    parent = list(range(len(keep)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ii, jj = np.where(np.triu(C, 1) >= GROUP_THRESHOLD)
    for a, b in zip(ii, jj):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = defaultdict(list)
    for idx in range(len(keep)):
        groups[find(idx)].append(items[keep[idx]])
    return list(groups.values())


def assign_folds(groups, k):
    """Greedy balanced assignment, per track, largest groups first."""
    rng = np.random.RandomState(SEED)
    folds = [[] for _ in range(k)]
    for track in ("hard", "regular"):
        sel = [g for g in groups if g[0]["track"] == track]
        rng.shuffle(sel)
        sel.sort(key=len, reverse=True)          # place big groups while slack remains
        for g in sel:
            counts = [sum(1 for x in f if x["track"] == track) for f in folds]
            folds[int(np.argmin(counts))].extend(g)
    return folds


def build_fold_dirs(folds, out_root, names):
    """Symlink rather than copy: 5 folds x ~190 MB of duplication is pointless."""
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    paths = []
    for k, val_items in enumerate(folds):
        train_items = [x for j, f in enumerate(folds) if j != k for x in f]
        root = os.path.join(out_root, f"fold_{k}")
        for split, items in (("train", train_items), ("val", val_items)):
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(root, sub, split), exist_ok=True)
            for it in items:
                os.symlink(it["img"], os.path.join(root, "images", split,
                                                   os.path.basename(it["img"])))
                os.symlink(it["lbl"], os.path.join(root, "labels", split,
                                                   os.path.basename(it["lbl"])))
        yml = os.path.join(root, "data.yaml")
        with open(yml, "w") as f:
            f.write(f"# fold {k} of {len(folds)} — grouped + track-stratified\n")
            f.write(f"path: {root}\ntrain: images/train\nval: images/val\n\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i in sorted(names):
                f.write(f"  {i}: {names[i]}\n")
        paths.append(yml)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "cv_folds"))
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=40,
                    help="higher than the 30 used before: the 1280 arm peaked at epoch 81, "
                         "so 30 nearly truncated it")
    ap.add_argument("--project", default=os.path.join(BASE_DIR, "runs_cv"))
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    import yaml
    with open(os.path.join(args.data, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]

    items = collect(args.data)
    print(f"pooled images: {len(items)}  ({Counter(i['track'] for i in items)})")

    groups = group_related(items)
    multi = [g for g in groups if len(g) > 1]
    print(f"correlation groups: {len(groups)}  ({len(multi)} with >1 image — kept intact)")
    for g in multi[:5]:
        print(f"    {[x['id'] for x in g]}")

    folds = assign_folds(groups, args.folds)
    print(f"\n{'fold':>5s} {'n':>6s} {'hard':>6s} {'regular':>8s}")
    for k, f in enumerate(folds):
        c = Counter(x["track"] for x in f)
        print(f"{k:5d} {len(f):6d} {c['hard']:6d} {c['regular']:8d}")

    ids = [x["id"] for f in folds for x in f]
    assert len(ids) == len(set(ids)) == len(items), "fold assignment lost or duplicated images"
    print("\nfold assignment is a clean partition ✓")

    yamls = build_fold_dirs(folds, args.out, names)
    print(f"fold datasets written under {args.out}")

    if args.build_only:
        print("\n[build-only] inspect the folds, then re-run without --build-only")
        return

    results = []
    for k, yml in enumerate(yamls):
        print(f"\n{'='*60}\n  FOLD {k}/{len(yamls)-1}\n{'='*60}", flush=True)
        name = f"cv_fold{k}"
        cmd = [sys.executable, os.path.join(BASE_DIR, "train_opg_seg.py"),
               "--data", yml, "--model", args.model, "--imgsz", str(args.imgsz),
               "--batch", str(args.batch), "--epochs", str(args.epochs),
               "--patience", str(args.patience), "--project", args.project, "--name", name]
        subprocess.run(cmd, check=False)
        cfg = os.path.join(args.project, name, "run_config.json")
        results.append({"fold": k, "done": os.path.exists(cfg)})
        with open(os.path.join(BASE_DIR, "cv_progress.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\nAll folds finished. Aggregate with summarise_cv.py")


if __name__ == "__main__":
    main()
