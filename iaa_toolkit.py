#!/usr/bin/env python3
"""
Inter-annotator agreement for FDI tooth segmentation: sample selection and scoring.

WHY THIS EXISTS
---------------
The annotation layer is the paper's contribution, so its reliability is the methods section
rather than a footnote. Reviewers at any medical venue will ask two things: who annotated, and
how much they agree. Nothing in the existing pipeline answers the second -- every task is
assigned to exactly one dentist, so no image has ever been independently annotated twice.
This has to be produced prospectively.

Agreement also gives the results a ceiling. "Model reaches 0.73 mask mAP50-95" means little on
its own; "two dentists agree at Dice 0.92 and the model reaches 0.90 of that" is an argument.
Without the human ceiling there is no way to say whether remaining error is model failure or
irreducible annotation ambiguity -- and given the lower incisors are ambiguous even to
clinicians, that distinction matters here.

WHAT TO MEASURE
---------------
Three different things get conflated as "agreement", and a reviewer will want them separated:

  presence   Did both annotators mark this tooth as present at all? Cohen's kappa over the
             32 classes, chance-corrected -- raw percent agreement is inflated because most
             teeth are present in most mouths.
  identity   Given both marked a tooth in the same place, did they assign the same FDI number?
             This is where systematic error lives (off-by-one along a quadrant with a missing
             tooth), and it is the failure mode with real clinical consequence.
  boundary   For teeth both marked, how closely do the outlines match? Dice and IoU for area,
             ASSD and HD95 for contour -- Dice saturates on compact shapes and will hide the
             boundary disagreement that matters most on small anterior teeth.

Usage:
  python iaa_toolkit.py sample --n 80 --out iaa_sample.csv
  python iaa_toolkit.py score --a labels_dentist_A --b labels_dentist_B --images images/
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
LABELS_MAP = os.path.join(BASE_DIR, "shared", "labels_map.json")
NUM_CLASSES = 32
SEED = 42


def load_names():
    with open(LABELS_MAP) as f:
        raw = json.load(f)
    return {int(k): v["name"] for k, v in raw.items()}


# ------------------------------------------------------------------ sampling
def cmd_sample(args):
    """Pick the re-annotation subset.

    Stratified by track so the hard cases are represented in proportion; a purely random draw
    from a pool that is 81% regular OPGs can leave too few hard cases to say anything about
    them, and those are exactly the images where annotators are most likely to diverge.
    """
    img_dir = os.path.join(args.data, "images")
    ids = []
    for split in ("train", "val"):
        d = os.path.join(img_dir, split)
        if os.path.isdir(d):
            ids += [os.path.splitext(f)[0] for f in os.listdir(d)]
    if not ids:
        sys.exit(f"no images under {img_dir}")

    strata = defaultdict(list)
    for i in ids:
        strata["hard" if i.startswith("HOPG") else "regular"].append(i)

    rng = random.Random(args.seed)
    total = len(ids)
    picked = []
    for name, pool in sorted(strata.items()):
        k = max(1, round(args.n * len(pool) / total))
        picked += [(i, name) for i in rng.sample(sorted(pool), min(k, len(pool)))]
    rng.shuffle(picked)
    picked = picked[:args.n]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "stratum", "annotator_a", "annotator_b", "notes"])
        for i, s in sorted(picked):
            w.writerow([i, s, "", "", ""])

    print(f"Selected {len(picked)} images ({args.n} requested) -> {args.out}")
    for name in sorted(strata):
        n = sum(1 for _, s in picked if s == name)
        print(f"   {name:8s}: {n:3d} of {len(strata[name])} available "
              f"({100*n/len(strata[name]):.1f}% of stratum)")
    print("\nNext: assign two dentists per row, have them annotate INDEPENDENTLY")
    print("(neither seeing the other's work, nor the existing annotation), then run `score`.")


# ------------------------------------------------------------------ scoring
def read_yolo(path, w, h):
    """YOLO polygon file -> {class_id: binary mask}."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            cls = int(p[0])
            pts = np.array([float(x) for x in p[1:]], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts.astype(np.int32)], 1)
            # A class can legally appear only once; if it repeats, union it rather than
            # silently keeping the last, so the disagreement shows up in the score.
            out[cls] = np.maximum(out[cls], m) if cls in out else m
    return out


def surface_distances(a, b):
    """Symmetric contour distances (ASSD, HD95) in pixels."""
    ca = cv2.Canny(a * 255, 100, 200) > 0
    cb = cv2.Canny(b * 255, 100, 200) > 0
    if not ca.any() or not cb.any():
        return float("nan"), float("nan")
    dt_a = cv2.distanceTransform((~ca).astype(np.uint8), cv2.DIST_L2, 3)
    dt_b = cv2.distanceTransform((~cb).astype(np.uint8), cv2.DIST_L2, 3)
    d_ab, d_ba = dt_b[ca], dt_a[cb]
    both = np.concatenate([d_ab, d_ba])
    return float(both.mean()), float(np.percentile(both, 95))


def cohens_kappa(both, only_a, only_b, neither):
    """Chance-corrected agreement on presence/absence."""
    n = both + only_a + only_b + neither
    if n == 0:
        return float("nan")
    po = (both + neither) / n
    pa1, pb1 = (both + only_a) / n, (both + only_b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def cmd_score(args):
    names = load_names()
    ids = sorted(set(os.path.splitext(f)[0] for f in os.listdir(args.a) if f.endswith(".txt"))
                 & set(os.path.splitext(f)[0] for f in os.listdir(args.b) if f.endswith(".txt")))
    if not ids:
        sys.exit("no image_ids present in BOTH annotation folders")
    print(f"Scoring {len(ids)} doubly-annotated images\n")

    pres = {c: dict(both=0, a=0, b=0, neither=0) for c in range(NUM_CLASSES)}
    dice = defaultdict(list)
    iou = defaultdict(list)
    assd = defaultdict(list)
    hd95 = defaultdict(list)
    per_image = []

    for iid in ids:
        img = None
        for split in ("train", "val", ""):
            p = os.path.join(args.images, split, f"{iid}.jpg")
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                break
        if img is None:
            print(f"  [WARN] no image for {iid}, skipping")
            continue
        h, w = img.shape[:2]
        A = read_yolo(os.path.join(args.a, f"{iid}.txt"), w, h)
        B = read_yolo(os.path.join(args.b, f"{iid}.txt"), w, h)

        img_dice = []
        for c in range(NUM_CLASSES):
            ia, ib = c in A, c in B
            if ia and ib:
                pres[c]["both"] += 1
                inter = np.logical_and(A[c], B[c]).sum()
                union = np.logical_or(A[c], B[c]).sum()
                sa, sb = A[c].sum(), B[c].sum()
                d = 2 * inter / (sa + sb) if (sa + sb) else 0.0
                dice[c].append(d)
                iou[c].append(inter / union if union else 0.0)
                m, hd = surface_distances(A[c], B[c])
                if not np.isnan(m):
                    assd[c].append(m)
                    hd95[c].append(hd)
                img_dice.append(d)
            elif ia:
                pres[c]["a"] += 1
            elif ib:
                pres[c]["b"] += 1
            else:
                pres[c]["neither"] += 1
        per_image.append((iid, len(A), len(B), float(np.mean(img_dice)) if img_dice else 0.0))

    kappas = [cohens_kappa(**{"both": v["both"], "only_a": v["a"],
                              "only_b": v["b"], "neither": v["neither"]})
              for v in pres.values()]
    kappas = [k for k in kappas if not np.isnan(k)]
    all_d = [x for v in dice.values() for x in v]
    all_i = [x for v in iou.values() for x in v]
    all_s = [x for v in assd.values() for x in v]
    all_h = [x for v in hd95.values() for x in v]
    disagree = sum(v["a"] + v["b"] for v in pres.values())

    print("=" * 66)
    print("  PRESENCE (did both mark the tooth at all?)")
    print(f"    mean Cohen's kappa over classes : {np.mean(kappas):.4f}")
    print(f"    teeth marked by exactly one     : {disagree}")
    print("  BOUNDARY (teeth both marked)")
    print(f"    mean Dice : {np.mean(all_d):.4f}   (sd {np.std(all_d):.4f})")
    print(f"    mean IoU  : {np.mean(all_i):.4f}")
    print(f"    mean ASSD : {np.mean(all_s):.2f} px    HD95: {np.mean(all_h):.2f} px")
    print("=" * 66)

    rows = sorted(((np.mean(dice[c]) if dice[c] else float("nan"), c)
                   for c in range(NUM_CLASSES)), key=lambda t: (np.isnan(t[0]), t[0]))
    print("\n  Lowest-agreement classes (this is the human ceiling per tooth):")
    for d, c in rows[:8]:
        n = len(dice[c])
        s = np.mean(assd[c]) if assd[c] else float("nan")
        print(f"    {names.get(c, c):34s} Dice {d:.4f}  ASSD {s:5.2f}px  (n={n})")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["class_id", "tooth", "n_both", "only_a", "only_b",
                         "kappa", "dice", "iou", "assd_px", "hd95_px"])
            for c in range(NUM_CLASSES):
                v = pres[c]
                w_.writerow([c, names.get(c, c), v["both"], v["a"], v["b"],
                             f"{cohens_kappa(v['both'], v['a'], v['b'], v['neither']):.4f}",
                             f"{np.mean(dice[c]):.4f}" if dice[c] else "",
                             f"{np.mean(iou[c]):.4f}" if iou[c] else "",
                             f"{np.mean(assd[c]):.2f}" if assd[c] else "",
                             f"{np.mean(hd95[c]):.2f}" if hd95[c] else ""])
        print(f"\n  Per-class table -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="choose the re-annotation subset")
    s.add_argument("--data", default=DEFAULT_DATA)
    s.add_argument("--n", type=int, default=80,
                   help="images to double-annotate (~5%% of the set is the usual convention)")
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", default=os.path.join(BASE_DIR, "iaa_sample.csv"))
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score", help="compute agreement between two annotation sets")
    c.add_argument("--a", required=True, help="folder of YOLO .txt from annotator A")
    c.add_argument("--b", required=True, help="folder of YOLO .txt from annotator B")
    c.add_argument("--images", default=os.path.join(DEFAULT_DATA, "images"))
    c.add_argument("--out", default=os.path.join(BASE_DIR, "iaa_per_class.csv"))
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
