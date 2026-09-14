#!/usr/bin/env python3
"""
Validate the arch-consistency rules against ground truth before using them to correct models.

THE TEST THAT MATTERS
---------------------
These rules were written to catch annotator mistakes at save time, where a false positive costs
a dentist ten seconds. Reusing them as model post-processing is a different contract: a rule
that fires on correct data will "correct" predictions that were already right, and make the
model worse.

So the question is not "do the rules encode real anatomy" -- they do -- but "how often do they
fire on annotations we believe are correct?" Every firing on ground truth is a false positive,
because the reviewed-and-approved annotations are the best available definition of correct.

  firing rate ~0%   -> safe as a hard constraint; violations in predictions are real errors
  firing rate a few % -> usable as a soft signal or a tie-breaker, not an automatic rewrite
  firing rate high  -> the rule encodes an idealisation that real anatomy violates; do not use

This mirrors the js implementation in backend/src/archConsistency.js; if that changes, this
must be re-run.

Usage:
  python validate_arch_rules.py
"""

import argparse
import os
from collections import Counter, defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")

CLASS_ID_TO_FDI = [
    18, 17, 16, 15, 14, 13, 12, 11,
    21, 22, 23, 24, 25, 26, 27, 28,
    38, 37, 36, 35, 34, 33, 32, 31,
    48, 47, 46, 45, 44, 43, 42, 41,
]
UPPER_ORDER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_ORDER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

SIDE_FLIP_MARGIN = 0.10
ARCH_Y_MARGIN = 0.08
MIN_TEETH = 6


def load_teeth(path):
    """YOLO polygons -> [{fdi, q, x, y}] with normalised centroids."""
    teeth = []
    for line in open(path):
        p = line.split()
        if len(p) < 7:
            continue
        cid = int(p[0])
        if not 0 <= cid < len(CLASS_ID_TO_FDI):
            continue
        pts = np.array([float(v) for v in p[1:]], dtype=np.float64).reshape(-1, 2)
        fdi = CLASS_ID_TO_FDI[cid]
        teeth.append({"fdi": fdi, "q": fdi // 10,
                      "x": float(pts[:, 0].mean()), "y": float(pts[:, 1].mean())})
    return teeth


def check(teeth):
    """Returns (errors, warnings) as lists of rule codes, mirroring the js logic."""
    errors, warnings = [], []
    if len(teeth) < MIN_TEETH:
        return errors, warnings

    seen = set()
    for t in teeth:
        if t["fdi"] in seen:
            errors.append("DUPLICATE_TOOTH")
        seen.add(t["fdi"])

    xs = [t["x"] for t in teeth]
    mid_x = float(np.median(xs))

    def mean_x(q):
        v = [t["x"] for t in teeth if t["q"] == q]
        return float(np.mean(v)) if v else None

    def side_of(m):
        if m is None:
            return 0
        if m < mid_x - SIDE_FLIP_MARGIN:
            return -1
        if m > mid_x + SIDE_FLIP_MARGIN:
            return 1
        return 0

    s1, s2, s3, s4 = side_of(mean_x(1)), side_of(mean_x(2)), side_of(mean_x(3)), side_of(mean_x(4))
    if s1 and s4 and s1 != s4:
        errors.append("SIDE_FLIP_RIGHT")
    if s2 and s3 and s2 != s3:
        errors.append("SIDE_FLIP_LEFT")

    upper = [t for t in teeth if t["q"] in (1, 2)]
    lower = [t for t in teeth if t["q"] in (3, 4)]
    if upper and lower:
        y_med = float(np.median([t["y"] for t in teeth]))
        for t in upper:
            if t["y"] > y_med + ARCH_Y_MARGIN:
                warnings.append("ARCH_MISMATCH")
        for t in lower:
            if t["y"] < y_med - ARCH_Y_MARGIN:
                warnings.append("ARCH_MISMATCH")

    for arch, order in ((upper, UPPER_ORDER), (lower, LOWER_ORDER)):
        s = sorted(arch, key=lambda t: t["x"])
        for a, b in zip(s, s[1:]):
            if order.index(a["fdi"]) > order.index(b["fdi"]):
                warnings.append("ORDER_INVERSION")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    files = []
    for split in ("train", "val"):
        d = os.path.join(args.data, "labels", split)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".txt")]
    if not files:
        raise SystemExit(f"no labels under {args.data}")

    n = 0
    skipped = 0
    img_with = Counter()
    total_fires = Counter()
    per_image_counts = defaultdict(list)
    examples = defaultdict(list)

    for f in files:
        teeth = load_teeth(f)
        if len(teeth) < MIN_TEETH:
            skipped += 1
            continue
        n += 1
        errs, warns = check(teeth)
        for code in set(errs) | set(warns):
            img_with[code] += 1
            if len(examples[code]) < 4:
                examples[code].append(os.path.basename(f).replace(".txt", ""))
        for code in errs + warns:
            total_fires[code] += 1
        per_image_counts["ORDER_INVERSION"].append(warns.count("ORDER_INVERSION"))
        per_image_counts["ARCH_MISMATCH"].append(warns.count("ARCH_MISMATCH"))

    print(f"Ground-truth annotations checked: {n}  (skipped {skipped} with <{MIN_TEETH} teeth)\n")
    print(f"{'rule':20s} {'images firing':>14s} {'rate':>8s} {'total fires':>12s}   verdict")
    print("-" * 78)
    for code in ("DUPLICATE_TOOTH", "SIDE_FLIP_RIGHT", "SIDE_FLIP_LEFT",
                 "ARCH_MISMATCH", "ORDER_INVERSION"):
        c = img_with[code]
        rate = 100.0 * c / n if n else 0.0
        if rate == 0:
            verdict = "SAFE as hard constraint"
        elif rate < 2:
            verdict = "usable as soft signal"
        elif rate < 15:
            verdict = "too noisy to auto-correct"
        else:
            verdict = "UNUSABLE - encodes an idealisation"
        print(f"{code:20s} {c:14d} {rate:7.2f}% {total_fires[code]:12d}   {verdict}")
        if c and examples[code]:
            print(f"{'':20s} e.g. {', '.join(examples[code])}")

    for code in ("ORDER_INVERSION", "ARCH_MISMATCH"):
        v = [x for x in per_image_counts[code] if x > 0]
        if v:
            print(f"\n{code}: when it fires, {np.mean(v):.1f} times per image on average "
                  f"(max {max(v)})")


if __name__ == "__main__":
    main()
