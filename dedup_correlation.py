#!/usr/bin/env python3
"""
Exhaustive duplicate detection by normalised pixel correlation.

WHY NOT PERCEPTUAL HASHING
--------------------------
Perceptual hashing is the usual tool and it fails on this modality. Measured on this dataset,
the median nearest-neighbour phash distance between our images and Tufts is 12 -- identical to
the median distance *within* our own set of definitely-distinct radiographs. Every panoramic
image shares the same global arch structure, so a 64-bit hash tuned for natural images has
almost no discriminative power here. The phash pass caught 18 duplicates and still missed
OPG_22/OPG_23, which are pixel-identical.

Correlation on brightness- and contrast-normalised thumbnails separates cleanly: verified
duplicates score 1.0000, unrelated panoramic pairs top out around 0.86. Z-normalising each
thumbnail removes exposure and contrast differences, so a re-encoded, re-scaled or
gamma-adjusted copy still scores ~1.0 while a genuinely different patient does not.

Comparison is exhaustive -- every pair -- because a cheap shortlist built from a weak signal
is exactly how the earlier pass missed things. At 64x128 the whole set is a single matrix
multiply, so there is no reason to approximate.

Usage:
  python dedup_correlation.py --check                  # report only
  python dedup_correlation.py --apply                  # remove duplicates + fix data.yaml
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
EMB_H, EMB_W = 64, 128
# Verified duplicates score 1.0000; the highest unrelated pair observed across 1.4M
# comparisons was 0.857. 0.99 sits far from both, so the threshold is not delicate.
DUP_THRESHOLD = 0.99


def embed(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    v = cv2.resize(im, (EMB_W, EMB_H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()                       # remove brightness
    n = np.linalg.norm(v)
    return v / n if n > 0 else v        # remove contrast -> dot product == Pearson r


def collect(data_dir):
    items = []
    for split in ("train", "val"):
        d = os.path.join(data_dir, "images", split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.splitext(f)[0], split, os.path.join(d, f)))
    return items


def find_duplicates(items, threshold):
    with ThreadPoolExecutor(8) as ex:
        embs = list(ex.map(embed, [p for _, _, p in items]))
    keep = [i for i, e in enumerate(embs) if e is not None]
    A = np.stack([embs[i] for i in keep])
    C = A @ A.T
    np.fill_diagonal(C, -1.0)

    # Union-find so a chain of near-identical images collapses to one group.
    parent = list(range(len(keep)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ii, jj = np.where(np.triu(C, 1) >= threshold)
    for a, b in zip(ii, jj):
        union(int(a), int(b))

    groups = {}
    for idx in range(len(keep)):
        groups.setdefault(find(idx), []).append(items[keep[idx]])
    return [g for g in groups.values() if len(g) > 1], C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--threshold", type=float, default=DUP_THRESHOLD)
    ap.add_argument("--apply", action="store_true", help="actually delete; default is report only")
    args = ap.parse_args()

    items = collect(args.data)
    if not items:
        sys.exit(f"no images under {args.data}")
    print(f"Comparing {len(items)} images exhaustively "
          f"({len(items)*(len(items)-1)//2:,} pairs) at threshold {args.threshold}")

    dupes, C = find_duplicates(items, args.threshold)
    best_unrelated = float(C[C < args.threshold].max()) if (C < args.threshold).any() else 0.0
    print(f"  highest correlation BELOW threshold: {best_unrelated:.4f}  "
          f"(margin to threshold: {args.threshold - best_unrelated:.4f})")

    if not dupes:
        print("\nNo duplicates found. Dataset is clean.")
        return

    total_extra = sum(len(g) - 1 for g in dupes)
    print(f"\n{len(dupes)} duplicate group(s), {total_extra} redundant image(s):")
    to_remove = []
    for g in dupes:
        splits = {s for _, s, _ in g}
        # Keep the val copy when a group straddles the split -- dropping the train copy is
        # what removes the leak. Otherwise keep the lowest id, for determinism.
        keeper = (next(x for x in g if x[1] == "val") if len(splits) > 1
                  else sorted(g, key=lambda x: x[0])[0])
        straddle = " ** STRADDLES SPLIT **" if len(splits) > 1 else ""
        desc = ", ".join(f"{i}({s})" for i, s, _ in g)
        print(f"   {desc}  -> keep {keeper[0]}{straddle}")
        to_remove += [x for x in g if x[0] != keeper[0]]

    if not args.apply:
        print(f"\n[report only] Re-run with --apply to remove {len(to_remove)} image(s).")
        return

    for iid, split, path in to_remove:
        lbl = os.path.join(args.data, "labels", split, f"{iid}.txt")
        for p in (path, lbl):
            if os.path.exists(p):
                os.remove(p)
        print(f"   removed {iid} ({split})")

    tr = len(os.listdir(os.path.join(args.data, "images", "train")))
    va = len(os.listdir(os.path.join(args.data, "images", "val")))
    yml = os.path.join(args.data, "data.yaml")
    if os.path.exists(yml):
        out = []
        for line in open(yml):
            if line.startswith("# Train images:"):
                line = f"# Train images: {tr}\n"
            elif line.startswith("# Val images:"):
                line = f"# Val images: {va}\n"
            elif line.startswith("# Total:"):
                line = f"# Total: {tr + va}\n"
            out.append(line)
        open(yml, "w").write("".join(out))
    print(f"\nNow: train {tr} / val {va} / total {tr + va}   (data.yaml updated)")


if __name__ == "__main__":
    main()
