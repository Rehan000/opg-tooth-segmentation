#!/usr/bin/env python3
"""
Fingerprint dataset images so overlap with public datasets can be proven or ruled out.

The OPG imagery came from Kaggle, which is a redistribution layer rather than an origin.
That creates two problems this script exists to answer:

  1. LICENSING — we need to know which upstream dataset(s) the images actually came from.
  2. LEAKAGE  — if the Kaggle pool absorbed a public dataset, then benchmarking against that
     dataset measures memorisation, not generalisation. A "we validated externally on X"
     claim is worthless, and worse than worthless in review, if X is already in training.

Kaggle re-uploads routinely renumber files, so filenames prove nothing (our own hard track
renumbered too). Only pixels are evidence. Three fingerprints are emitted per image:

  md5    exact bytes. Catches verbatim redistribution.
  dhash  64-bit gradient hash. Survives re-compression and mild resizing.
  phash  64-bit DCT hash. Survives re-compression, resizing, and small brightness/gamma
         shifts — which is what a re-upload pipeline typically does to an image.

md5 alone is not enough: a re-encoded JPEG of the same radiograph has a different md5 but is
the same patient, and would still leak across a train/test boundary. Compare perceptual
hashes by Hamming distance, not equality — see compare_fingerprints() usage in --help.

Usage:
  conda activate dentalenv
  python fingerprint_dataset.py                                  # fingerprint merged_opg_dataset
  python fingerprint_dataset.py --dir path/to/other/images       # any image folder
  python fingerprint_dataset.py --compare a.csv b.csv            # report overlap between two runs
"""

import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, "merged_opg_dataset", "images")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_WORKERS = 8

# Hamming distance at or below this counts as "same image, re-encoded".
# 64-bit hashes: <=5 is the conventional threshold for near-duplicate.
NEAR_DUP_THRESHOLD = 5


def iter_images(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in EXTS:
                yield os.path.join(dirpath, fn)


def dhash(gray, size=8):
    """Difference hash: compare each pixel to its right neighbour."""
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return bits_to_hex(diff.flatten())


def phash(gray, size=8, factor=4):
    """Perceptual hash: low-frequency DCT coefficients vs their median."""
    img = cv2.resize(gray, (size * factor, size * factor), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))
    low = dct[:size, :size]
    # Skip the DC term when taking the median; it carries overall brightness, not structure.
    med = np.median(low.flatten()[1:])
    return bits_to_hex((low > med).flatten())


def bits_to_hex(bits):
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def fingerprint(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, f"unreadable: {path}"
        h, w = img.shape[:2]
        return {
            "image_id": os.path.splitext(os.path.basename(path))[0],
            "path": os.path.relpath(path, BASE_DIR),
            "width": w,
            "height": h,
            "bytes": len(raw),
            "md5": md5,
            "dhash": dhash(img),
            "phash": phash(img),
        }, None
    except Exception as e:
        return None, f"{path}: {e}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compare(path_a, path_b):
    """Report exact and near-duplicate overlap between two fingerprint files."""
    a, b = load_csv(path_a), load_csv(path_b)
    print(f"A: {len(a):5d} images  ({path_a})")
    print(f"B: {len(b):5d} images  ({path_b})\n")

    by_md5 = {}
    for r in b:
        by_md5.setdefault(r["md5"], []).append(r)
    exact = [(r, by_md5[r["md5"]][0]) for r in a if r["md5"] in by_md5]
    print(f"Exact (md5) matches: {len(exact)}")

    # Near-duplicates: only worth checking for rows that were not already exact matches.
    exact_ids = {r["image_id"] for r, _ in exact}
    near = []
    for ra in a:
        if ra["image_id"] in exact_ids:
            continue
        for rb in b:
            if hamming(ra["phash"], rb["phash"]) <= NEAR_DUP_THRESHOLD and \
               hamming(ra["dhash"], rb["dhash"]) <= NEAR_DUP_THRESHOLD:
                near.append((ra, rb))
                break
    print(f"Near-duplicate (phash+dhash <= {NEAR_DUP_THRESHOLD}) matches: {len(near)}")

    total = len(exact) + len(near)
    print(f"\nTOTAL OVERLAP: {total} / {len(a)} of A ({100*total/max(1,len(a)):.1f}%)")
    if total:
        print("\nThis dataset is NOT safe to use as external validation.")
        print("Examples:")
        for ra, rb in (exact + near)[:10]:
            kind = "exact" if ra["image_id"] in exact_ids else "near "
            print(f"  [{kind}] {ra['image_id']:16s} <-> {rb['image_id']}")
    else:
        print("\nNo overlap detected — safe for external validation.")
        print("State this in the paper, with the threshold and method used.")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="image folder to fingerprint (recursive)")
    ap.add_argument("--out", default=None, help="output CSV (default: <dir-name>_fingerprints.csv)")
    ap.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"),
                    help="compare two fingerprint files instead of generating one")
    args = ap.parse_args()

    if args.compare:
        overlap = compare(*args.compare)
        sys.exit(1 if overlap else 0)

    if not os.path.isdir(args.dir):
        sys.exit(f"No such directory: {args.dir}")

    paths = list(iter_images(args.dir))
    if not paths:
        sys.exit(f"No images found under {args.dir}")
    print(f"Fingerprinting {len(paths)} images from {args.dir} ...")

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fingerprint, p) for p in paths]
        for i, fut in enumerate(as_completed(futures), 1):
            row, err = fut.result()
            (errors if err else rows).append(err or row)
            if i % 250 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}")

    out = args.out or os.path.join(
        BASE_DIR, f"{os.path.basename(os.path.normpath(args.dir))}_fingerprints.csv")
    rows.sort(key=lambda r: r["image_id"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {out}  ({len(rows)} rows)")

    # Internal duplicates matter as much as external ones: a re-encoded copy of the same
    # radiograph sitting in both train and val inflates every metric we report.
    for label, key in (("exact (md5)", "md5"), ("near-dup (phash)", "phash")):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["image_id"])
        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        n = sum(len(v) - 1 for v in dupes.values())
        print(f"  internal {label:18s}: {len(dupes)} group(s), {n} redundant image(s)")
        for ids in list(dupes.values())[:5]:
            print(f"      {', '.join(ids[:6])}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  [WARN] {e}")


if __name__ == "__main__":
    main()
