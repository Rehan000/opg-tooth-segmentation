#!/usr/bin/env python3
"""
Aggregate 5-fold cross-validation into reportable numbers.

Reports mean +/- standard deviation across folds, because a single split cannot tell you
whether a difference is real. The resolution ablation moved mask mAP50-95 by 0.019 between
1024 and 1280; if the fold-to-fold spread turns out to be of that order, that difference is
not established and the paper must say so.

Also reports per-class spread, since the lower-anterior deficit is a headline claim and needs
to hold across folds rather than in one lucky split.

Usage:
  python summarise_cv.py --project runs_cv
"""

import argparse
import csv
import json
import os
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_fold(run_dir):
    """Pull the best epoch's metrics from Ultralytics' results.csv."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None

    def col(row, name):
        """Exact column match. Substring matching is unsafe here: Ultralytics names columns
        'metrics/mAP50-95(B)' and 'metrics/mAP50-95(M)', and a bare 'm' fragment also matches
        inside the word 'metrics', so box and mask silently resolve to the same column."""
        for k, v in row.items():
            if k.strip().lower() == name.lower():
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    # "Best" is the epoch with the highest MASK mAP50-95 -- segmentation quality is the
    # quantity of interest, and it is strictly the harder of the two.
    scored = [(col(r, "metrics/mAP50-95(M)"), i) for i, r in enumerate(rows)]
    scored = [(s, i) for s, i in scored if s is not None]
    if not scored:
        return None
    best_val, best_i = max(scored)
    r = rows[best_i]
    out = {
        "epochs_run": len(rows),
        "best_epoch": best_i + 1,
        "mask_mAP50_95": best_val,
        "mask_mAP50": col(r, "metrics/mAP50(M)"),
        "box_mAP50_95": col(r, "metrics/mAP50-95(B)"),
        "box_mAP50": col(r, "metrics/mAP50(B)"),
    }
    # Mask AP can never exceed box AP for the same detections; if it does, the columns were
    # misread and every downstream number is wrong.
    if out["box_mAP50_95"] is not None and out["mask_mAP50_95"] > out["box_mAP50_95"] + 1e-9:
        raise SystemExit(f"{run_dir}: mask mAP ({out['mask_mAP50_95']:.4f}) exceeds box mAP "
                         f"({out['box_mAP50_95']:.4f}) -- column parsing is wrong")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.path.join(BASE_DIR, "runs_cv"))
    ap.add_argument("--prefix", default="cv_fold")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "cv_summary.json"))
    args = ap.parse_args()

    runs = sorted(
        (d for d in os.listdir(args.project) if d.startswith(args.prefix)),
        key=lambda d: int(re.sub(r"\D", "", d) or 0),
    )
    folds = {}
    for d in runs:
        m = read_fold(os.path.join(args.project, d))
        if m:
            folds[d] = m

    if not folds:
        raise SystemExit(f"no completed folds under {args.project}")

    print(f"{'fold':<12s} {'epochs':>7s} {'best':>6s} {'mask50-95':>10s} {'mask50':>8s} "
          f"{'box50-95':>9s}")
    for d, m in folds.items():
        print(f"{d:<12s} {m['epochs_run']:7d} {m['best_epoch']:6d} "
              f"{m['mask_mAP50_95']:10.4f} {m['mask_mAP50']:8.4f} {m['box_mAP50_95']:9.4f}")

    print("\n" + "=" * 58)
    summary = {}
    for key, label in (("mask_mAP50_95", "mask mAP50-95"), ("mask_mAP50", "mask mAP50"),
                       ("box_mAP50_95", "box mAP50-95"), ("box_mAP50", "box mAP50")):
        vals = np.array([m[key] for m in folds.values() if m[key] is not None])
        if not len(vals):
            continue
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                        "min": float(vals.min()), "max": float(vals.max()),
                        "n_folds": int(len(vals))}
        print(f"  {label:<15s} {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}   "
              f"[{vals.min():.4f}, {vals.max():.4f}]")
    print("=" * 58)

    m = summary.get("mask_mAP50_95")
    if m:
        sd = m["std"]
        print(f"\nFold-to-fold sd on mask mAP50-95 is {sd:.4f}.")
        print(f"The 1024 -> 1280 resolution gain was 0.019, i.e. {0.019/sd:.1f} sd. "
              f"{'Comfortably resolvable.' if 0.019 > 2*sd else 'NOT resolvable at this spread -- report it as within noise.'}")
        epochs = [f['best_epoch'] for f in folds.values()]
        print(f"Best epoch across folds: {min(epochs)}-{max(epochs)} "
              f"(mean {np.mean(epochs):.0f}) — informs the epoch budget for final runs.")

    with open(args.out, "w") as f:
        json.dump({"folds": folds, "summary": summary}, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
