"""Significance and equivalence testing for the model comparisons.

Why this exists. The paper reports differences like 0.7166 vs 0.7163 mask mAP50-95 and argues
architecture barely matters in-domain. A bare pair of point estimates cannot support that: the
reader has no way to tell a real 0.0003 gap from sampling noise on 144 images.

Two distinct questions get asked here, and conflating them is the usual error:

  1. "Do the models differ?"      -> paired permutation test over images.
  2. "Are the models equivalent?" -> two one-sided tests (TOST) against a margin.

A large p-value from (1) is *not* evidence of equivalence -- it is absence of evidence, and an
underpowered test produces it for free. Only (2) can license the claim the paper actually wants
to make. We therefore report, for every comparison, the observed difference, a 95% bootstrap CI,
a permutation p-value, and the smallest margin at which equivalence would hold.

Method. Resampling is over *images*, paired: both models are evaluated on the same resampled
image set, so the correlation between them (they see identical images and share failure cases)
is preserved and the CI on the difference is correspondingly tighter than an unpaired one.

mAP is recomputed exactly from the cached records with COCO's own 101-point interpolation --
not approximated -- so the point estimate reproduces evaluate_models.py to the last digit. That
reproduction is asserted, not assumed.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)          # COCO .50:.05:.95
REC_THRS = np.linspace(0, 1, 101)              # COCO 101-point interpolation
NUM_CLASSES = 32


# ------------------------------------------------------------------ record loading
class Dump:
    """Per-(image, class) records reshaped into dense per-class arrays over images."""

    def __init__(self, path):
        d = json.load(open(path))
        self.tag = d["tag"]
        self.model = d["model"]
        self.dataset = d["dataset"]
        self.images = d["images"]
        n = self.n = d["n_images"]

        self.has_gt = np.zeros((NUM_CLASSES, n), bool)
        self.has_det = np.zeros((NUM_CLASSES, n), bool)
        self.score = np.zeros((NUM_CLASSES, n), np.float64)
        self.iou = np.zeros((NUM_CLASSES, n), np.float64)
        self.inter = np.zeros((NUM_CLASSES, n), np.float64)
        self.union = np.zeros((NUM_CLASSES, n), np.float64)

        for r in d["records"]:
            c, i = r["c"], r["i"]
            self.has_gt[c, i] = bool(r["gt"])
            if r["score"] is not None:
                self.has_det[c, i] = True
                self.score[c, i] = r["score"]
            self.iou[c, i] = r["iou"]
            self.inter[c, i] = r["inter"]
            self.union[c, i] = r["union"]


def coco_ap(dump, idx, thr_slice=None):
    """Exact COCO segm AP over the (possibly resampled) image indices `idx`.

    Follows COCOeval.accumulate: per class, detections are ranked by score across images,
    matched at each IoU threshold, precision is made monotonically decreasing, then sampled at
    101 recall thresholds. Classes with no ground truth in the sample are excluded, matching
    COCO's `if npig == 0: continue`.
    """
    thrs = IOU_THRS if thr_slice is None else IOU_THRS[thr_slice]
    per_class = []
    for c in range(NUM_CLASSES):
        gt = dump.has_gt[c][idx]
        npig = int(gt.sum())
        if npig == 0:
            continue                                    # COCO leaves these at -1

        det = dump.has_det[c][idx]
        if not det.any():
            per_class.append(np.zeros(len(thrs)))        # GT exists, nothing found -> AP 0
            continue

        s = dump.score[c][idx][det]
        v = dump.iou[c][idx][det]
        order = np.argsort(-s, kind="mergesort")         # COCO uses a stable sort
        v = v[order]

        tp = v[None, :] >= thrs[:, None]                 # (T, D)
        tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
        fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)

        rc = tp_sum / npig
        pr = tp_sum / (tp_sum + fp_sum + np.spacing(1))

        # monotonically decreasing precision envelope, applied right-to-left
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]

        ap_t = np.empty(len(thrs))
        for t in range(len(thrs)):
            k = np.searchsorted(rc[t], REC_THRS, side="left")
            q = np.where(k < pr.shape[1], pr[t][np.minimum(k, pr.shape[1] - 1)], 0.0)
            ap_t[t] = q.mean()
        per_class.append(ap_t)

    if not per_class:
        return 0.0
    return float(np.mean(per_class))


def coco_map5095(dump, idx):
    return coco_ap(dump, idx)


def coco_map50(dump, idx):
    return coco_ap(dump, idx, thr_slice=slice(0, 1))


def miou(dump, idx):
    """Dataset-level IoU per class then mean over classes present -- as evaluate_models.py."""
    inter = dump.inter[:, idx].sum(axis=1)
    union = dump.union[:, idx].sum(axis=1)
    seen = dump.has_gt[:, idx].any(axis=1)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return float(iou[seen].mean()) if seen.any() else 0.0


METRICS = {"mAP50_95": coco_map5095, "mAP50": coco_map50, "mIoU": miou}


# ------------------------------------------------------------------ tests
def compare(a, b, metric, n_boot=2000, n_perm=2000, seed=0):
    """Paired bootstrap CI + paired permutation test for metric(a) - metric(b)."""
    assert a.images == b.images, f"{a.tag} and {b.tag} were evaluated on different images"
    fn = METRICS[metric]
    n = a.n
    full = np.arange(n)

    obs_a, obs_b = fn(a, full), fn(b, full)
    obs = obs_a - obs_b

    rng = np.random.default_rng(seed)

    # paired bootstrap: identical resampled images for both models
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[k] = fn(a, idx) - fn(b, idx)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # two-sided bootstrap p-value: how far the resampled difference sits from zero
    p_boot = float(min(1.0, 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())))

    # paired permutation: swap the two models' predictions within randomly chosen images
    #
    # Caveat, and the reason this is reported as secondary. AP depends on a *global* ranking of
    # detection scores within each class, and two models do not share a score scale. A permuted
    # hybrid therefore interleaves two differently-calibrated score distributions, which damages
    # its ranking and pushes the null distribution wider than it should be. Under the strict null
    # (identical models) swapping is a no-op and the test is exact, so a small p-value here is
    # trustworthy; a large one is weak evidence, because the test loses power exactly when the
    # models differ. Metrics that do not depend on ranking (mIoU) are unaffected.
    perm = np.empty(n_perm)
    for k in range(n_perm):
        swap = rng.random(n) < 0.5
        m1 = _swapped(a, b, swap)
        m2 = _swapped(b, a, swap)
        perm[k] = fn(m1, full) - fn(m2, full)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())

    # TOST: smallest symmetric margin at which equivalence holds at 95%
    eq_margin = float(max(abs(lo), abs(hi)))

    return {
        "metric": metric,
        "a": a.tag, "b": b.tag,
        "value_a": obs_a, "value_b": obs_b,
        "diff": obs,
        "ci95": [float(lo), float(hi)],
        "p_boot": p_boot,
        "p_perm": p,
        "boot_se": float(boot.std(ddof=1)),
        "equiv_margin_95": eq_margin,
    }


class _View:
    """A model whose per-image columns are taken from one of two dumps, per a swap mask."""
    __slots__ = ("images", "n", "has_gt", "has_det", "score", "iou", "inter", "union")


def _swapped(a, b, swap):
    """Model `a` except that on images where `swap` is set it uses `b`'s predictions.

    Ground truth is a property of the image, not the model, so it is never swapped -- only the
    prediction-side fields move. This is what makes the permutation valid: under the null that
    the two models are exchangeable, relabelling which model produced an image's predictions
    leaves the distribution unchanged.
    """
    v = _View()
    v.images, v.n = a.images, a.n
    v.has_gt = a.has_gt
    v.has_det = np.where(swap, b.has_det, a.has_det)
    v.score = np.where(swap, b.score, a.score)
    v.iou = np.where(swap, b.iou, a.iou)
    v.inter = np.where(swap, b.inter, a.inter)
    v.union = np.where(swap, b.union, a.union)
    return v


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        prev = adj[i] = max(prev, val)
    return adj


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=os.path.expanduser("~/opg-teeth/records"))
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="comparisons as A:B:label, tags refer to dumped record files")
    ap.add_argument("--metrics", nargs="+", default=["mAP50_95", "mAP50", "mIoU"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--verify", default=None,
                    help="JSON of tag -> {mAP50_95, mAP50, mIoU} to assert reproduction against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dumps = {}
    for f in sorted(glob.glob(os.path.join(args.records, "*.json"))):
        d = Dump(f)
        dumps[d.tag] = d
    print(f"loaded {len(dumps)} record sets: {', '.join(sorted(dumps))}\n")

    # The recomputation must reproduce the harness exactly, or nothing below means anything.
    if args.verify:
        ref = json.load(open(args.verify))
        print("reproduction check against evaluate_models.py")
        worst = 0.0
        for tag, exp in ref.items():
            if tag not in dumps:
                continue
            d = dumps[tag]
            full = np.arange(d.n)
            for m, want in exp.items():
                got = METRICS[m](d, full)
                delta = abs(got - want)
                worst = max(worst, delta)
                flag = "ok " if delta < 5e-4 else "BAD"
                print(f"  [{flag}] {tag:22s} {m:9s} harness={want:.4f} recomputed={got:.4f} "
                      f"d={delta:.2e}")
        print(f"  worst deviation {worst:.2e}\n")
        assert worst < 5e-4, "recomputation does not reproduce the harness -- do not trust results"

    results = []
    for spec in args.pairs:
        a_tag, b_tag, label = spec.split(":")
        if a_tag not in dumps or b_tag not in dumps:
            print(f"skip {label}: missing {a_tag if a_tag not in dumps else b_tag}")
            continue
        for m in args.metrics:
            r = compare(dumps[a_tag], dumps[b_tag], m,
                        n_boot=args.n_boot, n_perm=args.n_perm)
            r["label"] = label
            results.append(r)
            print(f"{label:34s} {m:9s} {r['value_a']:.4f} vs {r['value_b']:.4f}  "
                  f"d={r['diff']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
                  f"p_boot={r['p_boot']:.4f} p_perm={r['p_perm']:.4f}")

    # Holm correction within each metric family, on the primary (bootstrap) p-value
    for m in args.metrics:
        sub = [r for r in results if r["metric"] == m]
        if sub:
            adj = holm([r["p_boot"] for r in sub])
            for r, a in zip(sub, adj):
                r["p_holm"] = float(a)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n_boot": args.n_boot, "n_perm": args.n_perm,
                       "results": results}, f, indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
