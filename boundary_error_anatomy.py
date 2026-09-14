"""Where along a tooth is segmentation accuracy actually lost?

The paper's claim is that detection is solved and delineation is not: mAP50 sits at 0.98 while
mAP50-95 sits at 0.72. That establishes *that* boundaries are imprecise but says nothing about
*where* on the tooth the error lives, which is the question a clinician would ask first -- an
error at the occlusal surface and an error at the root apex have completely different
consequences for the downstream measurements the segmentation feeds.

Method. For each correctly detected tooth, PCA on the ground-truth mask gives the long axis.
Pixels are projected onto it and split into three equal bands from crown to apex. Crown/apex
orientation is resolved anatomically, not by image convention: in a panoramic projection the
upper arch has apices superior and the lower arch has apices inferior, so the sign flips by
quadrant. IoU is then computed within each band.

A band's IoU is a local quantity -- prediction and ground truth restricted to that band -- so
the three numbers decompose the whole-tooth figure by anatomy rather than partitioning it
arithmetically.

Also reports the interproximal case: mesial/distal (across-axis) versus axial displacement,
which distinguishes "the mask is the right shape in the wrong place" from "the mask stops short".
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_models as em

FDI_ORDER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28,
             38, 37, 36, 35, 34, 33, 32, 31, 48, 47, 46, 45, 44, 43, 42, 41]
BAND_NAMES = ["crown", "middle", "apical"]


def long_axis_bands(gt, is_upper, n_bands=3):
    """Split a tooth mask into bands along its long axis, ordered crown -> apex."""
    ys, xs = np.nonzero(gt)
    if len(ys) < 12:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    # principal direction of the tooth
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    # orient the axis so it points crown -> apex
    # image y grows downward; upper-arch apices are superior (smaller y), lower are inferior
    if (axis[1] > 0) == bool(is_upper):
        axis = -axis
    t = pts @ axis
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-6:
        return None
    edges = np.linspace(lo, hi, n_bands + 1)
    return axis, edges, (ys, xs)


def band_masks(shape, axis, edges, centre):
    """Boolean band membership for every pixel in the image, along the tooth axis."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    t = ((xx - centre[0]) * axis[0] + (yy - centre[1]) * axis[1])
    out = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        m = (t >= lo) & (t <= hi) if i == len(edges) - 2 else (t >= lo) & (t < hi)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/ablate_1280/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--data", default=em.DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="boundary_anatomy.json")
    args = ap.parse_args()

    items = em.gt_from_yolo(args.data, "val")
    if args.limit:
        items = items[:args.limit]
    print(f"boundary anatomy on {len(items)} images", flush=True)

    preds = em.predict_yolo(args.weights, items, args.imgsz, postproc="greedy")

    # accumulate intersection/union per band, globally and per class
    g_int = np.zeros(3)
    g_uni = np.zeros(3)
    per_class = {c: [np.zeros(3), np.zeros(3)] for c in range(32)}
    whole_int = 0.0
    whole_uni = 0.0
    lateral, axial = [], []
    n_teeth = 0

    for iid, path, polys, norm in items:
        h, w = em.image_size(path)
        gt_all = em.rasterise(polys, w, h, norm)
        pr = preds.get(iid, {})
        for c, g in gt_all.items():
            entry = pr.get(c)
            if entry is None:
                continue
            p = em._decode(entry[0])
            fdi = FDI_ORDER[c]
            is_upper = fdi // 10 in (1, 2)

            r = long_axis_bands(g, is_upper)
            if r is None:
                continue
            axis, edges, (ys, xs) = r
            centre = (xs.mean(), ys.mean())

            whole_int += np.logical_and(g, p).sum()
            whole_uni += np.logical_or(g, p).sum()
            n_teeth += 1

            # restrict to a crop around the union, so the meshgrid stays cheap
            uy, ux = np.nonzero(np.logical_or(g, p))
            y0, y1 = uy.min(), uy.max() + 1
            x0, x1 = ux.min(), ux.max() + 1
            gc, pc = g[y0:y1, x0:x1], p[y0:y1, x0:x1]
            cc = (centre[0] - x0, centre[1] - y0)
            for bi, bm in enumerate(band_masks(gc.shape, axis, edges, cc)):
                gi = np.logical_and(gc, bm)
                pi = np.logical_and(pc, bm)
                i_ = np.logical_and(gi, pi).sum()
                u_ = np.logical_or(gi, pi).sum()
                g_int[bi] += i_
                g_uni[bi] += u_
                per_class[c][0][bi] += i_
                per_class[c][1][bi] += u_

            # displacement of the mask centroid, decomposed along/across the tooth axis
            py, px = np.nonzero(p)
            if len(py):
                d = np.array([px.mean() - centre[0], py.mean() - centre[1]])
                axial.append(float(abs(d @ axis)))
                perp = np.array([-axis[1], axis[0]])
                lateral.append(float(abs(d @ perp)))
        del gt_all

    band_iou = np.divide(g_int, g_uni, out=np.zeros(3), where=g_uni > 0)
    whole = whole_int / whole_uni if whole_uni else 0.0

    print("\n" + "=" * 58)
    print(f"  teeth analysed: {n_teeth}")
    print(f"  whole-tooth IoU : {whole:.4f}")
    for name, v, u in zip(BAND_NAMES, band_iou, g_uni):
        print(f"    {name:8s} IoU {v:.4f}   ({u/g_uni.sum()*100:4.1f}% of union area)")
    print(f"  centroid displacement (px): axial {np.mean(axial):.2f} "
          f"| lateral {np.mean(lateral):.2f}")
    print("=" * 58)

    pc_out = {}
    for c, (i_, u_) in per_class.items():
        if u_.sum() == 0:
            continue
        pc_out[FDI_ORDER[c]] = [float(x) for x in
                                np.divide(i_, u_, out=np.zeros(3), where=u_ > 0)]

    with open(args.out, "w") as f:
        json.dump({"n_teeth": n_teeth, "whole_iou": float(whole),
                   "band_names": BAND_NAMES,
                   "band_iou": [float(x) for x in band_iou],
                   "band_union_share": [float(x) for x in g_uni / g_uni.sum()],
                   "axial_disp_px": float(np.mean(axial)),
                   "lateral_disp_px": float(np.mean(lateral)),
                   "per_class_band_iou": pc_out}, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
