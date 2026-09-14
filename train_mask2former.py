#!/usr/bin/env python3
"""
Mask2Former baseline for 32-class FDI tooth instance segmentation.

WHY THIS BASELINE
-----------------
YOLO-seg is a one-stage detector with a linear mask head; Mask2Former is a query-based
transformer that predicts masks directly via attention. They are the two dominant families in
instance segmentation, and a reviewer comparing a dental segmentation paper against the state
of the art will expect the transformer to be present. Without it, "YOLO wins" only means
"YOLO beat a foundation-model encoder", which is a narrower claim than the paper makes.

FAIRNESS NOTES (state these in the paper)
-----------------------------------------
- Mask2Former is substantially larger than YOLO11m (22.4M): swin-tiny is ~47M, swin-small
  ~69M. The comparison therefore favours Mask2Former on capacity, which makes a YOLO win the
  more conservative result.
- Both see the same images at the same aspect ratio, the same fold split, and the same
  augmentation philosophy (photometric only; horizontal flip is disabled because mirroring
  swaps left/right FDI labels).
- Mask2Former has no equivalent of YOLO's one-per-class selection. Since FDI numbering is
  unique per mouth, the same constraint is applied at inference for both, so the comparison
  isolates the model rather than the post-processing.

Usage:
  python train_mask2former.py --smoke                 # 2 epochs on a few images
  python train_mask2former.py --bench                 # time one step at several sizes
  python train_mask2former.py --epochs 60
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
NUM_CLASSES = 32
SEED = 42


class OPGInstanceDataset(Dataset):
    """YOLO polygons -> one binary mask + class label per tooth, which is what
    Mask2Former's loss consumes (it matches predicted queries to ground-truth instances)."""

    def __init__(self, root, split, size, augment=False):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.w, self.h = size
        self.augment = augment
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png")))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        path = os.path.join(self.img_dir, f"{iid}.jpg")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            # Photometric only. A horizontal flip would turn tooth 18 into tooth 28.
            if np.random.rand() < 0.8:
                a = 1.0 + np.random.uniform(-0.25, 0.25)
                b = np.random.uniform(-25, 25)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                g = np.random.uniform(0.7, 1.4)
                img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)

        masks, labels = [], []
        lp = os.path.join(self.lbl_dir, f"{iid}.txt")
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 7:
                    continue
                cls = int(p[0])
                pts = np.array([float(v) for v in p[1:]], np.float32).reshape(-1, 2)
                pts[:, 0] *= self.w
                pts[:, 1] *= self.h
                m = np.zeros((self.h, self.w), np.uint8)
                cv2.fillPoly(m, [pts.astype(np.int32)], 1)
                if m.sum() > 0:
                    masks.append(m)
                    labels.append(cls)

        x = np.repeat(img[None], 3, 0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        x = (x - mean) / std
        return {
            "pixel_values": torch.from_numpy(x),
            "mask_labels": torch.from_numpy(np.stack(masks)).float() if masks
            else torch.zeros((0, self.h, self.w)),
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "image_id": iid,
        }


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }


def build_model(checkpoint):
    from transformers import Mask2FormerForUniversalSegmentation
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,   # COCO's 80 classes -> our 32
    )


@torch.no_grad()
def evaluate(model, loader, device, size):
    """Dataset-level IoU per class. One instance per class is enforced, mirroring the
    production constraint applied to YOLO, so the comparison isolates the model."""
    model.eval()
    w, h = size
    inter = np.zeros(NUM_CLASSES); union = np.zeros(NUM_CLASSES)
    seen = np.zeros(NUM_CLASSES, bool)
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=pv)
        # queries -> per-class masks
        cls_logits = out.class_queries_logits.float()          # (B, Q, C+1)
        mask_logits = out.masks_queries_logits.float()         # (B, Q, h', w')
        mask_logits = torch.nn.functional.interpolate(
            mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        probs = cls_logits.softmax(-1)[..., :-1]               # drop "no object"
        for b in range(pv.shape[0]):
            best = {}
            score, qcls = probs[b].max(-1)                     # best class per query
            for q in range(probs.shape[1]):
                c = int(qcls[q]); s = float(score[q])
                if s < 0.5:
                    continue
                if c not in best or s > best[c][1]:
                    best[c] = (q, s)
            gt_m, gt_c = batch["mask_labels"][b], batch["class_labels"][b]
            gt = {int(c): gt_m[j].numpy() > 0.5 for j, c in enumerate(gt_c)}
            for c in range(NUM_CLASSES):
                g = gt.get(c)
                p = (mask_logits[b, best[c][0]].sigmoid() > 0.5).cpu().numpy() if c in best else None
                if g is None and p is None:
                    continue
                if g is not None:
                    seen[c] = True
                g = np.zeros((h, w), bool) if g is None else g
                p = np.zeros((h, w), bool) if p is None else p
                inter[c] += np.logical_and(g, p).sum()
                union[c] += np.logical_or(g, p).sum()
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    model.train()
    return float(iou[seen].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", default="facebook/mask2former-swin-tiny-coco-instance")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "runs", "mask2former"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time a step at several sizes, then exit")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = (args.width, args.height)

    print(f"loading {args.checkpoint}")
    model = build_model(args.checkpoint).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n/1e6:.1f}M   (YOLO11m-seg is 22.4M)")

    if args.bench:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"\n{'size':>12s} {'s/step':>8s} {'peakGB':>7s}   est 60ep @bs{args.batch}")
        for w, h in [(768, 416), (1024, 544), (1280, 672)]:
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                pv = torch.randn(args.batch, 3, h, w, device=device)
                ml = [torch.zeros((28, h, w), device=device) for _ in range(args.batch)]
                cl = [torch.arange(28, device=device) for _ in range(args.batch)]
                for i in range(3):
                    if i == 1:
                        torch.cuda.synchronize(); t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(pixel_values=pv, mask_labels=ml, class_labels=cl).loss
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                torch.cuda.synchronize()
                per = (time.time() - t0) / 2
                gb = torch.cuda.max_memory_allocated() / 1e9
                hrs = per * (1422 // args.batch) * 60 / 3600
                print(f"{w}x{h:<6} {per:8.2f} {gb:7.1f}   {hrs:5.1f} h")
            except RuntimeError as e:
                print(f"{w}x{h:<6}  FAILED: {str(e)[:60]}")
                torch.cuda.empty_cache()
        return

    tr = OPGInstanceDataset(args.data, "train", size, augment=True)
    va = OPGInstanceDataset(args.data, "val", size, augment=False)
    if args.smoke:
        tr.ids, va.ids, args.epochs = tr.ids[:8], va.ids[:4], 2
    print(f"train {len(tr)} | val {len(va)} | input {args.width}x{args.height}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       collate_fn=collate, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    os.makedirs(args.out, exist_ok=True)
    best, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        t0, run, seen = time.time(), 0.0, 0
        for batch in dl_tr:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            ml = [m.to(device) for m in batch["mask_labels"]]
            cl = [c.to(device) for c in batch["class_labels"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=pv, mask_labels=ml, class_labels=cl).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * pv.shape[0]; seen += pv.shape[0]
        miou = evaluate(model, dl_va, device, size)
        history.append({"epoch": epoch, "loss": run / max(seen, 1), "mIoU": miou})
        print(f"epoch {epoch:3d}/{args.epochs}  loss {run/max(seen,1):.4f}  mIoU {miou:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "args": vars(args), "mIoU": miou},
                       os.path.join(args.out, "best.pt"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump({"history": history, "best_mIoU": best, "args": vars(args)}, f, indent=2)

    print(f"\nBest mIoU {best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
