#!/usr/bin/env python3
"""
DINOv3 encoder + LoRA for 32-class FDI tooth segmentation on panoramic radiographs.

WHY SEMANTIC SEGMENTATION AND NOT INSTANCE SEGMENTATION
-------------------------------------------------------
FDI numbering assigns every tooth a unique identifier within a mouth, so a given class can
appear at most once per image. Verified across all 1,423 annotated OPGs in this dataset: zero
images contain a repeated class. That collapses the task -- 32-class *instance* segmentation
is exactly 33-class *semantic* segmentation (32 teeth + background), and each class's mask is
its own instance. No detection stage, no mask proposals, no NMS.

This is what makes a plain foundation-model encoder usable here. SAM-family models are
promptable but class-agnostic; they cannot number teeth, so they need a detector in front.
DINOv3 needs nothing in front, because dense per-pixel classification already answers both
"where is the tooth" and "which tooth is it".

CAVEAT worth stating in the paper: the assumption breaks on supernumerary teeth (mesiodens,
distomolars) -- roughly 1-3% of patients. This dataset's protocol excludes them, as AKUDENTAL's
does. A deployed system must detect that case and fall back, or it will silently mislabel.

ADAPTATION STRATEGY
-------------------
The encoder is frozen and adapted with LoRA on the attention projections; only the adapters and
the decoder head train. On a dataset of ~1.3k images, full fine-tuning of a ViT-L would mostly
memorise. This is also the comparison DinoDental (2026) reports as strongest for dental data.

Usage:
  python train_dinov3_lora.py --smoke                    # 2 epochs, tiny, proves the graph
  python train_dinov3_lora.py                            # ViT-B, full run
  python train_dinov3_lora.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --batch 2
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "merged_opg_dataset")
NUM_CLASSES = 33          # 32 FDI teeth + background at index 0
IGNORE_INDEX = 255
SEED = 42

# Source images are 1615x840 (ratio 1.923). Keep close to that rather than letterboxing to
# square: patch-16 means the input must be a multiple of 16 in both dimensions.
DEFAULT_W, DEFAULT_H = 1024, 528   # 64 x 33 = 2112 patch tokens, ratio 1.939


# ----------------------------------------------------------------------------- data
class OPGSegDataset(Dataset):
    """YOLO polygon labels rasterised into a dense 33-class mask.

    Labels are 'cls x1 y1 x2 y2 ...' with normalised coordinates, so rasterisation happens at
    the target resolution directly and no mask is ever resized -- resizing an integer label map
    with anything but nearest-neighbour silently invents classes at boundaries.
    """

    def __init__(self, root, split, size, augment=False):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.size = size
        self.augment = augment
        self.ids = sorted(os.path.splitext(f)[0] for f in os.listdir(self.img_dir))
        if not self.ids:
            raise RuntimeError(f"no images under {self.img_dir}")

    def __len__(self):
        return len(self.ids)

    def _load_mask(self, image_id, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)          # 0 = background
        path = os.path.join(self.lbl_dir, f"{image_id}.txt")
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                cls = int(parts[0])
                pts = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                cv2.fillPoly(mask, [pts.astype(np.int32)], cls + 1)   # +1: 0 is background
        return mask

    def __getitem__(self, i):
        image_id = self.ids[i]
        w, h = self.size
        img = cv2.imread(os.path.join(self.img_dir, f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = self._load_mask(image_id, w, h)

        if self.augment:
            # Photometric only. A horizontal flip would mirror the arch and turn tooth 18 into
            # tooth 28 -- the labels would need remapping, so it is simply not done. Scanner
            # and exposure variation is the real domain shift anyway.
            if np.random.rand() < 0.8:
                alpha = 1.0 + np.random.uniform(-0.25, 0.25)     # contrast
                beta = np.random.uniform(-25, 25)                # brightness
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                gamma = np.random.uniform(0.7, 1.4)
                img = np.clip(((img / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.2:
                img = np.clip(img.astype(np.float32) +
                              np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

        # DINOv3 expects 3-channel ImageNet-normalised input; radiographs are greyscale.
        x = np.repeat(img[None], 3, axis=0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x), torch.from_numpy(mask).long(), image_id


# ---------------------------------------------------------------------------- model
class FusionDecoder(nn.Module):
    """DPT-style head: fuse four encoder depths, then upsample to full resolution.

    A single linear probe on the last layer is the standard DINOv3 eval protocol, but it is
    too coarse here -- the weak classes are the lower incisors, where the deficit is boundary
    precision specifically, and boundaries live in the earlier, higher-frequency layers.
    """

    def __init__(self, in_dim, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(in_dim, hidden, 1) for _ in range(4)])
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.GELU(),
            ) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, feats, out_hw):
        x = None
        for proj, fuse, f in zip(self.projs, self.fuse, feats):
            y = proj(f)
            x = y if x is None else F.interpolate(x, size=y.shape[-2:],
                                                  mode="bilinear", align_corners=False) + y
            x = fuse(x)
        x = self.head(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class DINOv3Segmenter(nn.Module):
    def __init__(self, model_name, lora_r=16, lora_alpha=32, freeze_encoder=True):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        dim = self.encoder.config.hidden_size
        self.patch = getattr(self.encoder.config, "patch_size", 16)
        n_layers = self.encoder.config.num_hidden_layers
        # Evenly spaced depths; the last index must be the final layer.
        self.taps = [n_layers // 4 - 1, n_layers // 2 - 1, 3 * n_layers // 4 - 1, n_layers - 1]

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            targets = self._lora_targets()
            self.encoder = get_peft_model(self.encoder, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                target_modules=targets, bias="none"))
            print(f"  LoRA on {len(targets)} module name(s): {sorted(targets)}")

        self.decoder = FusionDecoder(dim)

    def _lora_targets(self):
        """Attention projections, discovered by name so this survives arch naming changes."""
        wanted = ("query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "dense")
        found = set()
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in wanted and "mlp" not in name:
                    found.add(leaf)
        if not found:
            sys.exit("Could not locate attention projections for LoRA; inspect named_modules().")
        return list(found)

    def forward(self, x):
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        feats = []
        for t in self.taps:
            tok = hs[t + 1]                       # hidden_states[0] is the embedding output
            # DINOv3 prepends a CLS token plus register tokens; their count varies by
            # checkpoint, so derive it rather than hardcoding.
            prefix = tok.shape[1] - gh * gw
            tok = tok[:, prefix:, :]
            feats.append(tok.transpose(1, 2).reshape(b, -1, gh, gw))
        return self.decoder(feats, (h, w))


# ----------------------------------------------------------------------------- loss
def dice_loss(logits, target, eps=1.0):
    """Soft Dice over present classes. CE alone under-weights the small lower incisors."""
    num_classes = logits.shape[1]
    probs = logits.softmax(1)
    valid = target != IGNORE_INDEX
    t = torch.where(valid, target, torch.zeros_like(target))
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    probs = probs * valid.unsqueeze(1)
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


# ------------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(model, loader, device, labels, amp_dtype):
    model.eval()
    inter = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    area_p = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    area_t = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x).argmax(1)
        for c in range(NUM_CLASSES):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
            area_p[c] += p.sum().item()
            area_t[c] += t.sum().item()
    iou = (inter / union.clamp(min=1)).numpy()
    dice = (2 * inter / (area_p + area_t).clamp(min=1)).numpy()
    seen = (area_t > 0).numpy()
    tooth = seen.copy()
    tooth[0] = False                                  # exclude background from the summary
    res = {
        "mIoU": float(iou[tooth].mean()),
        "mDice": float(dice[tooth].mean()),
        "bg_IoU": float(iou[0]),
        "per_class": {labels.get(c - 1, f"class{c-1}"): float(iou[c])
                      for c in range(1, NUM_CLASSES) if seen[c]},
    }
    model.train()
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4, help="decoder + LoRA adapters")
    ap.add_argument("--lora-r", type=int, default=16, help="0 disables LoRA (frozen probe)")
    ap.add_argument("--full-finetune", action="store_true",
                    help="unfreeze the encoder; expect overfitting at this dataset size")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "runs", "dinov3_lora"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        print("WARNING: no CUDA. This will be extremely slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else None

    import yaml
    with open(os.path.join(args.data, "data.yaml")) as f:
        labels = yaml.safe_load(f).get("names", {})

    # Input must tile exactly into patches. Patch size is a property of the checkpoint (16 for
    # DINOv3, 14 for DINOv2), so read it rather than assuming, and snap the requested size down
    # to the nearest valid multiple instead of failing.
    from transformers import AutoConfig
    patch = getattr(AutoConfig.from_pretrained(args.model), "patch_size", 16)
    width = (args.width // patch) * patch
    height = (args.height // patch) * patch
    if (width, height) != (args.width, args.height):
        print(f"  adjusted {args.width}x{args.height} -> {width}x{height} "
              f"(patch size {patch})")
    args.width, args.height = width, height
    size = (width, height)
    tr = OPGSegDataset(args.data, "train", size, augment=True)
    va = OPGSegDataset(args.data, "val", size, augment=False)
    if args.smoke:
        tr.ids, va.ids, args.epochs = tr.ids[:16], va.ids[:8], 2
    print(f"train {len(tr)} | val {len(va)} | input {args.width}x{args.height} "
          f"({args.width//16}x{args.height//16} patches)")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    print(f"Loading {args.model} ...")
    model = DINOv3Segmenter(args.model, lora_r=0 if args.full_finetune else args.lora_r,
                            freeze_encoder=not args.full_finetune).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    steps = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    os.makedirs(args.out, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0, run, n = time.time(), 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = ce(logits, y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        m = evaluate(model, dl_va, device, labels, amp_dtype)
        history.append({"epoch": epoch, "loss": run / max(1, n), **
                        {k: v for k, v in m.items() if k != "per_class"}})
        # flush=True: stdout is block-buffered when piped (tee, nohup), which otherwise hides
        # all progress until the process exits or 4 KB accumulates.
        print(f"epoch {epoch:3d}/{args.epochs}  loss {run/max(1,n):.4f}  "
              f"mIoU {m['mIoU']:.4f}  mDice {m['mDice']:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        with open(os.path.join(args.out, "progress.jsonl"), "a") as pf:
            pf.write(json.dumps(history[-1]) + "\n")
        if m["mIoU"] > best:
            best = m["mIoU"]
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": m},
                       os.path.join(args.out, "best.pt"))
            worst = sorted(m["per_class"].items(), key=lambda kv: kv[1])[:5]
            print("    new best; weakest: " +
                  ", ".join(f"{k.split('(')[0].strip()} {v:.3f}" for k, v in worst))

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": history, "best_mIoU": best, "args": vars(args)}, f, indent=2)
    print(f"\nBest mIoU {best:.4f}  ->  {args.out}/best.pt")


if __name__ == "__main__":
    main()
