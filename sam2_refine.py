#!/usr/bin/env python3
"""
SAM2 boundary refinement for predicted tooth masks.

WHY THIS EXPERIMENT
-------------------
Four independent results say the binding constraint in panoramic tooth segmentation is boundary
precision, not detection or labelling:

  - resolution 640->1280 moves mask mAP50-95 by 9 points and leaves mAP50 flat at ~0.988
  - zero-shot to DENTEX costs 62% of mAP50-95 but only 18% of mAP50
  - the weakest classes are the physically smallest teeth, under every architecture tested
  - a Hungarian assignment aimed at label space gained +0.0006, i.e. nothing

So the intervention should target mask edges. SAM2 is a promptable segmenter with strong
boundary priors; it cannot name teeth (it is class-agnostic, which is why it can never be the
whole system here), but it does not need to -- the detector supplies both the prompt and the
FDI label, and SAM2 only redraws the outline.

DESIGN
------
Each detection's box becomes a SAM2 prompt. SAM2 returns several candidate masks; picking by
its own predicted IoU is wrong here, because on a radiograph the highest-scoring candidate is
often the whole jaw or a restoration rather than the tooth. Instead we pick the candidate with
the highest overlap against the detector's original mask: the detector decides *which object*,
SAM2 decides *where its edge is*.

A refined mask is accepted only when it agrees with the original above --min-iou. Below that,
SAM2 has almost certainly latched onto a different structure and the original is kept. This
makes the method strictly conservative: it can improve boundaries or do nothing, but it cannot
silently substitute a different object.

EXPECTATION
-----------
SAM2 is trained on natural images. Panoramic radiographs are low-contrast, and adjacent teeth
overlap in projection with genuinely ambiguous cervical margins -- the exact conditions where a
natural-image prior may not transfer. A null or negative result is a real possibility and would
itself be informative, given how often SAM-family models are proposed for dental imaging.

Usage:
  from sam2_refine import Sam2Refiner
  r = Sam2Refiner()
  mask = r.refine(image_rgb, box_xyxy, original_mask)
"""

import numpy as np


class Sam2Refiner:
    def __init__(self, model_id="facebook/sam2.1-hiera-large", device="cuda",
                 min_iou=0.5, dtype=None):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = device
        self.min_iou = min_iou
        self.dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()
        self.stats = {"seen": 0, "refined": 0, "rejected": 0, "no_output": 0}

    @staticmethod
    def _iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 0.0

    def set_image(self, image_rgb):
        """Encode once per image; the vision encoder is the expensive part."""
        self._image = image_rgb

    def refine_batch(self, image_rgb, boxes, orig_masks):
        """Refine every detection in one image. boxes: [[x0,y0,x1,y1], ...] in pixels."""
        torch = self.torch
        if not len(boxes):
            return []
        out = []
        inputs = self.processor(images=image_rgb,
                                input_boxes=[[list(map(float, b)) for b in boxes]],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.autocast(self.device, dtype=self.dtype,
                                enabled=(self.device == "cuda")):
                res = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            res.pred_masks.float().cpu(),
            inputs["original_sizes"],
        )[0].numpy()                      # (n_boxes, n_candidates, H, W)

        for i, orig in enumerate(orig_masks):
            self.stats["seen"] += 1
            cands = masks[i]
            if cands.ndim == 2:
                cands = cands[None]
            best, best_iou = None, -1.0
            for c in cands:
                cm = (c > 0).astype(np.uint8)
                if cm.sum() == 0:
                    continue
                v = self._iou(cm, orig)
                if v > best_iou:
                    best, best_iou = cm, v
            if best is None:
                self.stats["no_output"] += 1
                out.append(orig)
            elif best_iou < self.min_iou:
                # SAM2 latched onto a different structure -- keep the detector's mask.
                self.stats["rejected"] += 1
                out.append(orig)
            else:
                self.stats["refined"] += 1
                out.append(best)
        return out

    def report(self):
        s = self.stats
        n = max(s["seen"], 1)
        return (f"SAM2 refinement: {s['refined']}/{s['seen']} masks replaced "
                f"({100*s['refined']/n:.1f}%), {s['rejected']} rejected below IoU "
                f"{self.min_iou}, {s['no_output']} produced no mask")
