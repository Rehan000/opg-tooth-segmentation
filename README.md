# Tooth segmentation and FDI numbering on panoramic radiographs

Code accompanying **"Detection is solved, delineation is not: what governs tooth segmentation on
panoramic radiographs"** (Rehan, Amjad, Ahmed, Adnan, Ali — AI Dentify).

The paper holds the corpus, the evaluation harness and the post-processing fixed, varies one
factor at a time, and reports every difference with a paired bootstrap confidence interval. This
repository is that harness: training, evaluation, cross-validation, the statistical tests, the
contamination audit, and each of the three interventions the paper reports as failures.

## What is and is not here

**Here:** all code needed to reproduce the analyses.

**Not here:** the radiographs and the annotation layer. The imagery derives from a public
redistribution whose upstream clinical origin and licence terms are not documented by the
redistributor, so we cannot redistribute it; the annotations are not publicly released. No
dataset, model weight file, or identifier mapping is included in this repository, and
`.gitignore` is written to keep it that way.

**The zero-shot evaluation is fully reproducible from public data.** DENTEX is available from its
original authors under CC BY-NC-SA 4.0. `evaluate_models.py` scores any of the architectures on
it with the same code path used for the in-domain numbers, so the transfer result in the paper
can be checked independently without access to our corpus.

## Layout

### Training
| | |
|---|---|
| `train_opg_seg.py` | YOLO11m-seg. Horizontal flip is disabled — mirroring a panoramic radiograph swaps left and right quadrants and inverts every FDI label |
| `train_mask2former.py` | Mask2Former with a Swin-T backbone |
| `train_dinov3_lora.py` | DINOv3-B frozen encoder, LoRA on the attention projections, DPT-style decoder |
| `run_cross_validation.py` | Five-fold grouped, track-stratified CV. Folds are grouped so the one patient imaged twice cannot straddle a split |
| `summarise_cv.py` | Aggregates fold results |

### Evaluation and statistics
| | |
|---|---|
| `evaluate_models.py` | Scores every architecture on identical metrics and identical mask representations, at original image resolution. In-domain or zero-shot on DENTEX |
| `dump_eval_records.py` | Caches per-(image, class) detection scores and IoUs |
| `stats_tests.py` | Paired bootstrap over images: observed difference, 95% percentile CI, two-sided bootstrap *p*, Holm correction, and the equivalence margin |

Because FDI uniqueness guarantees at most one ground-truth instance per class per image, and the
post-processing enforces at most one detection, the cached records permit exact reconstruction of
COCO's 101-point interpolation on any subset of images. The reconstruction reproduces the full
harness to within 5e-5 for every model and metric.

### Contamination audit
| | |
|---|---|
| `fingerprint_dataset.py` | Brightness/contrast-normalised embeddings, including mirrored variants |
| `dedup_correlation.py` | Exhaustive all-pairs normalised pixel correlation |

Perceptual hashing is unreliable on this modality: the median nearest-neighbour pHash distance
from our corpus to an external one equalled the median distance *within* our own set of distinct
radiographs, because every panoramic image shares the same global arch structure. Exact hashing
plus pixel correlation is required, which is what these two scripts implement.

### Interventions and analysis
| | |
|---|---|
| `sam2_refine.py` | SAM2 box-prompted mask refinement. Degrades mask mAP50-95 by 39% |
| `arch_assignment.py` | Globally optimal FDI assignment (Hungarian) with learned position priors |
| `validate_arch_rules.py` | Anatomical constraint checks on predictions |
| `boundary_error_anatomy.py` | Decomposes mask error along the tooth axis into crown / middle / apical thirds |
| `iaa_toolkit.py` | Inter-annotator agreement sampling utilities |

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is needed for training; evaluation runs on CPU.

## Reproducing the DENTEX result

Obtain DENTEX from its original authors, then:

```bash
python evaluate_models.py --model yolo --weights <checkpoint> --dataset dentex --out results.json
python dump_eval_records.py --model yolo --dataset dentex --out records.json
python stats_tests.py --a records_a.json --b records_b.json
```

FDI codes reconstruct from DENTEX's quadrant and enumeration annotations as
`quadrant * 10 + position`, an exact match to our taxonomy.

## Citation

```bibtex
@article{rehan2026delineation,
  title   = {Detection is solved, delineation is not: what governs tooth
             segmentation on panoramic radiographs},
  author  = {Rehan, Muhammad and Amjad, Moaz and Ahmed, Syed Danial and
             Adnan, Mariam and Ali, Haider},
  journal = {Medical Image Analysis},
  note    = {Under review},
  year    = {2026}
}
```

## Licence

MIT — see [LICENSE](LICENSE).

## Competing interests

All authors are employed by AI Dentify, which develops commercial software for dental
radiographic analysis. The tooth segmentation model evaluated in the paper is deployed in that
company's product.
