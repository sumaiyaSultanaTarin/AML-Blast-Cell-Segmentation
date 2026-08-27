# Ratio-Consistency Loss and Ratio-Feedback Gating for Nucleus–Cytoplasm Segmentation

**A Measurement-Reliability Evaluation in Acute Myeloid Leukemia**

Sumaiya Sultana Tarin, Shohana Khondoker, Fahim Tarik Al Said
Department of Computer Science, American International University-Bangladesh (AIUB)

---

## Overview

Segmentation models for white blood cell (WBC) images are typically trained with overlap-based losses (Dice, Focal), which do not guarantee that a downstream clinical measurement — such as the nucleus-to-cytoplasm (N:C) ratio used in acute myeloid leukemia (AML) morphology — is preserved. A model can achieve a high Dice score while still distorting the N:C ratio enough to affect a cytomorphological assessment.

This repository contains the implementation for a study that:

1. Proposes a **ratio-consistency loss** that penalizes N:C log-ratio error directly during segmentation training.
2. Proposes a **Ratio-Feedback Gate (RFG)**, a lightweight decoder module that estimates the network's own N:C ratio and uses it to recalibrate features before the segmentation head.
3. Evaluates both mechanisms in a controlled ablation (30 runs, 5 seeds) on the source-domain **WBCAtt+** dataset across two backbones (Attention U-Net with scSE decoder attention, and a plain U-Net).
4. Transfers the method to a pseudo-labeled, human-spot-checked target-domain subset of **AML-Cytomorphology_LMU**, comparing training from scratch against WBCAtt+-pretrained initialization.
5. Evaluates every configuration with a measurement-reliability battery — Dice, N:C ratio error, boundary F1, predictive-entropy calibration, Bland-Altman agreement, and ICC(A,1) — not just overlap metrics.

**Headline finding:** the two proposed mechanisms are backbone-dependent and do not uniformly improve reliability. The clearest result in the study is that transfer learning significantly improves Dice on the AML target domain (0.77 → 0.86, *p* = 0.007) **without** a matched improvement in N:C ratio error (*p* = 0.78) — direct evidence that overlap-metric gains and morphometric-reliability gains are separable outcomes, and that segmentation intended for morphometric use should be evaluated by the downstream measurement itself.

---

## Repository Structure

```
.
├── AML_Phase_I.ipynb    # Source-domain (WBCAtt+) pipeline: data prep, backbones,
│                         # RFG + ratio-consistency loss, full ablation, evaluation
├── AML_Phase_II.ipynb   # Target-domain (AML-Cytomorphology_LMU) pipeline: pseudo-label
│                         # generation, gold-subset creation, transfer learning, RFG ablation
├── main.tex              # Manuscript source (IEEE format)
├── references.bib     # Bibliography
├── Figures.draw.io         
├── requirements.txt      # Python package dependencies
├── config.yaml           # Consolidated reference of all hyperparameters used
└── README.md
```

**Scripts:** this project has no standalone `.py` scripts. All pipeline code — data loading, model/loss definitions, training loops, ablation orchestration, and statistical evaluation — is implemented inline within the two notebooks above, consistent with the Google Colab-based workflow described in the paper (Section IV-A, Software and Hardware).

---

## Method Summary

**Ratio-consistency loss.** Given predicted soft nucleus/cytoplasm areas $\hat{A}_n, \hat{A}_c$ and their expert-mask counterparts $A_n, A_c$:

$$\mathcal{L}_{ratio} = \text{Huber}\left[\log\frac{\hat{A}_n+\epsilon}{\hat{A}_c+\epsilon} - \log\frac{A_n+\epsilon}{A_c+\epsilon}\right]$$

**Ratio-Feedback Gate (RFG).** A lightweight probe head estimates the network's own current N:C log-ratio $r$ from the decoder's final feature map $x$; $r$ is passed through a two-layer MLP to produce a per-channel gate $g$, which recalibrates the features via a residual update $x' = x + x \odot g$ before the segmentation head.

**Total training loss:**

$$\mathcal{L} = \mathcal{L}_{Dice} + \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{ratio} + \lambda_3 \mathcal{L}_{probe}$$

where $\mathcal{L}_{probe}$ auxiliary-supervises the RFG probe head's own ratio estimate. $\lambda_2$ and $\lambda_3$ are ramped from 0 to 0.3 over the first 25% of training whenever the corresponding component is enabled.

---

## Results Summary

**Phase I — WBCAtt+ ablation (mean ± std, 5 seeds):**

| Backbone | RFG / Ratio loss | Dice | N:C ratio error |
|---|---|---|---|
| Attention U-Net | off / off (baseline) | 0.9885 ± 0.0005 | 0.0300 ± 0.0008 |
| Attention U-Net | off / on | 0.9874 ± 0.0009 | 0.0317 ± 0.0012 |
| Attention U-Net | on / off | 0.9872 ± 0.0012 | 0.0314 ± 0.0015 |
| Attention U-Net | on / on (full method) | 0.9876 ± 0.0005 | 0.0311 ± 0.0005 |
| Plain U-Net | off / on | 0.9851 ± 0.0035 | 0.0332 ± 0.0030 |
| Plain U-Net | on / on | 0.9866 ± 0.0010 | 0.0317 ± 0.0008 |

**Phase II — AML transfer learning (Attention U-Net, full method, gold-subset evaluation):**

| Regime | Dice | N:C ratio error |
|---|---|---|
| Random initialization | 0.768 | 0.499 |
| WBCAtt+-pretrained | 0.858 | 0.492 |

Pretraining significantly improves Dice (*p* = 0.007) but not N:C ratio error (*p* = 0.78).

Full ablation tables, confidence intervals, effect sizes, and target-domain RFG results are in the manuscript (Section V).

---

## Datasets

- **WBCAtt+** (source domain, 10,298 images, official 6,169/1,030/3,099 split): [Hugging Face](https://huggingface.co/datasets/apple2373/wbcattplus) · [GitHub](https://github.com/apple2373/wbcattplus)
- **AML-Cytomorphology_LMU** (target domain, 18,250 images, 15 classes): [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/) · DOI: [10.7937/tcia.2019.36f5o9ld](https://doi.org/10.7937/tcia.2019.36f5o9ld)

AML-Cytomorphology_LMU has no expert nucleus/cytoplasm masks. Reference masks for the 345-image training subset are generated automatically (Otsu thresholding of the HSV saturation channel + morphological cleanup + center-component selection; see `AML_Phase_II.ipynb`) and are explicitly **pseudo-labels, not ground truth**. A separate 50-image subset was manually corrected by a human reviewer to form an independent **gold subset** used for evaluation.

---

## Environment & Hardware

| | |
|---|---|
| Python | 3.11 |
| Deep learning framework | PyTorch (with `torch.amp` autocast + `GradScaler` for mixed-precision training) |
| Segmentation library | [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (`smp.Unet`, ResNet-34 encoder, ImageNet-pretrained weights) |
| Compute | Google Colab, single GPU per run |
| GPU | NVIDIA Tesla T4 (as reported by `torch.cuda.get_device_name(0)` in both notebooks) |
| Storage | Google Drive, mounted for persistent dataset/checkpoint storage across sessions |

## Requirements

- PyTorch, torchvision
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- Albumentations
- OpenCV (`opencv-python`)
- scikit-image (for `threshold_multiotsu`, used in AML pseudo-label generation)
- SciPy (for `ndimage` morphology and `stats.ttest_rel`)
- pandas
- statsmodels
- [pingouin](https://pingouin-stats.org/) (for ICC(A,1) computation)
- `huggingface_hub` (to download WBCAtt+)

```bash
pip install -r requirements.txt
```

**Note on versions:** the original experiments were run via unpinned `pip install` calls inside Colab (e.g. `!pip install -q segmentation-models-pytorch albumentations huggingface_hub`), so exact package versions were not logged and are not pinned in `requirements.txt`. Results should be close but may not be bit-for-bit identical on newer releases of these libraries. If you need an exact match, run `pip freeze` in a fresh install and compare against a Colab environment from the same period as the original runs.

## Configuration

`config.yaml` consolidates every hyperparameter used across both notebooks (optimizer settings, epoch/patience budgets, loss weights and ramp schedules, augmentation parameters, dataset targets, pseudo-label thresholds, and seeds) into one reference file. The notebooks themselves do not read from this file — each notebook cell sets these values inline, matching Google Colab's single-file, cell-by-cell workflow — so `config.yaml` exists as a single place to check or change a setting before manually editing the corresponding cell, not as a file the code loads at runtime.

## Usage

Both notebooks were developed and run on **Google Colab**, using a mounted Google Drive folder for persistent dataset/checkpoint storage across sessions.

1. **Phase I (source domain).** Open `AML_Phase_I.ipynb` in Colab.
   - Run the setup cells (seed/device setup, Google Drive mount, package installs).
   - Run the WBCAtt+ download and official-split cells (downloads from Hugging Face automatically).
   - Run the model/loss definition cells (`SegModel`, `RatioFeedbackGate`, `DiceLoss`, `FocalLoss`, `RatioConsistencyLoss`, `TotalLoss`).
   - Run the ablation-batch cell — this trains all 30 configurations (6 configurations × 5 seeds) via `maybe_run(...)`, which skips any configuration whose checkpoint already exists, so the cell is safe to re-run after an interruption. Checkpoints and per-epoch history are saved under `<PROJECT_DIR>/checkpoints/` and `<PROJECT_DIR>/logs/`.
   - Run the remaining cells to reproduce the results tables, confusion matrices, boundary-F1, entropy calibration, Bland-Altman, and ICC analysis.
2. **Between phases.** Obtain `AML-Cytomorphology_LMU` from TCIA and place it under the same Google Drive project folder (`PKG - AML-Cytomorphology_LMU/`, one subfolder per class, matching the paths read in `AML_Phase_II.ipynb`).
3. **Phase II (target domain).** Open `AML_Phase_II.ipynb`, mount the same Drive, and run top to bottom:
   - Dataset audit and pseudo-label generation (`generate_pseudo_label`) cells.
   - Gold-subset and training-subset construction cells.
   - Training cells for the transfer-learning comparison (scratch vs. WBCAtt+-pretrained) and the target-domain RFG ablation for both backbones — these load the best Phase I checkpoint (`attention_unet_rfg1_ratio1_seed2.pt`) where applicable, and use the same skip-if-already-run pattern as Phase I.
   - Evaluation cells for the gold-subset results, Bland-Altman plots, and ICC analysis.

Both notebooks are resumable at two levels: per-run state (optimizer/scheduler/scaler/history) is checkpointed every epoch so an interrupted training run can resume mid-way, and per-configuration state is checked before each run starts so a fully-finished configuration is skipped on re-execution.

---

## Mapping Paper Results to Code

Every quantitative result in the manuscript traces back to a specific notebook and cell group:

| Paper item | Notebook | Section / cells |
|---|---|---|
| Table I — dataset roles and counts | Both | WBCAtt+ split counts (`AML_Phase_I.ipynb`, split/mask-inspection cells); AML class counts and subset sampling (`AML_Phase_II.ipynb`, dataset-audit and subset-sampling cells) |
| Table II — backbone roles | `AML_Phase_I.ipynb` | `SegModel` class definition |
| Eq. 1–6, RFG and total loss | `AML_Phase_I.ipynb` (reused in `AML_Phase_II.ipynb`) | `RatioFeedbackGate`, `DiceLoss`, `FocalLoss`, `RatioConsistencyLoss`, `TotalLoss`, `lambda2_ramp` |
| Table III — Phase I ablation summary | `AML_Phase_I.ipynb` | Results-aggregation cell reading `logs/*_history.csv` into `results_df`/`summary_df` |
| Table IV — paired comparisons (Δ, 95% CI, Cohen's *d*) | `AML_Phase_I.ipynb` | `compare_configs` cell (paired *t*-test, Cohen's *d*, from the same per-seed values as Table III); 95% CIs computed from that per-seed data for the manuscript |
| Confusion matrices (Attention U-Net, plain U-Net) | `AML_Phase_I.ipynb` | `compute_confusion_matrix_streaming` + `plot_confusion_matrix` cells, run once per backbone |
| Boundary F1 | `AML_Phase_I.ipynb` | `compute_boundary_f1` cells (Attention U-Net and plain U-Net) |
| Predictive-entropy calibration | `AML_Phase_I.ipynb` | `compute_entropy_calibration` cells |
| Bland-Altman agreement, WBCAtt+ | `AML_Phase_I.ipynb` | `compute_ratios` + `bland_altman_plot` cells |
| ICC(A,1), WBCAtt+ | `AML_Phase_I.ipynb` | `compute_icc` cells (using `pingouin`) |
| AML pseudo-label generation | `AML_Phase_II.ipynb` | `generate_pseudo_label` function and visual-check cells |
| Gold subset (50 images) construction | `AML_Phase_II.ipynb` | Gold-candidate rebuild and `GoldDataset` cells |
| Training subset (345 images), data splits | `AML_Phase_II.ipynb` | Subset-sampling cell (`BLAST_CLASSES`, per-class targets) and `stratified_split` |
| Table V — transfer-learning comparison | `AML_Phase_II.ipynb` | `run_phase2_experiment` (scratch/pretrained) training cells and gold-subset evaluation cell |
| Table VI — target-domain RFG ablation | `AML_Phase_II.ipynb` | `run_phase2_experiment_v2`/`_v3` (RFG on/off, both backbones) training cells and gold-subset evaluation cells |
| Gate-value near-identity observation (Section V-C) | Both | `gate_mean` column logged in every training cell's per-epoch printout across both notebooks |
| Qualitative gold-subset confusion matrix / boundary F1 / entropy / ICC (single-seed diagnostics) | `AML_Phase_II.ipynb` | `run_gold_rich_evaluation` cells |

---

## Citation

If you use this code or method, please cite:

```bibtex
@article{tarin2026ratioconsistency,
  title   = {Ratio-Consistency Loss and Ratio-Feedback Gating for Nucleus--Cytoplasm
             Segmentation: A Measurement-Reliability Evaluation in Acute Myeloid Leukemia},
  author  = {Tarin, Sumaiya Sultana and Khondoker, Shohana and Al Said, Fahim Tarik},
  year    = {2026},
  note    = {Department of Computer Science, American International University-Bangladesh (AIUB)}
}
```

## License

Released under the [MIT License](LICENSE) unless noted otherwise — update this section if a different license is preferred.

## Contact

- Sumaiya Sultana Tarin — 26-94006-1@student.aiub.edu
- Shohana Khondoker — 26-94165-2@student.aiub.edu
- Fahim Tarik Al Said — 26-94096-2@student.aiub.edu
