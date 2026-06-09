# EchoPE: Adapting a Cardiac Ultrasound Foundation Model for Emergency Pulmonary-Embolism Right-Heart Phenotype Recognition

> **Peer-review notice.** This repository is the code accompanying the manuscript
> **submitted to *npj Digital Medicine*** and is provided **for peer review only**.
> It is a code snapshot intended to let reviewers inspect the experimental
> pipeline; it is not yet a public release. Trained weights and the clinical
> HOPE dataset are **not** included (see *Code & Data availability*).

EchoPE is a video-classification framework that adapts the large-scale
echocardiography foundation model **EchoPrime** to a point-of-care ultrasound
(POCUS) task: recognizing pulmonary-embolism (PE)-related right-heart strain
phenotypes from emergency-department cardiac ultrasound videos. It was developed
on the **HOPE dataset** (*ecHO-Pulmonary Embolism*), a real-world emergency POCUS
video dataset.

The framework has five components:

1. **Preprocessing** — remove machine-interface overlays, segment the ultrasound
   sector, crop the region of interest, and standardize videos to `256 x 256`.
2. **View encoder** — a ConvNeXt classifier that infers the echocardiographic
   view from the first frame and provides a view-level prior.
3. **Echo encoder** — the EchoPrime MViT-v2-s video encoder, adapted to learn
   PE-related right-heart motion/morphology features.
4. **PE classification head** — combines video-level and view-level features to
   output *normal* vs *PE-related right-heart phenotype*.
5. **Interpretability module** — attention maps, temporal attribution, occlusion
   sensitivity, and weakly-supervised ROI analysis.

---

## Repository layout

The repository is organized by function. Each top-level directory is an
importable package used by the others.

```
.
├── echo_paths.py            # Resolves the project root and sets the working dir for imports
├── utils/
│   └── split_cv.py          # Builds stratified 5-fold CV splits from the dataset manifest
├── data/
│   └── all_dataset_preprocessed_cv_seed2026/   # The 5-fold cross-validation splits used in the paper
├── classification/          # Main PE binary classification (EchoPE vs EchoPrime+head, 5-fold CV)
├── finetuning/              # Adaptation strategies: LoRA, knowledge distillation, partial-FT, view-aux
├── baseline/                # Lightweight cached-feature PE classifier prototype
├── inference/               # Batch EchoPrime inference helper over a video folder
└── experiment/              # Downstream analyses that consume the trained models
    ├── view/                # Echocardiographic view classification (6-/4-/3-way label spaces)
    ├── interpretability/    # Attention / gradient / temporal attribution and embedding analysis
    ├── roi_analysis/        # ROI attribution, perturbation, and EchoPrime semantic probing
    └── phenotype/           # EchoPrime phenotype probing (zero-shot transfer baseline)
```

### `classification/` — main PE binary classification
- `config.py` — default experiment configuration.
- `data.py` — scan the dataset, build the `6-view -> coarse4` mapping and split manifests.
- `train_full_run.py` — train a single `seed + task` (pooled or per-view) model.
- `train_cv_binary.py` — unified 5-fold CV training entry point.
- `run_full_suite.py` — run the full pooled + 4 per-view suite.
- `evaluate_full_run.py` — medical metrics and ROC/PR curves from saved probabilities.
- `evaluate_binary_cv.py` — high-sensitivity operating-point evaluation (recall fixed at 0.90 / 0.95).
- `export_cv_curve_data.py` — export pooled / mean±sd ROC-PR curve data and figures.
- `analyze_full_run.py` — aggregate multi-seed / multi-task results.
- `make_temporal_holdout_split.py` — build the temporal hold-out split.

### `finetuning/` — adaptation strategies (ablation)
- `train_lora.py` — pure LoRA fine-tuning of the MViT encoder.
- `train_lora_distill.py` — full entry point for LoRA + knowledge distillation (KD) + view auxiliary supervision.
- `distill.py` — KD losses (`cos` / `l2` / `rkd-d` / `rkd-a` / `combo`) and frozen-teacher loading.
- `aux_heads.py` — view auxiliary heads (A4/PSS CE head and 11-class view-KD head).
- `compare_distill_ablations.py` — 7-configuration (frozen / LoRA / partial-FT, ± KD, ± view-aux) ablation runner.
- `compare_with_baseline.py` — multi-seed LoRA vs frozen-feature baseline comparison.

### `experiment/view/` — view classification
- `evaluate_view_classifier.py` — evaluate the frozen EchoPrime view head, mapped to the local label groups.
- `view_raw_finetune_scaling.py` — 6-way data-scaling experiment (direct / linear-probe / partial-FT / full-FT / scratch).
- `view_raw_finetune_cv.py` — 6-way external 5-fold CV runner.
- `view_grouped_finetune_scaling.py` — 4-way (`coarse4`) and 3-way (`family3`) grouped-label experiments.
- `view_finetune_data_scaling.py` — earlier 6-way data-scaling experiment (kept for provenance).
- `analyze_view_raw_finetune_scaling.py`, `analyze_view_grouped_finetune_scaling.py` — summary CSVs and plots.
- `make_nature_label_space_comparison.py` — label-space comparison figures (6-/4-/3-way).
- `make_nature_embedding_granularity.py` — penultimate-feature UMAP/t-SNE hidden-space figure.

### `experiment/interpretability/` — attention / gradient / temporal analysis
- `load_interpret_model.py` — restore a classification checkpoint / manifest / dataloader.
- `extract_attributions.py` — extract embeddings, attention, gradient×input, and temporal occlusion curves.
- `analyze_interpretability.py` — view-level and representation-level aggregation (PCA, probes, retrieval).
- `compare_interpret_runs.py` — compare frozen-head vs full-fine-tune interpretability outputs.
- `run_interpret_suite.py` — multi-seed / multi-task batch runner.
- `make_nature_case_figure.py` — per-case interpretability figure.
- `temporal_attention_head.py` — optional lightweight temporal-pooling head.

### `experiment/roi_analysis/` — ROI / perturbation / semantics
- `select_roi_cases.py` — select a shared case panel across models.
- `make_weak_context_masks.py` — generate weak acquisition-context masks (foreground / background / sector border / probe near-field).
- `quantify_roi_attribution.py` — ROI mass, enrichment, and top-10% overlap.
- `run_roi_perturbation.py` — ROI occlusion / perturbation effect on PE probability and logit.
- `run_report_semantics.py` — EchoPrime semantic probing (cosine drift, phenotype Spearman).
- `compare_roi_runs.py` — cross-model ROI comparison.
- `summarize_for_paper.py` — paper-ready tables and compact figures.
- `roi_utils.py`, `roi_annotation_schema.json` — ROI utilities and the annotation schema.

### `experiment/phenotype/` — EchoPrime phenotype probing baseline
- `run_echoprime_pe_phenotypes.py` — per-video EchoPrime `predict_metrics()` phenotype predictions.
- `analyze_pe_phenotype_correlations.py` — focus-phenotype statistics and plot-ready data.
- `plot_pe_phenotype_figure.py` — Normal-vs-PE phenotype figure from saved analysis.

### `baseline/` and `inference/`
- `baseline/train_pe_classifier.py`, `baseline/compare_heads.py` — early cached-feature PE classifier prototype and head comparison.
- `inference/run_testset.py` — batch EchoPrime inference / report helper over a video folder.

---

## HOPE dataset format

The scripts expect preprocessed videos organized by clinical label and view:

```
dataset/preprocessed/
├── Normal/{A4,IVC,MS,PSL,PSS,SX}/*.mp4
└── PE/{A4,IVC,MS,PSL,PSS,SX}/*.mp4
```

- Cohort: **4,069** videos — **1,934** normal, **2,135** PE-related right-heart phenotype.
- Views: A4 (apical 4-chamber), PSL, PSS (parasternal long/short axis), IVC, SX
  (subxiphoid); `MS` = A4 videos with McConnell's sign.
- Coarse view mapping (`coarse4`): `A4+MS -> A4`, `PSL`, `PSS`, `IVC+SX -> Subcostal`.
- The exact five-fold cross-validation splits used in the paper are included in
  this repository under
  `data/all_dataset_preprocessed_cv_seed2026/` (seed 2026, stratified by
  `label + parsed_view`). Each `fold_{0..4}/{train,val,test}.csv` lists
  de-identified video IDs, labels (`normal=0`, `pe=1`), and views; the full
  per-video assignment and manifest are in `fold_assignments.csv` and
  `manifest.csv`. These files contain no protected health information.

The HOPE video data itself is restricted clinical data and is **not** distributed
with this repository.

---

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
# Install the CUDA 12.8 PyTorch build used for the manuscript:
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`echo_paths.py` resolves the project root and switches the working directory so
that the EchoPrime model package (`echo_prime`, `utils`) and its `model_data`
weights are importable; the EchoPE echo/view encoders are initialized from those
released EchoPrime weights. Run each script from inside its package directory,
e.g.:

```bash
cd classification
python run_full_suite.py --device cuda
```

---

## Software & hardware environment

Pinned to the exact environment used to produce the reported results.

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA RTX A6000 |
| NVIDIA driver | 560.35.05 |
| Python | 3.11.13 |
| CUDA / cuDNN | 12.8 / 9.19.0 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| umap-learn | 0.5.12 |
| opencv-python-headless | 4.13.0.92 |
| Pillow | 12.2.0 |
| transformers | 4.57.0 |
| tqdm | 4.67.3 |

Single-process PyTorch training on one CUDA device is the default; independent
seeds / label spaces can be launched on different devices (e.g. `--device cuda:0`,
`--device cuda:1`).

---

## Reproducing the paper results

Package to manuscript Results section mapping.

| Results section / figure | Package | Key scripts |
| --- | --- | --- |
| Zero-shot EchoPrime phenotype transfer (Fig. performance a) | `experiment/phenotype/` | `run_echoprime_pe_phenotypes.py` → `analyze_pe_phenotype_correlations.py` → `plot_pe_phenotype_figure.py` |
| EchoPE vs EchoPrime+head, 5-fold ROC/PR (Fig. performance b) | `classification/` | `train_cv_binary.py`, `evaluate_full_run.py`, `evaluate_binary_cv.py`, `export_cv_curve_data.py` |
| Fine-tuning-strategy ablation (Fig. performance c, d) | `finetuning/` + `classification/` | `compare_distill_ablations.py`, `train_lora_distill.py`, `train_cv_binary.py` |
| View-stratified PE performance (Fig. performance e) | `classification/` | `run_full_suite.py` (per-view tasks), `analyze_full_run.py` |
| View classification analysis (Fig. view a–d) | `experiment/view/` | `view_raw_finetune_scaling.py`, `view_raw_finetune_cv.py`, `view_grouped_finetune_scaling.py`, `make_nature_*` |
| Interpretability (Fig. interpret a–c) | `experiment/interpretability/` + `experiment/roi_analysis/` | `extract_attributions.py`, `analyze_interpretability.py`, then `select_roi_cases.py` → `make_weak_context_masks.py` → `quantify_roi_attribution.py` → `run_roi_perturbation.py` → `run_report_semantics.py` → `summarize_for_paper.py` |

Representative commands:

```bash
# Main 5-fold PE classification (EchoPE: full fine-tune + residual head)
cd classification
python train_cv_binary.py --device cuda
python evaluate_binary_cv.py
python export_cv_curve_data.py

# Adaptation-strategy ablation (frozen / LoRA / partial-FT, +/- KD, +/- view-aux)
cd ../finetuning
python compare_distill_ablations.py --seeds 42 123 456 --epochs 20

# View classification (6-way data scaling + external 5-fold CV)
cd ../experiment/view
python view_raw_finetune_scaling.py --output-dir ./view_raw_finetune_results_v3 \
  --seeds 2024 2025 2026 --ratios 0.15 0.30 0.50 0.75 1.00 --epochs 40 --device cuda
python view_raw_finetune_cv.py --output-dir ./view_raw_finetune_cv_seed2026 \
  --method pretrained_full_ft --epochs 40 --device cuda

# Interpretability (attribution then ROI analysis)
cd ../interpretability
python extract_attributions.py --checkpoint ../../classification/outputs/seed-2024/pooled/best_checkpoint.pt --save-figures
python analyze_interpretability.py --checkpoint ../../classification/outputs/seed-2024/pooled/best_checkpoint.pt
```

All result artifacts (checkpoints, per-fold predictions, summary CSVs, figures)
are produced by these scripts and are intentionally **not** committed; only the
code is versioned here.

---

## Code & Data availability

- **Code.** This repository is shared with the editors and reviewers for
  evaluation during peer review. A public release with a permanent DOI will be
  made available upon acceptance.
- **Data.** The HOPE dataset contains de-identified emergency-department clinical
  ultrasound videos and cannot be released publicly. It is available from the
  corresponding author on reasonable request, subject to institutional review
  board approval and a data-use agreement.
- **Pretrained model.** EchoPE is initialized from the publicly released EchoPrime
  model; its weights are not redistributed here.

## Acknowledgements & citation

This work uses the publicly released **EchoPrime** pretrained weights. Both the
EchoPE echo encoder (MViT-v2-s video encoder) and the view encoder (ConvNeXt
view classifier) are initialized from EchoPrime's pretrained checkpoints, and the
phenotype-probing and semantic-probing analyses rely on the released EchoPrime
model. We gratefully acknowledge the EchoPrime authors for releasing these
pretrained weights and code.

EchoPrime: Vukadinovic et al., *A Multi-Video View-Informed Vision-Language Model
for Comprehensive Echocardiography Interpretation*, arXiv:2410.09704,
https://github.com/echonet/EchoPrime.

If you use this code, please cite the EchoPE manuscript (under review at *npj
Digital Medicine*) and the EchoPrime paper.

## License

Released under the MIT License (see [LICENSE](LICENSE)). EchoPrime code and
weights remain under their original license.
