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

This repository contains **only the experiment code** (the contents of
`EchoPrime/experiments`). The EchoPrime base model (`echo_prime/`, `utils/`,
`assets/`, `model_data/`) is **not** vendored here — see *Installation*.

```
.
├── echo_paths.py                 # Resolves the EchoPrime root and sets the working dir for imports
├── utils/split_cv.py             # Builds stratified 5-fold CV splits from the dataset manifest
├── full_run/                     # Main PE binary classification (EchoPE vs EchoPrime+head, 5-fold CV)
├── lora_run/                     # Fine-tuning strategies: LoRA, KD, partial-FT, view-aux ablations
├── view_run/                     # Echocardiographic view classification (6-/4-/3-way label spaces)
├── interpretable_run/            # Stage-1 interpretability: attention / grad / temporal / embeddings
├── interpretable_roi_run/        # Stage-2 ROI attribution, perturbation, semantic probing
├── text_run/                     # EchoPrime phenotype probing (zero-shot transfer baseline)
├── pe_run/                       # Lightweight cached-feature PE classifier prototype
└── simple_run/                   # Batch EchoPrime inference helper over a video folder
```

### `full_run/` — main PE binary classification
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

### `lora_run/` — fine-tuning strategies (adaptation ablation)
- `train_lora.py` — pure LoRA fine-tuning of the MViT encoder.
- `train_lora_distill.py` — full entry point for LoRA + knowledge distillation (KD) + view auxiliary supervision.
- `distill.py` — KD losses (`cos` / `l2` / `rkd-d` / `rkd-a` / `combo`) and frozen-teacher loading.
- `aux_heads.py` — view auxiliary heads (A4/PSS CE head and 11-class view-KD head).
- `compare_distill_ablations.py` — 7-configuration (frozen / LoRA / partial-FT, ± KD, ± view-aux) ablation runner.
- `compare_with_baseline.py` — multi-seed LoRA vs frozen-feature baseline comparison.

### `view_run/` — view classification
- `evaluate_view_classifier.py` — evaluate the frozen EchoPrime view head, mapped to the local label groups.
- `view_raw_finetune_scaling.py` — 6-way data-scaling experiment (direct / linear-probe / partial-FT / full-FT / scratch).
- `view_raw_finetune_cv.py` — 6-way external 5-fold CV runner.
- `view_grouped_finetune_scaling.py` — 4-way (`coarse4`) and 3-way (`family3`) grouped-label experiments.
- `view_finetune_data_scaling.py` — earlier 6-way data-scaling experiment (kept for provenance).
- `analyze_view_raw_finetune_scaling.py`, `analyze_view_grouped_finetune_scaling.py` — summary CSVs and plots.
- `make_nature_label_space_comparison.py` — label-space comparison figures (6-/4-/3-way).
- `make_nature_embedding_granularity.py` — penultimate-feature UMAP/t-SNE hidden-space figure.
- `view_raw_finetune_results_v3/analysis/make_nature_plots.py` — figure-generation helper for the 6-way result set.

### `interpretable_run/` — stage-1 interpretability
- `load_interpret_model.py` — restore a `full_run` checkpoint / manifest / dataloader.
- `extract_attributions.py` — extract embeddings, attention, gradient×input, and temporal occlusion curves.
- `analyze_interpretability.py` — view-level and representation-level aggregation (PCA, probes, retrieval).
- `compare_interpret_runs.py` — compare frozen-head vs full-fine-tune interpretability outputs.
- `run_interpret_suite.py` — multi-seed / multi-task batch runner.
- `make_nature_case_figure.py` — per-case interpretability figure.
- `temporal_attention_head.py` — optional lightweight temporal-pooling head (stage-2 reserved branch).

### `interpretable_roi_run/` — stage-2 ROI / perturbation / semantics
- `select_roi_cases.py` — select a shared case panel across models.
- `make_weak_context_masks.py` — generate weak acquisition-context masks (foreground / background / sector border / probe near-field).
- `quantify_roi_attribution.py` — ROI mass, enrichment, and top-10% overlap.
- `run_roi_perturbation.py` — ROI occlusion / perturbation effect on PE probability and logit.
- `run_report_semantics.py` — EchoPrime semantic probing (cosine drift, phenotype Spearman).
- `compare_roi_runs.py` — cross-model ROI comparison.
- `summarize_for_paper.py` — paper-ready tables and compact figures.
- `roi_utils.py`, `roi_annotation_schema.json` — ROI utilities and the annotation schema.

### `text_run/` — EchoPrime phenotype probing baseline
- `run_echoprime_pe_phenotypes.py` — per-video EchoPrime `predict_metrics()` phenotype predictions.
- `analyze_pe_phenotype_correlations.py` — focus-phenotype statistics and plot-ready data.
- `plot_pe_phenotype_figure.py` — Normal-vs-PE phenotype figure from saved analysis.

### `pe_run/` and `simple_run/`
- `pe_run/train_pe_classifier.py`, `pe_run/compare_heads.py` — early cached-feature PE classifier prototype and head comparison.
- `simple_run/run_testset.py` — batch EchoPrime inference / report helper over a video folder.

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
- Five-fold CV splits live under
  `dataset/preprocessed/all_dataset_preprocessed_cv_seed2026/fold_{0..4}/{train,val,test}.csv`
  (seed 2026, stratified by `label + parsed_view`).

The HOPE dataset is restricted clinical data and is **not** distributed with this
repository.

---

## Installation

The experiment scripts import the EchoPrime model code and read `model_data/`
from the EchoPrime repository root. `echo_paths.py` assumes this repository lives
at `EchoPrime/experiments`:

```python
ECHO_ROOT = Path(__file__).resolve().parent.parent   # -> the EchoPrime repo root
```

Set up as follows:

```bash
# 1) Clone the official EchoPrime repository.
git clone https://github.com/echonet/EchoPrime
cd EchoPrime

# 2) Download and unpack the EchoPrime model data (weights + assets).
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p1.pt
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/candidate_embeddings_p2.pt
unzip model_data.zip
mv candidate_embeddings_p1.pt model_data/candidates_data/
mv candidate_embeddings_p2.pt model_data/candidates_data/

# 3) Place THIS repository as the EchoPrime experiments directory.
git clone git@github.com:skylynf/EchoPE.git experiments

# 4) Create an environment and install dependencies.
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r experiments/requirements.txt
pip install -r requirements.txt   # EchoPrime base requirements
```

Run scripts from inside their run directory, e.g.:

```bash
cd experiments/full_run
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

Experiment directory to manuscript Results section mapping.

| Results section / figure | Directory | Key scripts |
| --- | --- | --- |
| Zero-shot EchoPrime phenotype transfer (Fig. performance a) | `text_run/` | `run_echoprime_pe_phenotypes.py` → `analyze_pe_phenotype_correlations.py` → `plot_pe_phenotype_figure.py` |
| EchoPE vs EchoPrime+head, 5-fold ROC/PR (Fig. performance b) | `full_run/` | `train_cv_binary.py`, `evaluate_full_run.py`, `evaluate_binary_cv.py`, `export_cv_curve_data.py` |
| Fine-tuning-strategy ablation (Fig. performance c, d) | `lora_run/` + `full_run/` | `compare_distill_ablations.py`, `train_lora_distill.py`, `train_cv_binary.py` |
| View-stratified PE performance (Fig. performance e) | `full_run/` | `run_full_suite.py` (per-view tasks), `analyze_full_run.py` |
| View classification analysis (Fig. view a–d) | `view_run/` | `view_raw_finetune_scaling.py`, `view_raw_finetune_cv.py`, `view_grouped_finetune_scaling.py`, `make_nature_*` |
| Interpretability (Fig. interpret a–c) | `interpretable_run/` + `interpretable_roi_run/` | `extract_attributions.py`, `analyze_interpretability.py`, then `select_roi_cases.py` → `make_weak_context_masks.py` → `quantify_roi_attribution.py` → `run_roi_perturbation.py` → `run_report_semantics.py` → `summarize_for_paper.py` |

Representative commands:

```bash
# Main 5-fold PE classification (EchoPE: full fine-tune + residual head)
cd experiments/full_run
python train_cv_binary.py --device cuda
python evaluate_binary_cv.py
python export_cv_curve_data.py

# Adaptation-strategy ablation (frozen / LoRA / partial-FT, +/- KD, +/- view-aux)
cd ../lora_run
python compare_distill_ablations.py --seeds 42 123 456 --epochs 20

# View classification (6-way data scaling + external 5-fold CV)
cd ../view_run
python view_raw_finetune_scaling.py --output-dir ./view_raw_finetune_results_v3 \
  --seeds 2024 2025 2026 --ratios 0.15 0.30 0.50 0.75 1.00 --epochs 40 --device cuda
python view_raw_finetune_cv.py --output-dir ./view_raw_finetune_cv_seed2026 \
  --method pretrained_full_ft --epochs 40 --device cuda

# Interpretability (stage 1 then stage 2 ROI)
cd ../interpretable_run
python extract_attributions.py --checkpoint ../full_run/outputs/seed-2024/pooled/best_checkpoint.pt --save-figures
python analyze_interpretability.py --checkpoint ../full_run/outputs/seed-2024/pooled/best_checkpoint.pt
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
- **Pretrained model.** EchoPE is initialized from EchoPrime; its weights and
  `model_data` are obtained from the official EchoPrime release and are not
  redistributed here.

## Acknowledgements & citation

EchoPE builds on **EchoPrime** (Vukadinovic et al., *A Multi-Video View-Informed
Vision-Language Model for Comprehensive Echocardiography Interpretation*,
https://github.com/echonet/EchoPrime, arXiv:2410.09704). We thank the EchoPrime
authors for releasing their model and code.

If you use this code, please cite the EchoPE manuscript (under review at *npj
Digital Medicine*) and the EchoPrime paper.

## License

Released under the MIT License (see [LICENSE](LICENSE)). EchoPrime code and
weights remain under their original license.
