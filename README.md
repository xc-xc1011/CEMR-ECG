# CEMR-ECG

Reproducibility code for **CEMR-ECG**, a classifier-agnostic morphology-rhythm evidence framework for imbalanced ECG heartbeat classification.

CEMR-ECG builds named ECG evidence from each heartbeat, including morphology, RR-rhythm, lead-aware waveform, derivative and class-prototype descriptors. The evidence is encoded once, combined with each backbone probability interface, and passed through a bounded BioAdaptive decoder selected from training/validation information.

## Repository Layout

```text
.
|-- config.py
|-- feature_engineering.py
|-- metrics_eval.py
|-- optimize_external_datasets.py
|-- run_datasetwise_multimethod_permethod_bioadaptive_framework.py
|-- run_representative_supplementary_audits.py
|-- run_modern_deep_baselines.py
|-- run_external_raw_baselines.py
|-- run_cemr_bio_evidence_core.py
|-- run_cemr_bio_adaptive_decoder.py
|-- build_outlier_removed_cemr_summary.py
|-- generate_cemr_ecg_bspc_figures.R
|-- generate_cemr_ecg_python_schematics.py
|-- generate_representative_supplementary_figures.py
|-- src/
|   |-- download_data.py
|-- requirements.txt
|-- RELEASE_NOTES.md
|-- LICENSE
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

For GPU deep-learning baselines, install a PyTorch build that matches your CUDA version. CPU execution is supported but slower.

Optional R packages for regenerating manuscript-style data figures:

```r
install.packages(c(
  "ggplot2", "patchwork", "svglite", "ragg", "dplyr", "tidyr",
  "readr", "scales", "ggrepel", "cowplot", "stringr", "purrr",
  "viridisLite"
))
```

## Data Preparation

The experiments use public ECG databases from PhysioNet:

- MIT-BIH Arrhythmia Database
- St Petersburg INCART 12-lead Arrhythmia Database
- MIT-BIH Supraventricular Arrhythmia Database

Place the downloaded records under:

```text
data/raw/mitdb/
data/raw/incartdb/
data/raw/svdb/
```

MIT-BIH records can be downloaded with:

```bash
python src/download_data.py
```

Build the shared preprocessed feature cache:

```bash
python optimize_external_datasets.py
```

This creates `results/external_base_features_cache.npz`, which is used by the dataset-wise framework experiments.

## Running CEMR-ECG Experiments

Quick smoke run:

```bash
python run_datasetwise_multimethod_permethod_bioadaptive_framework.py ^
  --datasets MIT-BIH ^
  --families ml ^
  --methods ExtraTrees_raw ^
  --seeds 303 ^
  --cpu
```

Run the machine-learning family:

```bash
python run_datasetwise_multimethod_permethod_bioadaptive_framework.py --families ml
```

Run the deep/time-series family:

```bash
python run_datasetwise_multimethod_permethod_bioadaptive_framework.py --families deep
```

Run all configured methods:

```bash
python run_datasetwise_multimethod_permethod_bioadaptive_framework.py
```

The main runner writes dataset-method detail tables, summary tables, confusion matrices and reports under `results/`.

## Representative Diagnostics

Run representative probability-quality, split-integrity and class-cost sensitivity diagnostics:

```bash
python run_representative_supplementary_audits.py
```

Smoke run:

```bash
python run_representative_supplementary_audits.py ^
  --datasets MIT-BIH ^
  --methods ExtraTrees_raw ^
  --seeds 303 ^
  --phase all ^
  --cpu ^
  --smoke
```

The diagnostic runner writes calibration, reliability-bin, split-audit and class-cost sensitivity outputs under `results/`.

## Regenerating Figures

The figure scripts can be run after the corresponding result files have been generated:

```bash
Rscript generate_cemr_ecg_bspc_figures.R
python generate_cemr_ecg_python_schematics.py
python generate_representative_supplementary_figures.py
```

## Method Notes

- CEMR-ECG is an evidence framework for public-database ECG heartbeat classification.
- The experiments use retrospective public ECG databases and AAMI-style heartbeat grouping.
- SVDB F-class results should be interpreted as a limited-support endpoint.
- Direct leaderboard comparison with studies using different label spaces, splits or endpoints is not appropriate.
- Probability-quality diagnostics are representative checks and do not establish formal calibration for every backbone.

## Citation

A formal citation can be added after manuscript acceptance or preprint release. For now:

```text
Xu C. CEMR-ECG: a reusable morphology-rhythm evidence framework for imbalanced ECG heartbeat classification. GitHub repository, 2026.
```

## License

Code is released under the MIT License. The PhysioNet datasets used by the experiments remain governed by their original database licenses.

