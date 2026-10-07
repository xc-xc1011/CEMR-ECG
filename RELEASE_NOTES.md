# Release Notes

## Record-level validation protocol (revision)

This update aligns the released code with the revised manuscript.

- Added `record_aware_split.py`: record-level (subject-independent) fitting/validation splitting, plus a record-overlap audit.
- The dataset-wise framework runner now exposes `--val-protocol {record,beat}`. The record-level protocol is the one behind the reported results; `beat` reproduces the archived beat-level sensitivity analysis and remains the script default for backward compatibility.
- Added `--write-split-audit` to emit the per-dataset--seed fitting/validation/test record-overlap table.
- Added `run_hierarchical_statistics.py`: dependence-aware mixed model (backbone random intercept), clustered bootstraps and effective sample size for the paired M-F1(4) gain.
- Added `generate_calibration_curves.py`: reliability curves before and after adaptation (embedded as TrueType, not Type 3).
- Updated `feature_engineering.py` and added `statsmodels` to `requirements.txt`.

## Reproducibility Code Release

This release provides the experiment code used to reproduce the CEMR-ECG framework evaluations.

Included components:

- CEMR-ECG evidence encoder and BioAdaptive decoder runners.
- Dataset-wise multi-method experiment runner.
- Machine-learning and deep/time-series baseline interfaces.
- Representative probability-quality and robustness diagnostic runner.
- Data preparation helper for PhysioNet-derived feature caches.
- Optional scripts for regenerating manuscript-style figures after results have been produced.

Generated experiment outputs are written by the scripts under `results/`. Large local data, caches and generated artifacts are ignored by default so the repository stays focused on reproducible code.

Before public release, review:

- Author name and license choice in `LICENSE`.
- Citation text in `README.md`.
- Whether to add a Zenodo DOI after repository archival.

