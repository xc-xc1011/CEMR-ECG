# Release Notes

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

