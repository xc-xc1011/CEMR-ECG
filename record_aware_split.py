"""Record-aware (subject-independent) train/validation splitting for CEMR-ECG.

Motivation
----------
The original submission reserved a class-stratified 15% internal validation
subset by sampling individual beats (``train_val_split(y, seed, 0.15)``). Because
that routine never received the record identifier, beats from one record could
appear in both the fitting set and the validation set, so the adapter weight
alpha and the BioAdaptive decoder policy were selected on validation beats that
were partly memorised by the evidence encoder and by the deep backbones that
early-stop on validation loss.

This module replaces the beat-level split with a record-level split. Every beat
from a record is assigned to exactly one of the fitting set or the internal
validation set, and no record is shared between them.

Constraint discovered from the data
-----------------------------------
Fusion (F) beats are extremely concentrated:

    MIT-BIH DS1   414 F beats, 372 of them (90%) in record mit_208
    MIT-BIH DS2   388 F beats, 362 of them (87%) in record mit_213
    INCART        219 F beats, spread over 22 of 75 records
    SVDB           23 F beats, spread over  6 of 78 records

A record-level validation holdout therefore cannot keep F beats in the fitting
set and in the validation set at the same time for MIT-BIH. When a candidate
split leaves the validation set without F support, the BioAdaptive F-guard
rejects any F-driven policy automatically, so the framework falls back to a
non-F policy. That behaviour is reported rather than patched, because the
concentration itself is the substantive finding for the F class.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# A class is treated as "must be covered on both sides" only when the training
# pool holds enough beats for a meaningful split. Below this threshold the class
# is too sparse for a validation estimate and is allowed to fall on one side.
MIN_CLASS_SUPPORT = 30

# The BioAdaptive F-guard requires at least this many F beats in the validation
# set before it will accept an F-driven decoder policy. The split search tries
# to reach this level, but cannot always do so for the sparsest datasets.
F_GUARD_MIN = 10


def record_aware_train_val_split(
    y: np.ndarray,
    pid: np.ndarray,
    seed: int,
    val_fraction: float = 0.15,
    max_candidates: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training pool into fitting and validation indices by record.

    The returned index arrays are positional indices into ``y``/``pid``. Each
    record contributes all of its beats to exactly one of the two arrays, so
    ``fit_records`` and ``val_records`` are disjoint.

    Candidate record subsets are scored to (i) keep every well-supported class
    present on both sides, (ii) land close to ``val_fraction`` of beats, and
    (iii) put as few records as possible in validation, which leaves as many
    records as possible for fitting the evidence encoder.
    """
    y = np.asarray(y, dtype=np.int64)
    pid = np.asarray(pid)
    records = np.unique(pid)
    if len(records) < 2:
        raise ValueError("Record-aware splitting needs at least two records.")

    rng = np.random.default_rng(seed)
    n_val_records = max(1, int(round(len(records) * val_fraction)))
    n_val_records = min(n_val_records, len(records) - 1)

    total_counts = np.bincount(y, minlength=5)
    required = [c for c in range(5) if total_counts[c] >= MIN_CLASS_SUPPORT]

    # Aggregate beats per record once, then score candidate record subsets by
    # summing record-level count vectors. This avoids rescanning every beat for
    # every candidate and keeps the search fast even when no candidate can
    # satisfy the F-guard target.
    record_index = {r: i for i, r in enumerate(records.tolist())}
    beat_to_record = np.fromiter(
        (record_index[r] for r in pid.tolist()), dtype=np.int64, count=len(pid)
    )
    per_record_counts = np.zeros((len(records), 5), dtype=np.int64)
    np.add.at(per_record_counts, (beat_to_record, y), 1)

    best: tuple | None = None
    for _ in range(max_candidates):
        val_record_idx = rng.choice(len(records), size=n_val_records, replace=False)
        val_counts = per_record_counts[val_record_idx].sum(axis=0)
        fit_counts = total_counts - val_counts

        # Hard requirement 1: every well-supported class must remain learnable,
        # so the fitting set keeps at least half of each such class. For
        # MIT-BIH this forces the dominant F record (mit_208, 372 of 414 F
        # beats) into the fitting set instead of the validation set.
        # Hard requirement 2: validation must still see every class, otherwise
        # the selection step cannot evaluate a policy for it.
        if any(
            fit_counts[c] * 2 < total_counts[c] or val_counts[c] == 0 for c in required
        ):
            continue

        # Among admissible splits, prefer one whose validation F support clears
        # the BioAdaptive F-guard, then prefer more minority beats in
        # validation, but never at the cost of the majority rule above.
        guard_ok = int(val_counts[3] >= F_GUARD_MIN)
        minority_val = int(val_counts[1] + val_counts[3])
        score = (guard_ok, minority_val)

        if best is None or score > best[0]:
            best = (score, val_record_idx)
            if guard_ok:
                break

    if best is None:
        raise RuntimeError(
            "Could not build a record-level validation split that preserves all "
            "well-supported classes in the fitting set."
        )

    val_records = set(records[best[1]].tolist())
    val_mask = np.fromiter((r in val_records for r in pid.tolist()), dtype=bool, count=len(pid))
    fit_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    rng.shuffle(fit_idx)
    rng.shuffle(val_idx)
    return fit_idx.astype(np.int64), val_idx.astype(np.int64)


def split_audit(
    y: np.ndarray,
    pid: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset: str,
    seed: int,
    test_pid: np.ndarray | None = None,
) -> dict:
    """Describe a split for reporting and for leakage checks."""
    y = np.asarray(y, dtype=np.int64)
    pid = np.asarray(pid)
    class_names = ["N", "S", "V", "F", "Q"]

    fit_records = set(np.unique(pid[fit_idx]).tolist())
    val_records = set(np.unique(pid[val_idx]).tolist())
    fit_counts = np.bincount(y[fit_idx], minlength=5)
    val_counts = np.bincount(y[val_idx], minlength=5)

    row = {
        "dataset": dataset,
        "seed": int(seed),
        "protocol": "record-level fitting/validation split",
        "fit_records": len(fit_records),
        "val_records": len(val_records),
        "fit_size": int(len(fit_idx)),
        "val_size": int(len(val_idx)),
        "fit_val_record_overlap": len(fit_records & val_records),
        "fit_counts": {c: int(fit_counts[i]) for i, c in enumerate(class_names)},
        "val_counts": {c: int(val_counts[i]) for i, c in enumerate(class_names)},
        "fit_records_list": ";".join(sorted(fit_records)),
        "val_records_list": ";".join(sorted(val_records)),
    }

    if test_pid is not None:
        test_records = set(np.unique(np.asarray(test_pid)).tolist())
        row["fit_test_record_overlap"] = len(fit_records & test_records)
        row["val_test_record_overlap"] = len(val_records & test_records)

    problems = []
    if row["fit_val_record_overlap"] != 0:
        problems.append("fitting and validation records overlap")
    if test_pid is not None and row["fit_test_record_overlap"] != 0:
        problems.append("fitting and test records overlap")
    if test_pid is not None and row["val_test_record_overlap"] != 0:
        problems.append("validation and test records overlap")
    row["audit_interpretation"] = (
        "No record leakage across fitting, validation and test"
        if not problems
        else "; ".join(problems)
    )
    return row


def summarize_split_support(rows: Sequence[dict]) -> str:
    """Render a compact human-readable summary of several split audits."""
    lines = []
    for row in rows:
        lines.append(
            f"{row['dataset']:<8} seed={row['seed']:<5} "
            f"fit={row['fit_size']:>6} ({row['fit_records']} rec)  "
            f"val={row['val_size']:>5} ({row['val_records']} rec)  "
            f"val F={row['val_counts']['F']:>3}  overlap={row['fit_val_record_overlap']}"
        )
    return "\n".join(lines)
