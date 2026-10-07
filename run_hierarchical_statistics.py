"""Dependence-aware statistics for the paired backbone comparison (reviewer 2, point 9).

The manuscript reports a paired-t and a sign test over 60 dataset-backbone
means. Those 60 observations are not independent: 20 backbones are evaluated on
the same dataset splits with largely the same test beats, and the same
backbones recur across the three databases. Reviewer 2 asked for a statistical
treatment that accounts for this hierarchical structure.

Three analyses are produced:

1. **Variance decomposition.** A linear mixed model
   ``gain ~ C(dataset) + (1|backbone)`` estimates dataset fixed effects and
   backbone and residual variance components. If a backbone
   component exists, the 60 means carry less information than their count
   suggests.

2. **Mixed-effects estimate of the mean gain**, averaged equally over the
   observed datasets, with an asymptotic Wald standard error and interval.

3. **Cluster-robust and cluster-bootstrap alternatives.** A backbone-clustered
   bootstrap and a dataset-clustered bootstrap resample whole clusters instead
   of individual means, which is the most transparent way to show how much the
   significance depends on the independence assumption.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DETAIL = ROOT / "results" / "datasetwise_multimethod_permethod_bioadaptive_valrec_combined_detail.csv"
DETAIL_BEAT = ROOT / "results" / "datasetwise_multimethod_permethod_bioadaptive_detail.csv"
OUT_SUMMARY = ROOT / "results" / "hierarchical_statistics_summary.csv"
OUT_VARIANCE = ROOT / "results" / "hierarchical_variance_components.csv"

N_BOOT = 10_000
BOOT_SEED = 20260928


def load_gains(path: Path, protocol_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "family" not in df.columns:
        raise ValueError(f"{path} lacks a 'family' column")
    means = (
        df.groupby(["dataset", "method"], as_index=False)["delta_macro_f1_4"].mean()
    )
    means["protocol"] = protocol_label
    return means


def variance_components(means: pd.DataFrame) -> dict:
    """Variance decomposition of the dataset-backbone gains.

    Datasets enter as fixed effects and backbones as a random intercept. Only
    three datasets exist, which is too few to estimate a dataset variance
    component: a crossed random-effects specification drives the dataset
    variance to the boundary and returns a zero or undefined standard error.
    Twenty backbones, by contrast, support a stable random-intercept estimate.

    The backbone ICC then yields a design effect, which is the honest way to
    express how much information the 60 means actually carry: the effective
    sample size is ``n / (1 + (m_bar - 1) * ICC)`` for a mean cluster size of
    ``m_bar``.
    """
    import statsmodels.formula.api as smf

    df = means.copy()
    df["gain"] = df["delta_macro_f1_4"]

    model = smf.mixedlm("gain ~ C(dataset)", df, groups=df["method"])
    fit = model.fit(reml=True)

    # Treatment coding makes the intercept the reference database (INCART),
    # not the overall gain. Average the fixed-effect design over datasets.
    design = pd.DataFrame(fit.model.exog, index=df.index)
    contrast = design.groupby(df["dataset"]).mean().mean().to_numpy()[None, :]
    marginal = fit.t_test(contrast)
    marginal_ci = marginal.conf_int()[0]

    backbone_var = float(np.asarray(fit.cov_re, dtype=float).ravel()[0])
    residual_var = float(fit.scale)
    total = backbone_var + residual_var
    icc = backbone_var / total if total else float("nan")
    m_bar = len(df) / max(1, df["method"].nunique())
    design_effect = 1 + (m_bar - 1) * icc if np.isfinite(icc) else float("nan")

    return {
        "n_obs": int(len(df)),
        "n_datasets": int(df["dataset"].nunique()),
        "n_backbones": int(df["method"].nunique()),
        "mean_gain": float(marginal.effect[0]),
        "se_mean_gain": float(marginal.sd[0, 0]),
        "p_mean_gain": float(marginal.pvalue),
        "mean_gain_ci_low": float(marginal_ci[0]),
        "mean_gain_ci_high": float(marginal_ci[1]),
        "reference_dataset_gain": float(fit.params["Intercept"]),
        "se_intercept": float(fit.bse["Intercept"]),
        "p_intercept": float(fit.pvalues["Intercept"]),
        "backbone_var": backbone_var,
        "residual_var": residual_var,
        "total_var": total,
        "icc_backbone": icc,
        "mean_cluster_size": m_bar,
        "design_effect": design_effect,
        "effective_n": len(df) / design_effect if np.isfinite(design_effect) else float("nan"),
    }


def cluster_bootstrap(means: pd.DataFrame, cluster_col: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Bootstrap the mean gain by resampling whole clusters."""
    rng = np.random.default_rng(seed)
    clusters = means[cluster_col].unique()
    grouped = {c: means.loc[means[cluster_col] == c, "delta_macro_f1_4"].to_numpy() for c in clusters}
    means_out = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([grouped[c] for c in drawn])
        means_out[i] = vals.mean()
    return {
        "cluster": cluster_col,
        "n_clusters": int(len(clusters)),
        "boot_mean": float(means_out.mean()),
        "ci_low": float(np.percentile(means_out, 2.5)),
        "ci_high": float(np.percentile(means_out, 97.5)),
        "frac_positive": float(np.mean(means_out > 0)),
    }


def naive_paired_t(means: pd.DataFrame) -> dict:
    """The submitted test, for comparison: paired-t on signed gains vs zero."""
    from scipy import stats

    g = means["delta_macro_f1_4"].to_numpy()
    t, p = stats.ttest_1samp(g, 0.0)
    return {
        "n": int(len(g)),
        "mean": float(g.mean()),
        "t": float(t),
        "p": float(p),
        "cohen_dz": float(g.mean() / g.std(ddof=1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", default=str(DEFAULT_DETAIL))
    parser.add_argument("--label", default="record-level validation")
    args = parser.parse_args()

    path = Path(args.detail)
    if not path.exists():
        print(f"missing {path}")
        return 1
    means = load_gains(path, args.label)
    print(f"Loaded {len(means)} dataset-backbone means from {path.name}")
    print(f"datasets={sorted(means['dataset'].unique())}")
    print(f"backbones={means['method'].nunique()}")

    print("\n=== submitted-style test (assumes 60 independent means) ===")
    naive = naive_paired_t(means)
    for k, v in naive.items():
        print(f"  {k}: {v:.6g}")

    print("\n=== variance decomposition (dataset fixed effects, backbone random intercept) ===")
    try:
        vc = variance_components(means)
        for k, v in vc.items():
            print(f"  {k}: {v:.6g}")
        pd.DataFrame([vc]).to_csv(OUT_VARIANCE, index=False, encoding="utf-8")
        print(f"  saved {OUT_VARIANCE}")
    except Exception as exc:  # noqa: BLE001
        print(f"  mixed model failed: {type(exc).__name__}: {exc}")
        vc = {}

    print("\n=== cluster bootstrap (resamples whole clusters) ===")
    rows = []
    for cluster_col in ["dataset", "method"]:
        res = cluster_bootstrap(means, cluster_col)
        rows.append(res)
        print(
            f"  by {cluster_col:<8} n_clusters={res['n_clusters']:>3} "
            f"mean={res['boot_mean'] * 100:+6.2f} pp "
            f"95% CI [{res['ci_low'] * 100:+6.2f}, {res['ci_high'] * 100:+6.2f}] "
            f"P(>0)={res['frac_positive']:.4f}"
        )

    boot = pd.DataFrame(rows)
    boot["protocol"] = args.label
    boot["naive_t_p"] = naive["p"]
    boot["naive_cohen_dz"] = naive["cohen_dz"]
    boot["n_means"] = naive["n"]
    if vc:
        boot["icc_backbone"] = vc["icc_backbone"]
        boot["design_effect"] = vc["design_effect"]
        boot["effective_n"] = vc["effective_n"]
        boot["mixed_mean_gain"] = vc["mean_gain"]
        boot["mixed_se"] = vc["se_mean_gain"]
        boot["mixed_p"] = vc["p_mean_gain"]
        boot["mixed_ci_low"] = vc["mean_gain_ci_low"]
        boot["mixed_ci_high"] = vc["mean_gain_ci_high"]
    boot.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
