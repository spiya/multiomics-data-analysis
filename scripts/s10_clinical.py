#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 10 — Clinical associations (+ optional survival)

Usage:
  python scripts/s10_clinical.py                 # uses outdir from config/paths.yaml
  python scripts/s10_clinical.py --outdir <RUN>  # override outdir
  python scripts/s10_clinical.py --config config/paths.yaml --params config/params.yaml

Inputs (under <outdir>):
  aligned/clinical.aligned.csv     (rows = samples)
  tables/SNF_clusters.csv          (first col = cluster labels from Step 9)

Outputs (under <outdir>):
  tables/clinical_with_clusters.csv
  tables/clinical_assoc_categorical.csv
  tables/clinical_assoc_numeric.csv
  figures/clinical_numeric_boxplots.png  (if matplotlib available)
  (if lifelines available and OS/PFS columns exist)
    tables/cox_OS.csv, figures/km_OS.png, tables/logrank_OS.csv
    tables/cox_PFS.csv, figures/km_PFS.png, tables/logrank_PFS.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Optional deps
try:
    import scipy.stats as st
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Use non-interactive backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False

import yaml

# -----------------------------
# Small utilities
# -----------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for a 1D array of p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0, 1)

def is_categorical(series: pd.Series, max_levels: int = 10) -> bool:
    """Heuristic: object/category dtype OR few unique numeric levels."""
    if str(series.dtype) in ("object", "category", "bool"):
        return True
    # treat small-cardinality integers as categorical (e.g., stage coded as 1..4)
    nun = series.dropna().nunique()
    return (nun > 0) and (nun <= max_levels) and pd.api.types.is_integer_dtype(series)

def safe_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.dropna().shape[0] > 0

def ensure_dirs(outdir: Path):
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
def load_inputs(outdir: Path):
    clin_path = outdir / "aligned" / "clinical.aligned.csv"
    clu_path  = outdir / "tables"  / "SNF_clusters.csv"
    if not clin_path.exists():
        raise SystemExit(f"Missing clinical file: {clin_path}")
    if not clu_path.exists():
        raise SystemExit(f"Missing clusters file (Step 9 output): {clu_path}")

    clin = pd.read_csv(clin_path, index_col=0)
    clusters = pd.read_csv(clu_path, index_col=0).iloc[:, 0]
    # align indices
    common = sorted(set(clin.index) & set(clusters.index))
    if len(common) == 0:
        raise SystemExit("No overlapping samples between clinical.aligned.csv and SNF_clusters.csv")
    clin = clin.loc[common].copy()
    clusters = clusters.loc[common].astype(str)
    # merge for convenience
    merged = clin.copy()
    merged["cluster"] = clusters.values
    return clin, clusters, merged

# -----------------------------
# Association tests
# -----------------------------
def assoc_categorical(merged: pd.DataFrame, max_levels: int = 10) -> pd.DataFrame:
    """Chi-square test of independence between each categorical clinical feature and clusters."""
    rows = []
    for col in merged.columns:
        if col == "cluster":
            continue
        s = merged[col]
        if not is_categorical(s, max_levels=max_levels):
            continue
        # contingency table
        tab = pd.crosstab(merged["cluster"], s)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        if HAVE_SCIPY:
            try:
                chi2, p, dof, _ = st.chi2_contingency(tab)
            except Exception:
                chi2, p, dof = np.nan, 1.0, np.nan
        else:
            chi2, p, dof = np.nan, np.nan, np.nan
        rows.append({"feature": col, "test": "chi2", "chi2": chi2, "dof": dof, "pval": p,
                     "levels": int(s.dropna().nunique())})
    if not rows:
        return pd.DataFrame(columns=["feature", "test", "chi2", "dof", "pval", "FDR", "levels"])
    df = pd.DataFrame(rows)
    df["FDR"] = bh_fdr(df["pval"].fillna(1.0).values)
    return df.sort_values("pval")

def assoc_numeric(merged: pd.DataFrame, min_per_group: int = 5) -> pd.DataFrame:
    """Kruskal-Wallis (nonparametric) across clusters for numeric features; includes per-cluster means."""
    rows = []
    clusters = merged["cluster"].unique()
    for col in merged.columns:
        if col == "cluster":
            continue
        s = merged[col]
        if not safe_numeric(s):
            continue
        # split by cluster
        groups = [merged.loc[merged["cluster"] == g, col].dropna().values for g in clusters]
        # require enough groups with samples
        valid = [g for g in groups if g.size >= min_per_group]
        if len(valid) < 2:
            continue
        if HAVE_SCIPY:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    stat, p = st.kruskal(*groups, nan_policy="omit")
                except Exception:
                    stat, p = np.nan, 1.0
        else:
            stat, p = np.nan, np.nan
        # per-cluster means (for context)
        means = {f"mean_{g}": float(np.nanmean(merged.loc[merged['cluster'] == g, col].values))
                 for g in clusters}
        rows.append({"feature": col, "test": "kruskal", "H": stat, "pval": p, **means})
    if not rows:
        base_cols = ["feature", "test", "H", "pval", "FDR"]
        return pd.DataFrame(columns=base_cols)
    df = pd.DataFrame(rows)
    df["FDR"] = bh_fdr(df["pval"].fillna(1.0).values)
    return df.sort_values("pval")

def plot_numeric_boxplots(merged: pd.DataFrame, numeric_cols: list[str], out_png: Path):
    if not HAVE_MPL or not numeric_cols:
        return
    # Build a grid of small multiples (up to 12 features to keep readable)
    cols = numeric_cols[:12]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        data = [merged.loc[merged["cluster"] == g, col].values for g in sorted(merged["cluster"].unique())]
        ax.boxplot(data, labels=sorted(merged["cluster"].unique()), showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("cluster"); ax.set_ylabel(col)
    # hide any empty axes
    for j in range(i+1, nrows*ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# -----------------------------
# Survival (optional)
# -----------------------------
def run_survival(outdir: Path, merged: pd.DataFrame, time_col: str, event_col: str, label: str):
    """Cox by cluster (categorical) + KM plot + multivariate logrank test."""
    if not HAVE_LIFELINES:
        print(f"[Survival] lifelines not installed; skipping {label}.")
        return

    if time_col not in merged.columns or event_col not in merged.columns:
        print(f"[Survival] Missing columns for {label}: {time_col}/{event_col}; skipping.")
        return

    df = merged[["cluster", time_col, event_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # ensure numeric types
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna()
    if df.shape[0] < 30:
        print(f"[Survival] Too few rows after cleaning for {label}: n={df.shape[0]}; skipping.")
        return

    # One-hot clusters for Cox; drop one to avoid collinearity
    X = pd.get_dummies(df[["cluster"]], prefix="cl", drop_first=True)
    X[time_col] = df[time_col].values
    X[event_col] = df[event_col].values

    # Fit CoxPH
    try:
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
        cph.fit(X, duration_col=time_col, event_col=event_col, show_progress=False)
        cox_summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
        cox_summary.to_csv(outdir / "tables" / f"cox_{label}.csv", index=False)
        print(f"[Survival] Wrote tables/cox_{label}.csv")
    except Exception as e:
        print(f"[Survival] Cox fit failed for {label}: {e}")
        cox_summary = None

    # Multivariate logrank (clusters as groups)
    try:
        res = multivariate_logrank_test(df[time_col], df["cluster"], df[event_col])
        pd.DataFrame({
            "test_statistic": [res.test_statistic_],
            "p_value": [res.p_value_],
            "df": [res.degrees_freedom_],
        }).to_csv(outdir / "tables" / f"logrank_{label}.csv", index=False)
        print(f"[Survival] Wrote tables/logrank_{label}.csv")
    except Exception as e:
        print(f"[Survival] Logrank failed for {label}: {e}")

    # KM plot by cluster
    try:
        km = KaplanMeierFitter()
        plt.figure(figsize=(6, 4.5))
        for cl in sorted(df["cluster"].unique()):
            mask = (df["cluster"] == cl)
            km.fit(df.loc[mask, time_col], df.loc[mask, event_col], label=f"cl {cl}")
            km.plot(ci_show=False)
        plt.title(f"KM: {label}")
        plt.xlabel("time"); plt.ylabel("survival probability")
        plt.tight_layout()
        plt.savefig(outdir / "figures" / f"km_{label}.png", dpi=200)
        plt.close()
        print(f"[Survival] Wrote figures/km_{label}.png")
    except Exception as e:
        print(f"[Survival] KM plotting failed for {label}: {e}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--params", default="config/params.yaml")  # not required, kept for symmetry
    ap.add_argument("--outdir", default=None, help="Override outdir; if omitted, read from paths.yaml")
    ap.add_argument("--max_levels", type=int, default=10, help="Max unique levels to treat numeric as categorical")
    args = ap.parse_args()

    # Resolve outdir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        paths = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        outdir = Path(paths["outdir"])

    ensure_dirs(outdir)

    # Load & merge
    clin, clusters, merged = load_inputs(outdir)

    # Save merged convenience table
    merged.to_csv(outdir / "tables" / "clinical_with_clusters.csv")
    print(f"[OK] Wrote tables/clinical_with_clusters.csv (n={merged.shape[0]})")

    # Categorical associations
    cat_df = assoc_categorical(merged, max_levels=args.max_levels)
    cat_df.to_csv(outdir / "tables" / "clinical_assoc_categorical.csv", index=False)
    print(f"[OK] Wrote tables/clinical_assoc_categorical.csv (rows={cat_df.shape[0]})")

    # Numeric associations
    num_df = assoc_numeric(merged)
    num_df.to_csv(outdir / "tables" / "clinical_assoc_numeric.csv", index=False)
    print(f"[OK] Wrote tables/clinical_assoc_numeric.csv (rows={num_df.shape[0]})")

    # Quick boxplots for the top numeric hits (optional)
    if HAVE_MPL and not num_df.empty:
        top_numeric = [r["feature"] for _, r in num_df.head(9).iterrows()]
        plot_numeric_boxplots(merged, top_numeric, outdir / "figures" / "clinical_numeric_boxplots.png")
        print("[OK] Wrote figures/clinical_numeric_boxplots.png")

    # Survival (optional): run for any of these pairs that exist
    survival_pairs = [
        ("OS_time",  "OS_event",  "OS"),
        ("PFS_time", "PFS_event", "PFS"),
        ("DSS_time", "DSS_event", "DSS"),   # if present
        ("DFI_time", "DFI_event", "DFI"),   # if present
    ]
    for tcol, ecol, label in survival_pairs:
        run_survival(outdir, merged, tcol, ecol, label)

    print("Step 10 clinical associations complete.")

if __name__ == "__main__":
    # Quiet some scipy/stat warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
