#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s01_07_prep.py (v3, robust imports)
-----------------------------------
Preprocess Steps 1–7 with safer imports and alignment:
  - Reliable import of funct.{preprocess, io, snf_utils} when funct lives under scripts/funct
  - TCGA ID normalization: align.id_level = tcga_sample | tcga_patient | none
  - Iteratively drop worst-overlap modality until align.min_samples is met (default 30)
  - Empty-safe impute/z-score; variance filters; mutation prevalence filtering
  - Saves processed matrices, optional affinities, and a manifest

Run:
  python scripts\\s01_07_prep_v3.py --config config\\paths.yaml --params config\\params.yaml
"""

import argparse, json, math, sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------
# Robust import helpers: prefer normal package import; otherwise load by path
# ---------------------------------------------------------------------
def _import_functdule(module_name: str):
    """
    Try: from funct import <module_name>
    Fallback: load from file next to this script: scripts/funct/<module_name>.py
    Also tries one level up if needed.
    """
    # 1) normal import if funct is on sys.path and is a package
    try:
        mod = __import__(f"funct.{module_name}", fromlist=[module_name])
        return mod
    except Exception:
        pass

    # 2) fallbacks by file
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "funct" / f"{module_name}.py",
        script_dir.parent / "funct" / f"{module_name}.py",  # in case you move the script
    ]
    for cand in candidates:
        if cand.exists():
            spec = importlib.util.spec_from_file_location(f"funct_{module_name}", str(cand))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod

    raise SystemExit(
        f"Could not import funct.{module_name}. "
        f"Ensure 'scripts/funct/{module_name}.py' exists and 'scripts/funct/__init__.py' is present."
    )

pp = _import_functdule("preprocess")
io = _import_functdule("io")
snf_utils = _import_functdule("snf_utils")

# ---------------------------------------------------------------------
# ID normalization helpers
# ---------------------------------------------------------------------
def norm_tcga(ids: pd.Index, level: str = "tcga_sample") -> pd.Index:
    s = pd.Index([str(x) for x in ids])
    s = s.str.replace(r"[.]", "-", regex=True).str.upper().str.strip()
    if level == "tcga_patient":
        # TCGA-XX-YYYY
        s = s.str.replace(r"^([A-Z0-9]{4}-[A-Z0-9]{2}-[A-Z0-9]{4}).*$", r"\1", regex=True)
    elif level == "tcga_sample":
        # TCGA-XX-YYYY-ZZ (first 4 tokens)
        s = s.str.replace(r"^((?:[A-Z0-9]{4}-){3}[A-Z0-9]{2}).*$", r"\1", regex=True)
    return s

def apply_id_norm_cols(df: pd.DataFrame, level: str) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df.columns = norm_tcga(df.columns, level=level)
    if df.columns.duplicated().any():
        df = df.groupby(axis=1, level=0).mean()
    return df

def apply_id_norm_index(df: pd.DataFrame, level: str) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df.index = norm_tcga(df.index, level=level)
    if df.index.duplicated().any():
        df = df.groupby(level=0).first()
    return df

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _filter_mut(mut_df: pd.DataFrame | None, min_prev=0.01, min_samples=None) -> pd.DataFrame | None:
    if mut_df is None:
        return None
    X = pp.binarize_mut(mut_df)
    n = X.shape[1]
    if n == 0:
        return X
    thresh = max(int(min_samples) if min_samples is not None else int(math.ceil(float(min_prev)*n)), 1)
    keep = (X.sum(axis=1) >= thresh)
    X = X.loc[keep]
    X = X.loc[X.sum(axis=1) > 0]
    return X

def _variance(df: pd.DataFrame | None, q: float) -> pd.DataFrame | None:
    if df is None:
        return None
    return pp.variance_filter(df, q=float(q))

def _impute_then_z(df: pd.DataFrame | None, log2=False) -> pd.DataFrame | None:
    if df is None:
        return None
    df2 = pp.impute_median(df)
    if log2:
        df2 = pp.log2p1(df2)
    return pp.zscore_rows(df2)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/paths.yaml')
    ap.add_argument('--params', default='config/params.yaml')
    args = ap.parse_args()

    paths = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    params = yaml.safe_load(open(args.params, 'r', encoding='utf-8'))

    outdir = Path(paths['outdir']); outdir.mkdir(parents=True, exist_ok=True)
    out_aligned = outdir / "aligned"; out_aligned.mkdir(parents=True, exist_ok=True)
    out_aff = outdir / "affinities"; out_aff.mkdir(parents=True, exist_ok=True)
    out_meta = outdir / "meta"; out_meta.mkdir(parents=True, exist_ok=True)
    out_feat = out_aligned / "features"; out_feat.mkdir(parents=True, exist_ok=True)

    id_level = params.get('align', {}).get('id_level', 'tcga_sample')  # tcga_sample|tcga_patient|none
    min_samples = int(params.get('align', {}).get('min_samples', 30))

    # --- Load
    print("[1/7] Loading raw CSVs ...")
    rna  = io.read_csv(paths['raw']['rna']) if paths['raw'].get('rna') else None
    lnc  = io.read_csv(paths['raw'].get('lncrna')) if paths['raw'].get('lncrna') else None
    meth = io.read_csv(paths['raw'].get('meth')) if paths['raw'].get('meth') else None
    mut  = io.read_csv(paths['raw'].get('mut')) if paths['raw'].get('mut') else None
    cnv  = io.read_csv(paths['raw'].get('cnv')) if paths['raw'].get('cnv') else None
    clin = io.read_csv(paths['raw']['clinical'])

    # --- Normalize IDs
    rna  = apply_id_norm_cols(rna, id_level)
    lnc  = apply_id_norm_cols(lnc, id_level)
    meth = apply_id_norm_cols(meth, id_level)
    mut  = apply_id_norm_cols(mut, id_level)
    cnv  = apply_id_norm_cols(cnv, id_level)
    clin = apply_id_norm_index(clin, id_level)

    mats = {'rna': rna, 'lncrna': lnc, 'meth': meth, 'mut': mut, 'cnv': cnv}

    # Sample diagnostics
    print("[2/7] Sample counts per modality (after ID normalization):")
    for k, df in mats.items():
        if df is None:
            print(f"  - {k}: None")
            continue
        print(f"  - {k}: {df.shape[1]} samples")
    print(f"  - clinical: {clin.shape[0]} rows")

    # --- Align: iteratively drop worst-overlap modality until threshold
    keep = {k for k, df in mats.items() if df is not None and df.shape[1] > 0}
    def overlap(mods):
        commons = set(clin.index)
        for m in mods:
            commons &= set(mats[m].columns)
        return commons

    commons = overlap(keep)
    while len(commons) < min_samples and len(keep) > 1:
        best_mod, best_size = None, -1
        for m in list(keep):
            test = keep - {m}
            c = overlap(test)
            if len(c) > best_size:
                best_mod, best_size = m, len(c)
        print(f"[WARN] Overlap {len(commons)} < min_samples {min_samples}. Dropping '{best_mod}' -> overlap {best_size}.")
        keep.remove(best_mod)
        commons = overlap(keep)

    if len(commons) == 0:
        # Print a few example IDs to help debug
        def sample5(x): return list(sorted(set(x)))[:5]
        print("[ERROR] No overlapping samples across modalities and clinical after normalization.")
        for k, df in mats.items():
            if df is None: continue
            print(f"  {k} example columns: {sample5(df.columns)}")
        print(f"  clinical example index: {sample5(clin.index)}")
        raise SystemExit("Fix sample ID formats or adjust align.id_level in params.yaml.")

    common = sorted(commons)
    print(f"[3/7] Aligned common samples: n={len(common)}; modalities kept: {sorted(keep)}")

    # Apply alignment
    mats_aligned = {k: (v[common] if (k in keep and v is not None) else None) for k, v in mats.items()}
    clin_a = clin.loc[common].copy()

    # --- Preprocess
    print("[4/7] Impute + log2(if set) + z-score ...")
    rna_p  = _impute_then_z(mats_aligned['rna'],   log2=bool(params.get('preprocess',{}).get('rna_log2', False))) if 'rna' in keep else None
    lnc_p  = _impute_then_z(mats_aligned['lncrna'],log2=bool(params.get('preprocess',{}).get('lnc_log2', False))) if 'lncrna' in keep else None
    meth_M = pp.beta_to_M(mats_aligned['meth']) if 'meth' in keep else None
    meth_p = pp.zscore_rows(meth_M) if meth_M is not None else None
    mut_b  = _filter_mut(mats_aligned['mut'],
                         min_prev=params.get('filters',{}).get('mut_min_prevalence', 0.01),
                         min_samples=params.get('filters',{}).get('mut_min_samples', None)) if 'mut' in keep else None
    cnv_p  = _impute_then_z(mats_aligned['cnv'], log2=False) if 'cnv' in keep else None

    # --- Variance filters
    print("[5/7] Variance filtering ...")
    rna_p  = _variance(rna_p,  params.get('filters',{}).get('rna_filter_q', 0.95)) if rna_p is not None else None
    lnc_p  = _variance(lnc_p,  params.get('filters',{}).get('lnc_filter_q', 0.95)) if lnc_p is not None else None
    meth_p = _variance(meth_p, params.get('filters',{}).get('meth_filter_q', 0.95)) if meth_p is not None else None
    cnv_p  = _variance(cnv_p,  params.get('filters',{}).get('cnv_filter_q', 0.95)) if cnv_p is not None else None

    # --- Save processed
    print("[6/7] Saving processed matrices ...")
    def _save(path: Path, df: pd.DataFrame | None):
        if df is None: return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    _save(out_aligned / "rna.processed.csv", rna_p)
    _save(out_aligned / "lncrna.processed.csv", lnc_p)
    _save(out_aligned / "meth.processed.csv", meth_p)
    _save(out_aligned / "meth.mvalues.processed.csv", meth_p)
    _save(out_aligned / "mut.binary.processed.csv", mut_b)
    _save(out_aligned / "cnv.processed.csv", cnv_p)
    clin_a.to_csv(out_aligned / "clinical.aligned.csv")

    for name, df in [('rna', rna_p), ('lncrna', lnc_p), ('meth', meth_p), ('mut', mut_b), ('cnv', cnv_p)]:
        if df is None or df.shape[0] == 0:
            continue
        (out_feat / f"{name}.features_kept.txt").write_text("\n".join(map(str, df.index)), encoding="utf-8")

    # --- Affinities
    skip_aff = bool(params.get('snf',{}).get('skip_affinities', False))
    if skip_aff:
        print("[7/7] Skipping affinity computation per params.snf.skip_affinities=true")
    else:
        print("[7/7] Building affinities ...")
        for name, df in [('rna', rna_p), ('lncrna', lnc_p), ('meth', meth_p), ('mut', mut_b), ('cnv', cnv_p)]:
            if df is None or df.shape[1] == 0 or df.shape[0] == 0:
                print(f"  - {name}: skipped (empty after alignment/filtering)")
                continue
            try:
                W = snf_utils.affinity_rbf(df.values, K=params['snf']['K'], mu=params['snf']['mu'])
                np.save(out_aff / f"{name}.affinity.npy", W)
                print(f"  - {name}: affinity saved ({W.shape})")
            except Exception as e:
                print(f"  - {name}: affinity FAILED ({e})")

    # --- Manifest
    manifest = {
        "samples_kept": len(common),
        "modalities_kept": sorted(keep),
        "align": {"id_level": id_level, "min_samples": min_samples},
        "filters": params.get('filters', {}),
        "preprocess": params.get('preprocess', {}),
        "snf": params.get('snf', {}),
        "features": {
            "rna": None if rna_p is None else int(rna_p.shape[0]),
            "lncrna": None if lnc_p is None else int(lnc_p.shape[0]),
            "meth": None if meth_p is None else int(meth_p.shape[0]),
            "mut": None if mut_b is None else int(mut_b.shape[0]),
            "cnv": None if cnv_p is None else int(cnv_p.shape[0]),
        },
    }
    (out_meta / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Step 1–7 complete.")

if __name__ == "__main__":
    main()
