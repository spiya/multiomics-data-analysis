#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s08_fuse.py (v2)
----------------
Step 8: Fuse per-modality affinities with SNF (or weighted average fallback).

Improvements vs v1:
- Loads affinities if present OR rebuilds them from processed matrices when missing.
- Ensures identical sample order across modalities (intersects samples and reindexes).
- Optional modality weights (params.snf.weights) for the average fallback.
- Saves the *sample order* used for the fused matrix for downstream clustering.
- Writes a fusion manifest with included/excluded modalities and counts.

Inputs expected under --config/--params:
  config/paths.yaml:
    outdir: "<results folder>"

  config/params.yaml:
    modalities: ['rna','lncrna','meth','mut','cnv']
    snf:
      K: 10
      t: 10
      mu: 0.5
      weights: {'rna':1.0,'lncrna':1.0,'meth':1.0,'mut':1.0,'cnv':1.0}  # optional

Requires: numpy, pandas, scikit-learn (for pairwise distances if affinity rebuild is needed), snfpy optional.
"""

import argparse, yaml, json
from pathlib import Path
import numpy as np
import pandas as pd

from funct import io, snf_utils

PROC_MAP = {
    'rna':   ('aligned/rna.processed.csv',           True),
    'lncrna':('aligned/lncrna.processed.csv',        True),
    'meth':  ('aligned/meth.mvalues.processed.csv',  True),  # prefer M-values
    'mut':   ('aligned/mut.binary.processed.csv',    False), # binary
    'cnv':   ('aligned/cnv.processed.csv',           True),
}

def _load_matrix(outdir: Path, mod: str):
    rel, _ = PROC_MAP[mod]
    p = outdir / rel
    if not p.exists():
        # try alternate for methylation
        if mod == 'meth':
            p2 = outdir / 'aligned' / 'meth.processed.csv'
            return (io.read_csv(p2) if p2.exists() else None)
        return None
    return io.read_csv(p)

def _rebuild_affinity(df: pd.DataFrame, K: int, mu: float):
    return snf_utils.affinity_rbf(df.values, K=K, mu=mu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/paths.yaml')
    ap.add_argument('--params', default='config/params.yaml')
    args = ap.parse_args()

    paths  = yaml.safe_load(open(args.config))
    params = yaml.safe_load(open(args.params))
    outdir = Path(paths['outdir'])
    (outdir / 'fused').mkdir(parents=True, exist_ok=True)
    (outdir / 'meta').mkdir(parents=True, exist_ok=True)

    mods = params.get('modalities', ['rna','lncrna','meth','mut','cnv'])
    K = int(params['snf']['K']); t = int(params['snf']['t']); mu = float(params['snf']['mu'])
    weights = {m: float(w) for m, w in params.get('snf', {}).get('weights', {}).items()} if params.get('snf', {}).get('weights') else {}

    # Load matrices to infer sample sets
    matrices = {}
    for m in mods:
        if m not in PROC_MAP: 
            print(f"[WARN] Unknown modality '{m}', skipping.")
            continue
        df = _load_matrix(outdir, m)
        if df is None or df.shape[1] < 3:
            print(f"[WARN] Missing or tiny matrix for {m}, skipping.")
            continue
        matrices[m] = df

    if not matrices:
        raise SystemExit("No usable processed matrices found under aligned/. Run Step 1–7 first.")

    # Intersect sample IDs across all kept modalities
    sample_sets = [set(df.columns) for df in matrices.values()]
    common = sorted(set.intersection(*sample_sets))
    if len(common) < 10:
        raise SystemExit(f"Only {len(common)} overlapping samples across modalities. Check alignment.")
    print(f"[1/3] Common samples across modalities: n={len(common)}")

    # Collect affinities aligned to common sample order (rebuild if missing)
    W_list, used_mods, skipped_mods = [], [], []
    for m, df in matrices.items():
        aff_path = outdir / 'affinities' / f"{m}.affinity.npy"
        if aff_path.exists():
            W = np.load(aff_path, allow_pickle=False)
            # Reorder by common sample order via df columns (affinity built in df order originally)
            idx = [df.columns.get_loc(s) for s in common]
            try:
                Wm = W[np.ix_(idx, idx)]
            except Exception:
                print(f"[WARN] Affinity for {m} has wrong shape; rebuilding from matrix.")
                Wm = _rebuild_affinity(df.loc[:, common], K=K, mu=mu)
        else:
            print(f"[INFO] No affinity file for {m}; building from processed matrix.")
            Wm = _rebuild_affinity(df.loc[:, common], K=K, mu=mu)

        # Basic validity checks
        if not np.isfinite(Wm).all() or Wm.shape[0] != len(common) or Wm.shape[0] != Wm.shape[1]:
            print(f"[WARN] Invalid affinity for {m}; skipping.")
            skipped_mods.append(m); 
            continue

        W_list.append((m, Wm))
        used_mods.append(m)

    if not W_list:
        raise SystemExit("No valid affinities to fuse.")

    print(f"[2/3] Modalities used: {used_mods}")
    # SNF fuse (or weighted average fallback inside snf_utils)
    try:
        from snf import snf as _snf
        Wf = _snf.snf([W for _, W in W_list], K=None, t=t)
        method = "SNF"
    except Exception:
        # Weighted average fallback
        ws = np.array([weights.get(m, 1.0) for m, _ in W_list], dtype=float)
        ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(W_list))/len(W_list)
        stack = np.stack([W for _, W in W_list], axis=0)
        Wf = np.tensordot(ws, stack, axes=(0,0))
        method = "weighted_average"

    np.save(outdir / 'fused' / 'snf_fused.npy', Wf)
    (outdir / 'fused' / 'sample_ids.txt').write_text("\n".join(common), encoding="utf-8")

    manifest = {
        "method": method,
        "modalities_used": used_mods,
        "modalities_skipped": skipped_mods,
        "n_samples": len(common),
        "snf_params": {"K": K, "t": t, "mu": mu},
        "weights": {m: weights.get(m, 1.0) for m in used_mods},
        "affinity_sources": {m: str(outdir / 'affinities' / f"{m}.affinity.npy") for m in used_mods},
    }
    (outdir / 'meta' / 'fusion_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"[3/3] Saved fused/snf_fused.npy with {len(common)} samples; method={method}")

if __name__ == "__main__":
    main()
