#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
s09_cluster.py (v2)
-------------------
Step 9: Cluster the fused similarity matrix.

Enhancements:
- Reads fused matrix AND sample IDs (if saved by Step 8).
- Auto-k search using silhouette **on a distance matrix** derived from affinity.
- Optional --clusters to force k (e.g., 5 to mirror PAM50).
- Computes ARI/NMI vs PAM50 (if column present in aligned clinical).
- Saves: SNF_clusters.csv, cluster_metrics.csv, silhouette_vs_k.png, 2D spectral embedding plot.
- Robustness checks: symmetry, normalization, diagonal zeroing.

Requires: numpy, pandas, scikit-learn, matplotlib
"""

import argparse, yaml, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import SpectralEmbedding

def _load_fused(outdir: Path):
    Wf = np.load(outdir / "fused" / "snf_fused.npy", allow_pickle=False)
    sid_path = outdir / "fused" / "sample_ids.txt"
    sample_ids = None
    if sid_path.exists():
        sample_ids = [l.strip() for l in sid_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(sample_ids) != Wf.shape[0]:
            sample_ids = None
    return Wf, sample_ids

def _to_distance(W):
    # Normalize to [0,1], then convert to distance
    W = np.array(W, dtype=float)
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, W.diagonal())  # keep as is
    wmax = np.max(W) if np.max(W) > 0 else 1.0
    S = W / wmax
    np.fill_diagonal(S, 1.0)  # self-similarity at max
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D

def _spectral_cluster(W, k, seed=7):
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=seed)
    lab = sc.fit_predict(W)
    return lab

def _auto_k(W, k_min=2, k_max=8, seed=7):
    D = _to_distance(W)
    best_k, best_s = None, -1
    rows = []
    for k in range(k_min, k_max+1):
        try:
            lab = _spectral_cluster(W, k, seed=seed)
            s = silhouette_score(D, lab, metric='precomputed')
            rows.append({"k": k, "silhouette": s})
            if s > best_s:
                best_k, best_s = k, s
        except Exception as e:
            rows.append({"k": k, "silhouette": np.nan})
            continue
    return best_k, pd.DataFrame(rows)

def _plot_silhouette(df, out_png):
    plt.figure(figsize=(5,3))
    plt.plot(df['k'], df['silhouette'], marker='o')
    plt.xlabel("k (clusters)")
    plt.ylabel("Silhouette (distance)")
    plt.title("Silhouette vs k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def _spectral_embed_plot(W, labels, sample_ids, out_png):
    emb = SpectralEmbedding(n_components=2, affinity='precomputed', random_state=7)
    X2 = emb.fit_transform(W)
    plt.figure(figsize=(5,4))
    sc = plt.scatter(X2[:,0], X2[:,1], c=labels, s=12, alpha=0.9, cmap='tab10')
    plt.xlabel("Spectral-1"); plt.ylabel("Spectral-2")
    plt.title("Spectral embedding (colored by cluster)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/paths.yaml')
    ap.add_argument('--params', default='config/params.yaml')
    ap.add_argument('--clusters', type=int, default=None, help='Force k. If omitted, auto-k search is used.')
    ap.add_argument('--k_min', type=int, default=None, help='Min k for auto search (overrides params).')
    ap.add_argument('--k_max', type=int, default=None, help='Max k for auto search (overrides params).')
    ap.add_argument('--seed',   type=int, default=7)
    args = ap.parse_args()

    paths  = yaml.safe_load(open(args.config))
    params = yaml.safe_load(open(args.params))
    outdir = Path(paths['outdir'])
    (outdir / 'tables').mkdir(parents=True, exist_ok=True)
    (outdir / 'figures').mkdir(parents=True, exist_ok=True)

    # Load fused similarity and sample IDs
    Wf, sample_ids = _load_fused(outdir)
    n = Wf.shape[0]
    if sample_ids is None:
        sample_ids = [f"S{i+1}" for i in range(n)]

    # Decide k
    if args.clusters is not None:
        k = int(args.clusters)
        metrics_df = pd.DataFrame([{"k": k, "silhouette": np.nan}])
    else:
        kmin = args.k_min or params.get('cluster', {}).get('k_min', 2)
        kmax = args.k_max or params.get('cluster', {}).get('k_max', 8)
        k, metrics_df = _auto_k(Wf, k_min=int(kmin), k_max=int(kmax), seed=args.seed)
        _plot_silhouette(metrics_df, outdir / 'figures' / 'silhouette_vs_k.png')

    # Final clustering
    labels = _spectral_cluster(Wf, k, seed=args.seed)
    lab_s = pd.Series(labels, index=sample_ids, name='cluster')
    lab_s.to_frame().to_csv(outdir / 'tables' / 'SNF_clusters.csv')
    print(f"Clustering done with k={k}. Saved tables/SNF_clusters.csv")

    # Compute final silhouette for chosen k
    try:
        sil = silhouette_score(_to_distance(Wf), labels, metric='precomputed')
    except Exception:
        sil = np.nan

    # Optional: ARI/NMI vs PAM50 if available
    clin_path = outdir / 'aligned' / 'clinical.aligned.csv'
    ari = nmi = np.nan
    if clin_path.exists():
        clin = pd.read_csv(clin_path, index_col=0)
        pam_col = None
        for c in clin.columns:
            if c.strip().upper() == 'PAM50':
                pam_col = c
                break
        if pam_col:
            pam = clin.loc[lab_s.index, pam_col].astype(str)
            # drop NA-like
            mask = pam.notna() & pam.ne("") & pam.str.upper().ne("NA")
            pam = pam[mask]; lab_cmp = lab_s[mask]
            if pam.size > 10:
                # Map PAM50 strings to ints for ARI/NMI
                _, pam_codes = np.unique(pam.values, return_inverse=True)
                ari = adjusted_rand_score(pam_codes, lab_cmp.values)
                nmi = normalized_mutual_info_score(pam_codes, lab_cmp.values)

    # Save metrics table
    metrics_df = metrics_df.copy()
    metrics_df.loc[metrics_df['k']==k, 'silhouette_final'] = sil
    metrics_df['ARI_vs_PAM50'] = ari
    metrics_df['NMI_vs_PAM50'] = nmi
    metrics_df.to_csv(outdir / 'tables' / 'cluster_metrics.csv', index=False)

    # Embedding plot
    _spectral_embed_plot(Wf, labels, sample_ids, outdir / 'figures' / 'cluster_embedding_spectral.png')

if __name__ == "__main__":
    main()
