#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
multiome_heatmap_fixed_v6.py

- Guarantees legends do NOT overlap the heatmaps by reserving bottom margin.
- Controls:
    --legend_space 0.16   (fraction of figure reserved at the bottom)
    --figscale 0.90       (overall figure size scaler; smaller = shorter panels)

Run:
  python "C:\Users\Utsala\Desktop\Sarbottam\Final\multiomics-data-analysis\extras\multiome_heatmap_fixed_v5.py" ^
  --outdir "C:\Users\Utsala\Desktop\Sarbottam\Final\multiomics-data-analysis\results" ^
  --panels rna,lncrna,meth,mut ^
  --pam50_col PAM50 --stage_col pstage --age_col age ^
  --cluster_file "C:\Users\Utsala\Desktop\Sarbottam\Final\multiomics-data-analysis\results\tables\SNF_clusters.csv" ^
  --no_titles
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _read_df(p: Path) -> pd.DataFrame | None:
    if p is None or not p.exists():
        print(f"[WARN] Missing file: {p}")
        return None
    try:
        return pd.read_csv(p, index_col=0)
    except Exception as e:
        print(f"[WARN] Failed reading {p}: {e}")
        return None

def _cap(df: pd.DataFrame, vmin=-2.5, vmax=2.5) -> pd.DataFrame:
    return df.clip(lower=vmin, upper=vmax)

def _pick_top_rows_by_var(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    v = df.var(axis=1).sort_values(ascending=False)
    return df.loc[v.index[:min(k, len(v))]]

def _pick_top_rows_by_prevalence(df_bin: pd.DataFrame, k: int) -> pd.DataFrame:
    if df_bin is None or df_bin.empty:
        return df_bin
    prev = df_bin.mean(axis=1).sort_values(ascending=False)
    return df_bin.loc[prev.index[:min(k, len(prev))]]

def _to_rgb(color_spec):
    try:
        return mcolors.to_rgb(color_spec)
    except Exception:
        return (0.9, 0.9, 0.9)

def _make_age_cmap(cmin: str, cmed: str, cmax: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("age3", [_to_rgb(cmin), _to_rgb(cmed), _to_rgb(cmax)], N=256)

def _draw_annotation_bar(ax, categories, mapping, label=None):
    colors = np.array([_to_rgb(mapping.get(v, (0.9, 0.9, 0.9))) for v in categories])[None, :, :]
    ax.imshow(colors, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=6)

def _draw_continuous_bar(ax, values, cmap, vmin=None, vmax=None, label=None):
    vals = np.array(values, dtype=float)[None, :]
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, labelpad=6)
    return im

def plot_multiome_heatmap(
        outdir: Path,
        panels=('rna','lncrna','meth','mut','cnv'),
        rows_per_panel=100,
        vcap=2.5,
        figscale=0.9,
        legend_space=0.16,
        annotate_clusters=True,
        show_titles=False,
        pam50_col="PAM50",
        stage_col="pstage",
        age_col="age",
        cluster_file=None
    ):
    # Load matrices
    rna   = _read_df(outdir / "aligned" / "rna.processed.csv")
    lnc   = _read_df(outdir / "aligned" / "lncrna.processed.csv")
    meth  = _read_df(outdir / "aligned" / "meth.processed.csv")
    mut   = _read_df(outdir / "aligned" / "mut.binary.processed.csv")
    cnv   = _read_df(outdir / "aligned" / "cnv.processed.csv")
    clin  = _read_df(outdir / "aligned" / "clinical.aligned.csv")

    # Clusters
    cpath = Path(cluster_file) if cluster_file else (outdir / "tables" / "SNF_clusters.csv")
    cltab = _read_df(cpath)

    # Sample order
    order = None
    if annotate_clusters and cltab is not None and not cltab.empty:
        lab = cltab.iloc[:,0].astype(str)
        candidates = set()
        for df in [rna, lnc, meth, mut, cnv]:
            if df is not None:
                candidates |= set(df.columns)
        if clin is not None:
            candidates &= set(clin.index)
        idx = [i for i in lab.index if i in candidates]
        print(f"[i] Cluster labels: {len(lab)} total; overlap with matrices/clinical: {len(idx)}")
        lab = lab.loc[idx]
        order = lab.sort_values(kind="mergesort").index.tolist()

    if order is None:
        for df in [rna, lnc, meth, mut, cnv]:
            if df is not None and df.shape[1] > 0:
                order = list(df.columns); break
        if order is None:
            raise SystemExit("No matrices found to infer sample order. Pass the run root as --outdir.")

    def _align_cols(df):
        return None if df is None else df.loc[:, [s for s in order if s in df.columns]]
    rna, lnc, meth, mut, cnv = map(_align_cols, [rna, lnc, meth, mut, cnv])
    if clin is not None:
        clin = clin.reindex(order)

    # Show clinical columns
    if clin is not None:
        print(f"[i] clinical.aligned.csv columns (first 12): {list(clin.columns)[:12]}")
        missing = [x for x in [pam50_col, stage_col, age_col] if x not in clin.columns]
        if missing:
            print(f"[WARN] Clinical missing columns: {missing}")

    # Build display matrices
    panel_data = {}
    if 'rna' in panels and rna is not None:
        panel_data['rna'] = _cap(_pick_top_rows_by_var(rna, rows_per_panel), -vcap, vcap)
    if 'lncrna' in panels and lnc is not None:
        panel_data['lncrna'] = _cap(_pick_top_rows_by_var(lnc, rows_per_panel), -vcap, vcap)
    if 'meth' in panels and meth is not None:
        panel_data['meth'] = _cap(_pick_top_rows_by_var(meth, rows_per_panel), -vcap, vcap)
    if 'cnv' in panels and cnv is not None:
        panel_data['cnv'] = _cap(_pick_top_rows_by_var(cnv, rows_per_panel), -vcap, vcap)
    if 'mut' in panels and mut is not None:
        panel_data['mut'] = _pick_top_rows_by_prevalence(mut, rows_per_panel)

    panels = [p for p in panels if p in panel_data]
    if not panels:
        raise SystemExit("No panels to plot after filtering.")

    n_samples = len(order)
    ann_rows = 3 + (1 if (annotate_clusters and cltab is not None and not cltab.empty) else 0)

    panel_heights = []
    for p in panels:
        nrows = panel_data[p].shape[0]
        panel_heights.append(max(1.5, nrows * 0.035))  # slightly shorter than v5

    total_height = figscale * (ann_rows * 0.5 + sum(panel_heights) + 1.0)
    total_width  = figscale * max(8.0, 0.015 * n_samples + 3.5)

    fig = plt.figure(figsize=(total_width, total_height), constrained_layout=False)
    # Reserve bottom margin for legends so nothing overlaps
    fig.subplots_adjust(left=0.10, right=0.95, top=0.98, bottom=legend_space, hspace=0.08)

    gs = fig.add_gridspec(nrows=ann_rows + len(panels),
                          ncols=1,
                          height_ratios=[0.35]*ann_rows + panel_heights)

    pam_colors = {"Basal":"blue", "Her2":"red", "LumA":"yellow", "LumB":"green", "Normal":"black"}
    stage_colors = {"T1":"green", "T2":"blue", "T3":"red", "T4":"yellow", "TX":"black"}
    cluster_palette = None
    clser = None
    if annotate_clusters and cltab is not None and not cltab.empty:
        clser = cltab.iloc[:,0].astype(str).reindex(order)
        uniq = sorted([u for u in clser.dropna().unique()], key=lambda x: (len(str(x)), str(x)))
        base_cmap = plt.get_cmap("tab10")
        cluster_palette = {c: base_cmap(i % 10) for i, c in enumerate(uniq)}
        print(f"[i] Cluster levels: {uniq}")

    # Store age stats for colorbar
    age_vals = None; age_cmap = _make_age_cmap("#0000AA", "#555555", "#AAAA00")
    arow = 0
    if annotate_clusters and clser is not None:
        ax = fig.add_subplot(gs[arow, 0])
        _draw_annotation_bar(ax, clser.fillna("NA").astype(str).tolist(), cluster_palette, label="Cluster")
        arow += 1

    if clin is not None and pam50_col in clin.columns:
        ax = fig.add_subplot(gs[arow, 0])
        _draw_annotation_bar(ax, clin[pam50_col].astype(str).tolist(), pam_colors, label="PAM50")
        arow += 1

    if clin is not None and stage_col in clin.columns:
        ax = fig.add_subplot(gs[arow, 0])
        _draw_annotation_bar(ax, clin[stage_col].astype(str).tolist(), stage_colors, label="pStage")
        arow += 1

    if clin is not None and age_col in clin.columns:
        ax = fig.add_subplot(gs[arow, 0])
        age_vals = pd.to_numeric(clin[age_col], errors="coerce")
        vmin, vmax = float(np.nanmin(age_vals)), float(np.nanmax(age_vals))
        _draw_continuous_bar(ax, age_vals.fillna(age_vals.median()), age_cmap, vmin=vmin, vmax=vmax, label="Age")
        arow += 1

    cmap_cont = plt.get_cmap("RdBu_r")
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[ann_rows + i, 0])
        mat = panel_data[p].values
        if p == 'mut':
            colors = np.where(mat > 0.5, 0.0, 0.85)
            ax.imshow(colors, aspect="auto", cmap="gray", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(panel_data[p].values, aspect="auto", cmap=cmap_cont, vmin=-vcap, vmax=vcap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.2%", pad=0.10)  # size and pad you can tweak
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=8)

        ax.set_yticks([]); ax.set_xticks([]); ax.set_xlabel("")
        ax.set_ylabel(p.upper(), rotation=0, ha="right", va="center", fontsize=10, labelpad=14)
        if show_titles:
            title_map = {"rna":"mRNA (z)", "lncrna":"lncRNA (z)", "meth":"Methylation (z of M-values)", "cnv":"CNV (z)", "mut":"Mutation (binary)"}
            ax.set_title(title_map.get(p, p), fontsize=10, pad=4)

    # === Legends area (in the reserved bottom margin) ===
    y0 = 0.02  # within the reserved bottom
    # Cluster legend (left)
    if annotate_clusters and cluster_palette:
        axL = fig.add_axes([0.06, y0, 0.26, legend_space - 0.04]); axL.axis("off")
        handles = [plt.Line2D([0], [0], marker='s', color=_to_rgb(v), markerfacecolor=_to_rgb(v),
                               linestyle='None', markersize=8, label=str(k)) for k, v in cluster_palette.items()]
        axL.legend(handles=handles, title="Cluster", ncol=min(4, len(handles)), frameon=False,
                   loc='center left', fontsize=8, title_fontsize=9)

    # PAM50 legend (middle-left)
    axP = fig.add_axes([0.33, y0, 0.30, legend_space - 0.04]); axP.axis("off")
    handles_p = [plt.Line2D([0], [0], marker='s', color=_to_rgb(v), markerfacecolor=_to_rgb(v),
                        linestyle='None', markersize=8, label=str(k))
             for k, v in {"Basal":"blue","Her2":"red","LumA":"yellow","LumB":"green","Normal":"black"}.items()]
    axP.legend(handles=handles_p, title="PAM50", ncol=5, frameon=False,
           loc='center left', fontsize=8, title_fontsize=9,
           handletextpad=0.6, columnspacing=0.8)

    # pStage legend (middle-right)
    axS = fig.add_axes([0.66, y0, 0.22, legend_space - 0.04]); axS.axis("off")
    handles_s = [plt.Line2D([0], [0], marker='s', color=_to_rgb(v), markerfacecolor=_to_rgb(v),
                        linestyle='None', markersize=8, label=str(k))
             for k, v in {"T1":"green","T2":"blue","T3":"red","T4":"yellow","TX":"black"}.items()]
    axS.legend(handles=handles_s, title="pStage", ncol=5, frameon=False,
           loc='center left', fontsize=8, title_fontsize=9,
           handletextpad=0.6, columnspacing=0.8)

    # Age colorbar (right)
    if age_vals is not None:
        axA = fig.add_axes([0.82, y0 + 0.01, 0.14, legend_space - 0.06])  # horizontal bar
        norm = plt.Normalize(vmin=float(np.nanmin(age_vals)), vmax=float(np.nanmax(age_vals)))
        sm = ScalarMappable(norm=norm, cmap=age_cmap); sm.set_array([])
        cb = plt.colorbar(sm, cax=axA, orientation='horizontal')
        cb.set_label("Age", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    out_fig = outdir / "figures" / "multiome_heatmap.png"
    out_pdf = outdir / "figures" / "multiome_heatmap.pdf"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved heatmap -> {out_fig}")
    print(f"[OK] Saved heatmap -> {out_pdf}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Run outdir (contains aligned/, tables/)")
    ap.add_argument("--panels", default="rna,lncrna,meth,mut,cnv", help="Comma-separated subset to plot")
    ap.add_argument("--rows_per_panel", type=int, default=100)
    ap.add_argument("--vcap", type=float, default=2.5)
    ap.add_argument("--figscale", type=float, default=0.9)
    ap.add_argument("--legend_space", type=float, default=0.16, help="Bottom margin reserved for legends (0-0.3)")
    ap.add_argument("--no_cluster_annot", action="store_true")
    ap.add_argument("--no_titles", action="store_true")
    ap.add_argument("--pam50_col", default="PAM50")
    ap.add_argument("--stage_col", default="pstage")
    ap.add_argument("--age_col", default="age")
    ap.add_argument("--cluster_file", default=None, help="Override path to clusters CSV")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    panels = [p.strip().lower() for p in args.panels.split(",") if p.strip()]
    plot_multiome_heatmap(
        outdir=outdir,
        panels=panels,
        rows_per_panel=args.rows_per_panel,
        vcap=args.vcap,
        figscale=args.figscale,
        legend_space=args.legend_space,
        annotate_clusters=(not args.no_cluster_annot),
        show_titles=(not args.no_titles),
        pam50_col=args.pam50_col,
        stage_col=args.stage_col,
        age_col=args.age_col,
        cluster_file=args.cluster_file
    )

if __name__ == "__main__":
    main()
