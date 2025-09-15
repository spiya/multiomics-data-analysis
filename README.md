# BRCA Multi-Omics Modular Pipeline

A modular, scriptable pipeline to discover **molecular subtypes** and **biomarkers** in breast cancer from multi-omics data (mRNA, lncRNA, DNA methylation, mutations, ±CNV), with optional **survival**, **classifiers**, and **GSEA**.  
It is designed to run step-by-step, cache intermediate results, and let you re-run only the parts you change.

---

## Highlights
- **Robust sample alignment** with TCGA-style ID normalization (patient or sample level).
- **Per-modality preprocessing** (impute → transform → z-score → variance filter).
- **Similarity Network Fusion (SNF)** to combine omics into a single fused patient network.
- **Clustering** (k fixed or estimated), **clinical associations**, **survival** (if available).
- **Differential features** per cluster and **integrated biomarker ranking** across omics.

---

## Project layout

```
brca_multiomics_modular/
├─ config/
│  ├─ paths.yaml               # where your inputs live and where outputs go (outdir)
│  └─ params.yaml              # algorithm settings, filters, modalities to use, GSEA options
├─ scripts/
│  ├─ s01_07_prep_v3.py        # Steps 1–7: load → align → preprocess → affinities
│  ├─ s08_fuse.py              # Step 8: SNF fusion
│  ├─ s09_cluster.py           # Step 9: clustering on fused network
│  ├─ s10_clinical.py          # Step 10: clinical associations (+ optional survival)
│  ├─ s11_de.py                # Step 11: differential features by cluster
│  ├─ s12_classify.py          # Step 12: optional classifiers
│  ├─ s13_biomarkers.py        # Step 13: integrated biomarker ranking
└─ scripts/brca_mo/            # lightweight “package” of helpers
```

---

## What each step does & what to check

1–7 **Load → Align → QC/Preprocess → Affinities**  
- Normalizes IDs, aligns common samples, imputes/log-transforms/z-scores, filters by variance, builds per-omic affinities.  
- **Outputs:** `aligned/*.processed.csv`, `affinities/*.npy`, `meta/preprocess_manifest.json`.  
- **Check:** samples kept > ~100 (for TCGA), no empty matrices, affinities saved.

8 **SNF fusion**  
- Fuses per-omic affinities into one robust patient×patient network.  
- **Outputs:** `fused/W_fused.npy` (+ optional heatmap).  
- **Check:** clear block structure; all intended modalities included.

9 **Clustering**  
- Spectral clustering on `W_fused` into `k` clusters (fix k=5 for PAM50-like).  
- **Outputs:** `tables/SNF_clusters.csv`.  
- **Check:** reasonable cluster sizes; separation in UMAP/heatmap.

10 **Clinical associations (+survival)**  
- χ² for categorical, Kruskal-Wallis for numeric; optional Cox/KM if survival columns exist.  
- **Outputs:** association tables + plots.  
- **Check:** sensible associations (e.g., stage, age), survival trends if present.

11 **Differential features**  
- Per cluster vs others across each omic; FDR control.  
- **Outputs:** `tables/DE_<OMIC>_by_cluster.csv`.  
- **Check:** top features match expected subtype biology.

12 **Classifiers (optional)**  
- Supervised models (e.g., multinomial logistic / elastic-net) with nested CV.  
- **Outputs:** CV metrics, confusion matrix, predictions.  
- **Check:** macro-F1 (not just accuracy), good balance across classes.

13 **Biomarker integration**  
- Integrates DE evidence across omics (weighted), flags cross-omic consistency, optionally annotates actionability.  
- **Outputs:** `tables/biomarkers_<OMIC>_cluster*.csv`, `tables/biomarkers_INTEGRATED_cluster*.csv`, optional Excel.  
- **Check:** top integrated hits make biological sense; cross-omic support (RNA↑ + CNV gain, RNA↓ + METH hyper).

---

## Citation & License

If you use this repository in your work, please cite the repo URL and this pipeline.  
License: MIT (or your preferred license—update this line).

---

## Contact

Questions/ideas/PRs welcome. Open an issue or pull request.
