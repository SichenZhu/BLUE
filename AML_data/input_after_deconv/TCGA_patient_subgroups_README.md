# TCGA_patient_subgroups.csv

Per-patient annotation table for the TCGA-LAML cohort (167 patients). It is the
`patient_info_csv` input to the AML downstream-analysis pipeline
(`scripts/aml_followup/`).

One row per patient; the first (unnamed) column is the TCGA patient ID
(e.g. `TCGA-AB-2803`).

## Columns

| Column | Description |
|--------|-------------|
| *(index)* | TCGA patient ID. |
| `HSC-like`, `Prog-like`, `GMP-like`, `ProMono-like`, `Mono-like`, `cDC-like` | BLUE's predicted cell-type proportions for each patient (one column per cell type). |
| `PC1`, `PC2` | First two principal components of a PCA decomposition of BLUE's predicted cell-type proportions. |
| `Cluster` | Patient-group label from **reproducing** the patient-grouping pipeline of the [AML study](https://www.nature.com/articles/s41591-022-01819-x) (Nature Medicine, 2022) on BLUE's predicted cell-type proportions. |
| `os` | Overall-survival time, in days. |
| `osi` | Overall-survival event indicator: `1` = deceased, `0` = alive / censored. |
| `paper_cluster` | The original four patient-group labels as **published** in the [AML study](https://www.nature.com/articles/s41591-022-01819-x) (Nature Medicine, 2022). |

## Column sources

1. **`os`, `osi`** — overall-survival information from the TCGA database, taken
   from `TCGA-LAML_survival.txt` (`os` ← its `OS.time` column, `osi` ← its `OS`
   column).

2. **`Cluster`** — produced by reproducing the patient-grouping pipeline of the
   [AML study](https://www.nature.com/articles/s41591-022-01819-x), **based on
   BLUE's predicted cell-type proportions**.

3. **`paper_cluster`** — the original four patient-group labels provided by that
   same [study](https://www.nature.com/articles/s41591-022-01819-x).

4. **`*-like` cell-type proportions and `PC1`, `PC2`** — BLUE's deconvolution
   predictions, and a PCA decomposition of those predicted proportions.
