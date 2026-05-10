"""
TCGA-LAML bulk preprocessing: clean the raw RPKM matrix and emit a TSV in the
default format expected by the deconvolution pipeline (genes x samples, first
column named `gene_id`).

Caveats handled here (TCGA-specific):
  - The TCGA `GeneID` column has the form "SYMBOL|ENTREZ_calculated"; we keep
    the symbol only.
  - Rows with "?" as the symbol (no annotation) are dropped.
  - After symbol extraction, multiple rows can collapse onto the same gene
    symbol; duplicates are averaged.

The script also performs a sanity check on common genes between TCGA and the
single-cell reference, warning the user if the overlap is below a threshold.

Usage:
  python aml_tcga_bulk_to_tsv.py \
      --bulk-txt /path/to/TCGA-LAML_rpkm_matrix.txt \
      --sc-h5ad  /path/to/AML_h5ad/AML210A.h5ad \
      --out-tsv  /path/to/TCGA-LAML_bulk.tsv \
      [--min-common-genes 5000]
"""

import argparse
import os
import sys

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess TCGA-LAML RPKM matrix into pipeline-ready TSV")
    parser.add_argument("--bulk-txt", required=True, help="Raw TCGA-LAML RPKM matrix (tab-separated)")
    parser.add_argument("--sc-h5ad",  required=True, help="A representative single-cell h5ad to read var_names from")
    parser.add_argument("--out-tsv",  required=True, help="Output TSV path (genes x samples, first col = gene_id)")
    parser.add_argument(
        "--min-common-genes", type=int, default=5000,
        help="Warn loudly if fewer common genes are found between bulk and sc reference (default: 5000)",
    )
    args = parser.parse_args()

    print(f"Reading TCGA bulk: {args.bulk_txt}")
    TCGA = pd.read_csv(args.bulk_txt, sep="\t")
    print(f"  raw: {TCGA.shape[0]} rows × {TCGA.shape[1] - 1} samples")

    # Strip "SYMBOL|ENTREZ_calculated" → "SYMBOL", drop unannotated "?" rows.
    TCGA["GeneID"] = TCGA["GeneID"].str.split("|").str[0]
    TCGA = TCGA[TCGA["GeneID"] != "?"].reset_index(drop=True)
    print(f"  after dropping '?' rows: {TCGA.shape[0]} rows")

    print(f"Reading sc reference var_names from: {args.sc_h5ad}")
    sc_genes = ad.read_h5ad(args.sc_h5ad).var_names

    common = sorted(set(TCGA["GeneID"]).intersection(sc_genes))
    print(f"Common genes between TCGA and sc reference: {len(common)}")
    if len(common) < args.min_common_genes:
        print(
            f"  WARNING: only {len(common)} common genes (< {args.min_common_genes}). "
            "Proceed with caution — the deconvolution model needs a rich gene overlap.",
            file=sys.stderr,
        )

    # Subset TCGA to common genes; average rows that collapsed to the same symbol.
    sample_cols = [c for c in TCGA.columns if c != "GeneID"]
    TCGA_common = TCGA[TCGA["GeneID"].isin(common)]
    TCGA_common = TCGA_common.groupby("GeneID", as_index=False)[sample_cols].mean()
    TCGA_common = TCGA_common.set_index("GeneID").loc[common].reset_index()
    TCGA_common = TCGA_common.rename(columns={"GeneID": "gene_id"})

    print(f"Output: {TCGA_common.shape[0]} genes × {TCGA_common.shape[1] - 1} samples")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)) or ".", exist_ok=True)
    TCGA_common.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Saved → {args.out_tsv}")


if __name__ == "__main__":
    main()
