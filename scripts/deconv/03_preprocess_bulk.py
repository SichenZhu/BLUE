"""Preprocess the bulk TSV into an AnnData aligned to the common gene list,
with libsize-norm + 1e6 + log2(+1) applied.

Bulk TSV format expected:
  - Tab-separated, first column named `gene_id`, remaining columns are samples.
  - Values are non-negative counts.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import anndata as ad

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402
from lib.preprocess import preprocess_counts, align_to_gene_list  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    common_path = cfg.paths.gene_lists_dir / "all_common_gene_lst.txt"
    target_genes = [g.strip() for g in common_path.read_text().splitlines() if g.strip()]
    print(f"Target genes (common): {len(target_genes)}")

    bulk_tsv = cfg.paths.bulk_tsv
    print(f"Reading {bulk_tsv}")
    df = pd.read_csv(bulk_tsv, sep="\t", index_col="gene_id")
    print(f"  raw: {df.shape[0]} genes × {df.shape[1]} samples")
    sample_ids = df.columns.tolist()

    mat = df.T.values
    src_genes = df.index.astype(str).tolist()
    aligned = align_to_gene_list(mat, src_genes, target_genes, fill_value=0.0)
    n_missing = sum(g not in set(src_genes) for g in target_genes)
    print(f"  aligned: {aligned.shape}; missing genes filled with 0: {n_missing}")

    processed = preprocess_counts(aligned, up_scaling_coeff=1e6)

    adata = ad.AnnData(X=processed)
    adata.obs_names = sample_ids
    adata.var_names = target_genes

    meta_path = cfg.paths.bulk_metadata_tsv
    if meta_path is not None and meta_path.exists():
        meta = pd.read_csv(meta_path, sep="\t")
        id_cols = [c for c in ("library_id", "sample_id", "scpca_library_id") if c in meta.columns]
        if id_cols:
            key = id_cols[0]
            meta = meta.drop_duplicates(key).set_index(key)
            shared = [s for s in sample_ids if s in meta.index]
            if shared:
                obs_new = meta.loc[shared].reindex(sample_ids)
                for c in obs_new.columns:
                    adata.obs[c] = obs_new[c].values
                print(f"  merged metadata on '{key}' for {len(shared)}/{len(sample_ids)} samples")

    out_path = cfg.paths.gene_lists_dir / "bulk_preprocessed.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"\nSaved {adata.shape} → {out_path}")
    print(f"X stats: min {adata.X.min():.3f}  max {adata.X.max():.3f}  mean {adata.X.mean():.3f}")


if __name__ == "__main__":
    main()
