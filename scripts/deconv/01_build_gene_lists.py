"""Build gene lists for the UNet (config-driven).

  all_common_gene_lst.txt : sc.var_names ∩ bulk gene_id
  all_DEG_lst.txt          : per-patient one-vs-rest DEGs (FDR < cfg.degs.fdr_rate,
                             up to cfg.degs.num_deg_per_ct positive-logFC per CT,
                             greedy dedup) ∩ common.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import anndata as ad

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402
from lib.degs import preprocess_for_degs, rank_genes_one_vs_rest, find_DEGs  # noqa: E402

CTKEY = "cell.type.for.deconv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = load_celltype_mapping(cfg.paths.celltype_mapping_yaml)
    ctlist = mapping["ctlist"]

    print(f"Loading {cfg.paths.combined_sc_h5ad} …")
    a = ad.read_h5ad(cfg.paths.combined_sc_h5ad)
    print(f"  {a.shape}")

    sc_genes = pd.Index(a.var_names).astype(str)
    bulk_genes = pd.read_csv(cfg.paths.bulk_tsv, sep="\t", usecols=["gene_id"])["gene_id"].astype(str)
    print(f"  sc genes: {len(sc_genes)}  bulk genes: {len(bulk_genes)}")

    common = sorted(set(sc_genes) & set(bulk_genes))
    print(f"Common genes (sc ∩ bulk): {len(common)}")

    out_dir = cfg.paths.gene_lists_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    common_path = out_dir / "all_common_gene_lst.txt"
    with common_path.open("w") as f:
        for g in common:
            f.write(g + "\n")
    print(f"  wrote {common_path}")

    fdr = cfg.degs.fdr_rate
    n_deg = cfg.degs.num_deg_per_ct
    print(f"\nPer-patient DEGs (FDR<{fdr}, up to {n_deg} per CT)…")

    deg_union: set[str] = set()
    libs = sorted(a.obs[cfg.sc_schema.library_id_col].astype(str).unique())
    for i, lib in enumerate(libs, 1):
        sub = a[a.obs[cfg.sc_schema.library_id_col].astype(str) == lib].copy()
        if sub.obs[CTKEY].nunique() < 2:
            print(f"[{i:2d}/{len(libs)}] {lib}: <2 cell types, skip")
            continue
        sub_p = preprocess_for_degs(sub, groupby=CTKEY, desired_ct_lst=ctlist, min_cells_per_ct=2)
        if sub_p.obs[CTKEY].nunique() < 2:
            print(f"[{i:2d}/{len(libs)}] {lib}: <2 cell types after filter, skip")
            continue
        print(f"[{i:2d}/{len(libs)}] {lib}: {sub_p.shape}, {sub_p.obs[CTKEY].nunique()} cell types")
        rank_genes_one_vs_rest(sub_p, groupby=CTKEY)
        degs = find_DEGs(sub_p, groupby=CTKEY, num_DEG_per_ct=n_deg, fdr_rate=fdr)
        deg_union |= set(degs)
        print(f"    DEGs: {len(degs)}  running union: {len(deg_union)}")

    deg_final = sorted(deg_union & set(common))
    deg_path = out_dir / "all_DEG_lst.txt"
    with deg_path.open("w") as f:
        for g in deg_final:
            f.write(g + "\n")
    print(f"\nDEG union {len(deg_union)} → ∩ common {len(deg_final)}")
    print(f"  wrote {deg_path}")


if __name__ == "__main__":
    main()
