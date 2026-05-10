"""Combine per-patient sc h5ads into one labeled reference h5ad.

For every file under <paths.sc_h5ad_dir>/<paths.sc_h5ad_glob>:
  1. read it,
  2. ensure raw counts are in .X (or pull from the layer named in
     sc_schema.raw_counts_layer),
  3. apply the fine → coarse mapping from <paths.celltype_mapping_yaml>
     to the obs column named in sc_schema.fine_celltype_col,
  4. store the result in obs['cell.type.for.deconv'],
  5. drop cells whose fine label is not in the mapping.

All h5ads are then concatenated (inner gene join) and saved to
<paths.combined_sc_h5ad>.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402

CTKEY = "cell.type.for.deconv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = load_celltype_mapping(cfg.paths.celltype_mapping_yaml)
    fine_to_coarse = mapping["fine_to_coarse"]
    print(f"Coarse cell types ({len(mapping['ctlist'])}): {mapping['ctlist']}")

    sc_dir = cfg.paths.sc_h5ad_dir
    glob = cfg.sc_schema.__dict__.get("glob") or "*.h5ad"
    glob = getattr(cfg.paths, "sc_h5ad_glob", glob) if hasattr(cfg.paths, "sc_h5ad_glob") else glob
    files = sorted(Path(sc_dir).rglob(glob))
    if not files:
        raise FileNotFoundError(f"No h5ads matched {sc_dir}/{glob}")
    print(f"Found {len(files)} per-patient h5ads under {sc_dir}")

    fine_col = cfg.sc_schema.fine_celltype_col
    lib_col = cfg.sc_schema.library_id_col
    layer_name = cfg.sc_schema.raw_counts_layer  # may be None

    adatas: list[ad.AnnData] = []
    stats = []
    seen_unmapped: set[str] = set()
    for f in files:
        a = ad.read_h5ad(f)
        if fine_col not in a.obs.columns:
            print(f"  skip {f.name}: no obs['{fine_col}']")
            continue

        if layer_name:
            if layer_name not in a.layers:
                raise ValueError(f"{f.name}: layer '{layer_name}' missing")
            a.X = a.layers[layer_name]

        fine = a.obs[fine_col].astype(str)
        unmapped = sorted(set(fine.unique()) - set(fine_to_coarse.keys()))
        if unmapped:
            seen_unmapped.update(unmapped)
        coarse = fine.map(fine_to_coarse)
        keep = coarse.notna().to_numpy()
        if keep.sum() == 0:
            print(f"  skip {f.name}: 0 cells after mapping")
            continue
        a = a[keep].copy()
        a.obs[CTKEY] = pd.Categorical(coarse[keep].values)
        if lib_col not in a.obs.columns:
            a.obs[lib_col] = f.stem  # fall back to filename
        adatas.append(a)
        stats.append({"file": f.name, "kept_cells": int(keep.sum()),
                      "library_id_unique": int(a.obs[lib_col].nunique())})
        print(f"  {f.name}: {keep.sum()} cells kept (of {len(fine)})")

    if seen_unmapped:
        print(f"\nWARNING — fine labels not in mapping (cells dropped):\n  "
              + ", ".join(sorted(seen_unmapped)))

    print(f"\nConcatenating {len(adatas)} h5ads…")
    combined = ad.concat(adatas, join="inner", merge="first")
    combined.obs_names_make_unique()

    # Sanity check: X has integer counts (warn if not)
    X = combined.X
    if sp.issparse(X):
        sample = X.data[:10000] if X.nnz else np.array([])
    else:
        sample = np.asarray(X).ravel()[:10000]
    if sample.size and not np.allclose(sample, np.round(sample)):
        print("WARNING: X does not look like integer raw counts. "
              "Set sc_schema.raw_counts_layer if your raw counts live in a layer.")

    out_path = cfg.paths.combined_sc_h5ad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting combined: {out_path}")
    print(f"  shape: {combined.shape}")
    print(f"  cell.type.for.deconv distribution:")
    print(combined.obs[CTKEY].value_counts().to_string())
    combined.write_h5ad(out_path)
    print(f"\nDone. {len(stats)} files merged.")


if __name__ == "__main__":
    main()
