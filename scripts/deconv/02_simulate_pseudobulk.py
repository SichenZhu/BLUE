"""Simulate pseudobulks from the combined sc reference (config-driven).

Per-patient: train_samplenum = SAMPLENUM_PER_CT * num_ct, val_samplenum = VAL_PER_PATIENT.
All-patients: train_samplenum = SAMPLENUM_ALL_TRAIN, val_samplenum = SAMPLENUM_ALL_VAL.

`held_out_patients` mode in 04_train_unet.py uses only the per-patient files,
so the all_*.h5ad files become optional in that scenario but are still produced
so either training mode can be tried.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402
from lib.pseudobulk import split_train_val_test_cells, generate_pseudobulk  # noqa: E402

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
    pb_cfg = cfg.pseudobulk

    # Dirichlet alphas — fall back to library defaults if not in config (backward-compat).
    from lib.pseudobulk import DEFAULT_UNIFORM_ALPHA, DEFAULT_DOMINANT_ALPHAS  # noqa: E402
    uniform_alpha   = getattr(pb_cfg, "uniform_alpha",   DEFAULT_UNIFORM_ALPHA)
    dominant_alphas = getattr(pb_cfg, "dominant_alphas", list(DEFAULT_DOMINANT_ALPHAS))
    print(f"Dirichlet α: uniform={uniform_alpha}  dominant={list(dominant_alphas)}")

    common_genes_path = cfg.paths.gene_lists_dir / "all_common_gene_lst.txt"
    common_genes = [g.strip() for g in common_genes_path.read_text().splitlines() if g.strip()]
    print(f"Common genes: {len(common_genes)}")

    print(f"Loading {cfg.paths.combined_sc_h5ad} …")
    a = ad.read_h5ad(cfg.paths.combined_sc_h5ad)
    print(f"  full: {a.shape}")
    a = a[:, common_genes].copy()
    a = a[a.obs[CTKEY].astype(str).isin(ctlist)].copy()
    print(f"  after gene + CTLIST filter: {a.shape}")

    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.tocsr().astype(np.float32)
    cell_types = a.obs[CTKEY].astype(str).to_numpy()
    library_ids = a.obs[cfg.sc_schema.library_id_col].astype(str).to_numpy()
    gene_names = a.var_names.tolist()

    out_dir = cfg.paths.pseudobulk_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_lib_train_idx: dict[str, np.ndarray] = {}
    per_lib_val_idx: dict[str, np.ndarray] = {}

    libs = sorted(np.unique(library_ids).tolist())
    print(f"\nPer-patient pseudobulks across {len(libs)} libraries")

    for i, lib in enumerate(libs, 1):
        print(f"\n[{i:2d}/{len(libs)}] {lib}")
        lib_mask = library_ids == lib
        lib_row_indices = np.where(lib_mask)[0]
        lib_cts = cell_types[lib_row_indices]
        num_ct = len(set(lib_cts.tolist()) & set(ctlist))
        if num_ct < 2:
            print(f"  skip: only {num_ct} cell type present")
            continue

        local_train, local_val, _ = split_train_val_test_cells(
            lib_cts, train_ratio=pb_cfg.train_ratio, val_ratio=pb_cfg.val_ratio, seed=pb_cfg.seed,
        )
        train_idx = lib_row_indices[local_train]
        val_idx = lib_row_indices[local_val]
        per_lib_train_idx[lib] = train_idx
        per_lib_val_idx[lib] = val_idx

        train_bulk = generate_pseudobulk(
            X, cell_types, train_idx, gene_names,
            desired_ct_lst=ctlist,
            samplenum=pb_cfg.samplenum_per_ct * num_ct,
            n_cells=pb_cfg.n_cells, seed=pb_cfg.seed + i,
            uniform_alpha=uniform_alpha, dominant_alphas=dominant_alphas,
        )
        train_bulk.write_h5ad(out_dir / f"{lib}_trainbulk.h5ad")
        print(f"  wrote {lib}_trainbulk.h5ad : {train_bulk.shape}")

        if len(val_idx) >= num_ct:
            val_bulk = generate_pseudobulk(
                X, cell_types, val_idx, gene_names,
                desired_ct_lst=ctlist,
                samplenum=pb_cfg.val_samplenum_per_patient,
                n_cells=pb_cfg.n_cells, seed=pb_cfg.seed + i + 1,
                uniform_alpha=uniform_alpha, dominant_alphas=dominant_alphas,
            )
            val_bulk.write_h5ad(out_dir / f"{lib}_valbulk.h5ad")
            print(f"  wrote {lib}_valbulk.h5ad   : {val_bulk.shape}")

    print("\n\nAll-patients pseudobulks")
    all_train_idx = np.concatenate(list(per_lib_train_idx.values()))
    all_val_idx = np.concatenate(list(per_lib_val_idx.values()))
    print(f"  train pool: {len(all_train_idx)}  val pool: {len(all_val_idx)}")

    all_train = generate_pseudobulk(
        X, cell_types, all_train_idx, gene_names,
        desired_ct_lst=ctlist,
        samplenum=pb_cfg.samplenum_all_train,
        n_cells=pb_cfg.n_cells, seed=pb_cfg.seed,
        uniform_alpha=uniform_alpha, dominant_alphas=dominant_alphas,
    )
    all_train.write_h5ad(out_dir / "all_trainbulk.h5ad")
    print(f"  wrote all_trainbulk.h5ad : {all_train.shape}")

    all_val = generate_pseudobulk(
        X, cell_types, all_val_idx, gene_names,
        desired_ct_lst=ctlist,
        samplenum=pb_cfg.samplenum_all_val,
        n_cells=pb_cfg.n_cells, seed=pb_cfg.seed + 1,
        uniform_alpha=uniform_alpha, dominant_alphas=dominant_alphas,
    )
    all_val.write_h5ad(out_dir / "all_valbulk.h5ad")
    print(f"  wrote all_valbulk.h5ad   : {all_val.shape}")


if __name__ == "__main__":
    main()
