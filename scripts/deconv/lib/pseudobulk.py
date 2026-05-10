"""Pseudobulk generation (ported from ori_deconv_simulation.py).

Algorithm:
- Dirichlet proportions: one uniform block (α = ``uniform_alpha`` for every
  cell type) plus one dominant block per α in ``dominant_alphas`` (where one
  cell type gets that high α and the rest stay at 1, repeated across cell
  types). All blocks are concatenated and shuffled.
- cell_num = ceil(n_cells * prop); sample that many cells (with replacement)
  of each type, sum raw counts to form the pseudobulk, average to form ctGEP.
- Output is an AnnData with X = pseudobulk averaged by total n_cells,
  uns['cellType_prop'], uns['cellType_cellNum'], and one layer per cell type.

Data-structure change: takes a sparse cell×gene matrix + cell_type array +
a per-cell-type dict of row indices. This avoids materializing a dense
DataFrame of the full reference.
"""
from __future__ import annotations
from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad


DEFAULT_UNIFORM_ALPHA = 1.0
DEFAULT_DOMINANT_ALPHAS: tuple[float, ...] = (50.0, 15.0, 5.0)


def split_train_val_test_cells(
    cell_types: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of cell indices by cell type.

    Matches original logic: train = floor(ratio·n); val = n - train;
    test = 0 when val_ratio leaves no remainder. So default is 80/20/0.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    unique_cts = sorted(np.unique(cell_types).tolist())
    for ct in unique_cts:
        idx = np.where(cell_types == ct)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(train_ratio * n)
        n_val = n - n_tr
        n_test = n - n_tr - n_val  # = 0 in the default config
        train.extend(idx[:n_tr].tolist())
        val.extend(idx[n_tr : n_tr + n_val].tolist())
        test.extend(idx[n_tr + n_val :].tolist())
        print(f"  {ct}: {n} cells → train {n_tr} | val {n_val} | test {n_test}")
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _dirichlet_props(
    num_celltype: int,
    samplenum: int,
    seed: int,
    uniform_alpha: float = DEFAULT_UNIFORM_ALPHA,
    dominant_alphas: Iterable[float] = DEFAULT_DOMINANT_ALPHAS,
) -> np.ndarray:
    """Concat of one uniform block (α = ``uniform_alpha`` everywhere) plus
    one dominant block per α in ``dominant_alphas`` (one cell type gets that
    α, the rest stay at 1, repeated across cell types). Then in-place shuffle.
    """
    rng = np.random.default_rng(seed)

    # base uniform
    prop = rng.dirichlet(np.full(num_celltype, uniform_alpha), samplenum)

    dSampleNum = ((samplenum // num_celltype) // 100) * 100
    if dSampleNum == 0:
        dSampleNum = ((samplenum // num_celltype) // 10) * 10
        if dSampleNum == 0:
            dSampleNum = 10

    for alpha_val in dominant_alphas:
        block = None
        for dct_i in range(num_celltype):
            alpha = np.ones(num_celltype)
            alpha[dct_i] = alpha_val
            part = rng.dirichlet(alpha, dSampleNum)
            block = part if block is None else np.vstack((block, part))
        prop = np.vstack((prop, block))

    rng.shuffle(prop)
    return prop


def generate_pseudobulk(
    X: sp.csr_matrix,
    cell_types: np.ndarray,
    row_indices: np.ndarray,
    gene_names: list[str],
    desired_ct_lst: list[str],
    samplenum: int,
    n_cells: int = 500,
    seed: int = 42,
    uniform_alpha: float = DEFAULT_UNIFORM_ALPHA,
    dominant_alphas: Iterable[float] = DEFAULT_DOMINANT_ALPHAS,
) -> ad.AnnData:
    """Generate pseudobulks from a sparse reference.

    Parameters
    ----------
    X : csr (N_all_cells, G) raw counts for the whole reference.
    cell_types : length-N_all_cells cell-type labels (same order as X).
    row_indices : indices (into X / cell_types) of cells eligible for sampling.
    gene_names : G-length list of var names.
    desired_ct_lst : full cell-type vocabulary (cell types absent from row_indices
                     are padded with zero proportions / zero ctGEP layers).
    """
    rng = np.random.default_rng(seed)

    subset_cts = cell_types[row_indices]
    cts_present = sorted(set(subset_cts.tolist()) & set(desired_ct_lst))
    num_ct = len(cts_present)
    if num_ct == 0:
        raise ValueError("No desired cell types present in row_indices.")

    # per-CT row pool (into X)
    ct_pool: dict[str, np.ndarray] = {
        ct: row_indices[subset_cts == ct] for ct in cts_present
    }

    prop = _dirichlet_props(
        num_ct, samplenum, seed=seed,
        uniform_alpha=uniform_alpha, dominant_alphas=dominant_alphas,
    )
    cell_num = np.ceil(n_cells * prop).astype(np.int64)  # (total_samples, num_ct)
    total_cells_per_sample = cell_num.sum(axis=1)

    total_samples = cell_num.shape[0]
    G = X.shape[1]

    sample_sum = np.zeros((total_samples, G), dtype=np.float64)
    ct_gep: dict[str, np.ndarray] = {ct: np.zeros((total_samples, G), dtype=np.float64) for ct in cts_present}

    print(f"  sampling {total_samples} pseudobulks × {n_cells} cells (avg)…")
    for i in range(total_samples):
        if total_samples > 500 and i % max(1, total_samples // 10) == 0:
            print(f"    {i}/{total_samples}")
        for j, ct in enumerate(cts_present):
            k = int(cell_num[i, j])
            if k == 0:
                continue
            sel = rng.choice(ct_pool[ct], size=k, replace=True)
            s = np.asarray(X[sel].sum(axis=0)).ravel()
            sample_sum[i] += s
            ct_gep[ct][i] = s / k

    sample_norm = sample_sum / total_cells_per_sample[:, None]
    prop_df = pd.DataFrame(
        cell_num / total_cells_per_sample[:, None], columns=cts_present
    )
    cellnum_df = pd.DataFrame(cell_num, columns=cts_present)

    missing = [ct for ct in desired_ct_lst if ct not in cts_present]
    for mct in missing:
        prop_df[mct] = 0.0
        cellnum_df[mct] = 0
    prop_df = prop_df[desired_ct_lst]
    cellnum_df = cellnum_df[desired_ct_lst]

    adata = ad.AnnData(X=sample_norm.astype(np.float32))
    adata.obs_names = [str(i) for i in range(total_samples)]
    adata.var_names = list(gene_names)
    adata.uns["cellType_prop"] = prop_df
    adata.uns["cellType_cellNum"] = cellnum_df
    for ct in cts_present:
        adata.layers[ct] = ct_gep[ct].astype(np.float32)
    for mct in missing:
        adata.layers[mct] = np.zeros_like(adata.X)

    return adata
