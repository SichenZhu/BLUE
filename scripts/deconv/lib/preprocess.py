"""Bulk / pseudobulk preprocessing: libsize norm + upscale 1e6 + log2(+1).

Ported from ori_deconv_algo.py:preprocessing_bulk.
Same transform is applied to both real bulk and pseudobulk so the UNet
sees them on identical scale.
"""
from __future__ import annotations
import numpy as np


def preprocess_counts(
    counts: np.ndarray,
    up_scaling_coeff: float = 1e6,
) -> np.ndarray:
    """Per-row libsize-normalize → *up_scaling_coeff → log2(+1).

    counts: (n_samples, n_genes) non-negative array. Rows that are all-zero
    are left as zeros (log2(1)=0).
    """
    x = np.asarray(counts, dtype=np.float64)
    x = np.where(x < 0, 0.0, x)

    row_sum = x.sum(axis=1, keepdims=True)
    nz = row_sum.squeeze(-1) > 0
    x[nz] = x[nz] / row_sum[nz]
    x = x * up_scaling_coeff
    x = np.log2(x + 1.0)
    return x.astype(np.float32)


def align_to_gene_list(
    mat: np.ndarray,
    src_genes: list[str],
    target_genes: list[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Reorder/subset columns of `mat` to `target_genes`; missing → fill_value."""
    src_idx = {g: i for i, g in enumerate(src_genes)}
    n_samples = mat.shape[0]
    out = np.full((n_samples, len(target_genes)), fill_value, dtype=mat.dtype)
    for j, g in enumerate(target_genes):
        i = src_idx.get(g)
        if i is not None:
            out[:, j] = mat[:, i]
    return out
