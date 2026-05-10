"""Per-cell-type DEG utilities (ported from ori_deconv_simulation.py).

Unchanged logic: rank_genes_groups(method='wilcoxon', reference='rest'),
FDR filter, positive logFC, greedy per-ct selection avoiding duplicates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

CTKEY_DEFAULT = "cell.type.for.deconv"
DEG_KEY = "DEGs_ref_rest"
NORM_LAYER = "norm_log_b4scale"


def preprocess_for_degs(
    adata: ad.AnnData,
    groupby: str = CTKEY_DEFAULT,
    desired_ct_lst: list | None = None,
    min_cells_per_ct: int = 2,
) -> ad.AnnData:
    """Subset to cell types with >= min_cells_per_ct cells, normalize+log, store layer."""
    a = adata.copy()
    if desired_ct_lst is not None:
        a = a[a.obs[groupby].isin(desired_ct_lst)].copy()

    ct_counts = a.obs[groupby].value_counts()
    keep_ct = ct_counts[ct_counts >= min_cells_per_ct].index.tolist()
    a = a[a.obs[groupby].isin(keep_ct)].copy()
    a.obs[groupby] = a.obs[groupby].astype(str)

    sc.pp.filter_genes(a, min_cells=1)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.layers[NORM_LAYER] = a.X.copy()
    a.uns.setdefault("log1p", {})["base"] = np.e
    return a


def rank_genes_one_vs_rest(adata: ad.AnnData, groupby: str = CTKEY_DEFAULT) -> None:
    """Wilcoxon rank_genes_groups vs. 'rest' on norm_log_b4scale layer."""
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        use_raw=False,
        layer=NORM_LAYER,
        reference="rest",
        method="wilcoxon",
        key_added=DEG_KEY,
    )


def find_DEGs(
    adata_DEG: ad.AnnData,
    groupby: str = CTKEY_DEFAULT,
    num_DEG_per_ct: int = 6000,
    fdr_rate: float = 0.1,
) -> set[str]:
    """Select up to num_DEG_per_ct positive-logFC DEGs per cell type (FDR filtered),
    in ascending-count order, skipping already-picked genes — returns the union.
    """
    pos_deg_by_ct: dict[str, pd.DataFrame] = {}
    ct_in_adata = sorted(adata_DEG.obs[groupby].astype(str).unique())

    for ct in ct_in_adata:
        names = adata_DEG.uns[DEG_KEY]["names"][ct]
        padj = adata_DEG.uns[DEG_KEY]["pvals_adj"][ct]
        lfc = adata_DEG.uns[DEG_KEY]["logfoldchanges"][ct]
        df = (
            pd.DataFrame({"gene": names, "pvals_adj": padj, "logfc": lfc})
            .sort_values("pvals_adj")
        )
        df_fdr = df[df["pvals_adj"] < fdr_rate]
        pos_deg_by_ct[ct] = df_fdr[df_fdr["logfc"] > 0].copy()

    ct_order = (
        pd.Series({k: v.shape[0] for k, v in pos_deg_by_ct.items()})
        .sort_values()
        .index.tolist()
    )

    gene_set: set[str] = set()
    first = True
    for ct in ct_order:
        v_rank = (
            pos_deg_by_ct[ct]
            .sort_values("logfc", ascending=False)
            .reset_index(drop=True)
        )
        if first:
            gene_set = set(v_rank.loc[: num_DEG_per_ct - 1, "gene"].tolist())
            first = False
        else:
            picked = []
            i = 0
            while len(picked) < num_DEG_per_ct and i < v_rank.shape[0]:
                g = v_rank.loc[i, "gene"]
                if g not in gene_set:
                    picked.append(g)
                i += 1
            gene_set |= set(picked)
        print(f"  {ct}: n_pos_DEG={pos_deg_by_ct[ct].shape[0]}, union_size={len(gene_set)}")

    return gene_set
