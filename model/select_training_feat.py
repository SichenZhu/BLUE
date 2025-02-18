
import scanpy as sc
import numpy as np
import pandas as pd



def DEGenes_one_vs_one(adata, groupBy, ref_ct_str, layername=None):
    print("\t calculating cell type DEGs...")
    sc.tl.rank_genes_groups(adata, groupby=groupBy, use_raw=False, layer=layername, reference=ref_ct_str, method='wilcoxon', key_added="DEGs_ref_"+ref_ct_str)


def FDR_procedure(df, fdr_rate): # manual FDR procedure
    # if pvals_adj is in the anndata, then no need to apply manual FDR
    df.loc[:, "rank"] = np.arange(df.shape[0])
    df.loc[:, "fdr"] = df.loc[:, "rank"] * fdr_rate / df.shape[0]
    return df[df["pval"] <= df["fdr"]]



def find_DEGs(adata, ctlist, num_DEG_per_ct=245, fdr_rate=0.1, layername=None):
    adata.uns["log1p"]["base"] = np.e
    
    DEGenes_one_vs_one(adata, "celltype", "rest", layername=layername)
    
    print("\t Start selecting DEGs...")
    gene_after_FDR_posLogFC_dict = {}
    for ct in ctlist:
        adata_gene = adata.uns["DEGs_ref_rest"]["names"][ct]
        adata_pval = adata.uns["DEGs_ref_rest"]["pvals_adj"][ct]
        adata_logfc = adata.uns["DEGs_ref_rest"]["logfoldchanges"][ct]

        df = pd.DataFrame({"gene":adata_gene, "pvals_adj":adata_pval, "logfc": adata_logfc}).sort_values(by=["pvals_adj"])
        df_FDR = df[df["pvals_adj"] < fdr_rate]
        gene_after_FDR_posLogFC_dict[ct] = df_FDR[df_FDR["logfc"] > 0].copy()

    k_lst, v_lst = [], []
    v_min = float("inf")
    for k, v in gene_after_FDR_posLogFC_dict.items():
        k_lst.append(k)
        v_lst.append(v.shape[0])
        if v_lst[-1] < v_min:
            v_min = v.shape[0]
    df = pd.DataFrame({"key":k_lst, "n_gene":v_lst})
    ct_order = df.sort_values(by=["n_gene"])["key"].tolist()

    first = True
    gene_set = set()

    for ct in ct_order:
        v_rank = gene_after_FDR_posLogFC_dict[ct].sort_values(by=["logfc"], ascending=False).reset_index(drop=True)

        if first:
            gene_set = set(v_rank.loc[:num_DEG_per_ct-1, "gene"].tolist())
            print("  {} final genes selected {}/{}".format(ct, len(gene_set), v_rank.shape[0]))
            first = False
        else:
            ct_gene_lst = []
            i = 0
            while len(ct_gene_lst) < num_DEG_per_ct and i < v_rank.shape[0]:
                if v_rank.loc[i, "gene"] not in gene_set:
                    ct_gene_lst.append(v_rank.loc[i, "gene"])
                i += 1
            print("  {} final genes selected {}/{}".format(ct, i, v_rank.shape[0]))
            gene_set = gene_set.union(set(ct_gene_lst))
        print("Length of selected DEG sets: {}".format(len(gene_set)))

    return gene_set