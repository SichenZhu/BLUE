"""step00 -- cluster BLUE's per-cell-type predicted-GEP "cell states".

Reads BLUE's per-cell-type predicted gene-expression profiles, stacks them into a
single AnnData where each observation is one (patient x cell type) GEP vector,
then runs PCA -> neighbors -> UMAP -> Leiden to cluster those cell states.

The per-cell-type GEP is read in one of following two formats (config `gep_h5ad`):
  * gep_h5ad null  -> per-cell-type GEP_<celltype>.txt files in `input_dir`.
  * gep_h5ad set   -> a single AnnData with one layer per cell type -- the
                      deconvolution pipeline's predicted_ctGEP_ep<N>.h5ad output.

Input  (paths.input_dir):  GEP_<ct>.txt for every cell type in `celltypes` (or a
                           predicted_ctGEP h5ad named by `gep_h5ad`); the
                           patient-metadata CSV (`patient_info_csv`).
Output (paths.output_dir): all_predCTGEP.h5ad  (consumed by step01).
With --plots: UMAP PNGs saved to paths.output_dir and displayed.

Usage:  uv run python scripts/aml_followup/step00_define_cell_states_predctGEP.py \
            --config config/followup_config_aml.yaml [--plots]
"""
from pathlib import Path
import argparse
import sys

import anndata
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: config/followup_config_aml.yaml)")
    p.add_argument("--plots", action="store_true",
                   help="save UMAP PNGs to the output dir and display them")
    return p.parse_args()


def load_per_celltype_gep(ctlist, input_dir, gep_h5ad):
    """Load BLUE's predicted per-cell-type GEP as {celltype: DataFrame(patients x genes)}.

    Two input formats:
      * gep_h5ad is None   -> read per-cell-type GEP_<celltype>.txt from input_dir.
      * gep_h5ad is a path -> read a single AnnData (the deconvolution pipeline's
        predicted_ctGEP_ep<N>.h5ad) and take one layer per cell type. A relative
        path is resolved against input_dir.
    """
    if not gep_h5ad:
        return {ct: pd.read_csv(input_dir / f"GEP_{ct}.txt", sep="\t", header=0, index_col=0)
                for ct in ctlist}

    gep_path = Path(gep_h5ad)
    if not gep_path.is_absolute():
        gep_path = input_dir / gep_path
    adata = anndata.read_h5ad(gep_path)
    missing = [ct for ct in ctlist if ct not in adata.layers]
    if missing:
        raise KeyError(f"{gep_path}: no layer for cell type(s) {missing}; "
                       f"available layers: {list(adata.layers)}")
    out = {}
    for ct in ctlist:
        layer = adata.layers[ct]
        mat = layer.toarray() if issparse(layer) else layer
        # float64 to match the text-file path (pd.read_csv yields float64)
        out[ct] = pd.DataFrame(mat, index=adata.obs_names,
                               columns=adata.var_names).astype("float64")
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = cfg.paths.input_dir
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ctlist = cfg.celltypes

    patient_info = pd.read_csv(input_dir / cfg.patient_info_csv,
                               sep=",", header=0, index_col=0)

    # One predicted-GEP table per cell type, restricted to the metadata patients.
    gep = load_per_celltype_gep(ctlist, input_dir, getattr(cfg, "gep_h5ad", None))
    predGEP_df_dic = {ct: df.loc[patient_info.index, :] for ct, df in gep.items()}

    # Stack all per-cell-type GEPs: each row is one (patient, cell type) state,
    # indexed "<patientID>_<celltype>".
    all_predGEP = None
    all_ct = None
    all_patients = None
    all_patient_clusters = None
    all_patient_clusters_paper = None
    first_key = True
    for k, v in predGEP_df_dic.items():
        v_new_index = [x + "_" + str(k) for x in v.index]
        if first_key:
            all_predGEP = v.copy()
            all_predGEP.index = v_new_index
            all_ct = pd.DataFrame(k, columns=["cell_type"], index=v_new_index)
            all_patients = pd.DataFrame(list(v.index), columns=["patientID"], index=v_new_index)
            all_patient_clusters = patient_info.loc[v.index, "Cluster"].copy()
            all_patient_clusters_paper = patient_info.loc[v.index, "paper_cluster"].copy()
            first_key = False
        else:
            v_ = v.loc[:, all_predGEP.columns].copy()
            v_.index = v_new_index
            all_predGEP = pd.concat([all_predGEP, v_], axis=0)
            all_ct = pd.concat([all_ct, pd.DataFrame(k, columns=["cell_type"], index=v_new_index)])
            all_patients = pd.concat([all_patients,
                                      pd.DataFrame(list(v.index), columns=["patientID"], index=v_new_index)])
            all_patient_clusters = pd.concat([all_patient_clusters,
                                              patient_info.loc[v.index, "Cluster"].copy()])
            all_patient_clusters_paper = pd.concat([all_patient_clusters_paper,
                                                    patient_info.loc[v.index, "paper_cluster"].copy()])
    all_patient_clusters.index = all_predGEP.index
    all_patient_clusters_paper.index = all_predGEP.index

    adata = anndata.AnnData(all_predGEP)
    adata.obs["cell_type"] = all_ct
    adata.obs["patientID"] = all_patients
    adata.obs["patient_clusters"] = all_patient_clusters
    adata.obs["paper_patient_clusters"] = all_patient_clusters_paper
    adata.layers["raw_prediction"] = adata.X

    cs = cfg.cell_states
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=cs.n_neighbors, n_pcs=cs.n_pcs)
    sc.tl.umap(adata, min_dist=cs.min_dist)
    sc.tl.leiden(adata, resolution=cs.leiden_resolution)
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

    out_h5ad = output_dir / "all_predCTGEP.h5ad"
    adata.write(out_h5ad)
    print(f"wrote {out_h5ad}  "
          f"({adata.n_obs} cell states, {adata.obs['leiden'].nunique()} Leiden clusters)")

    if args.plots:
        import matplotlib.pyplot as plt
        sc.pl.umap(adata, color=["leiden", "cell_type"], show=False)
        plt.savefig(output_dir / "umap_leiden_celltype.png", dpi=150, bbox_inches="tight")
        sc.pl.umap(adata, color=["patient_clusters", "paper_patient_clusters"], show=False)
        plt.savefig(output_dir / "umap_patient_clusters.png", dpi=150, bbox_inches="tight")
        print(f"wrote UMAP PNGs to {output_dir}")
        plt.show()


if __name__ == "__main__":
    main()
