"""step01 -- enumerate binary patient-group definitions from the cell states.

For each cell type, step00's Leiden clusters are paired (config: `group_pairs`).
For every combination of 1-6 cell types and every choice of one cluster per
selected cell type, the patients shared by those clusters form group 1 (the rest
group 0) -- one binary "definition". Definitions are written grouped by the
number of cell types involved.

Input  (paths.output_dir): all_predCTGEP.h5ad  (from step00)
Input  (paths.input_dir):  the patient-metadata CSV (`os`, `osi` columns)
Output (paths.output_dir): <n>/TCGA_patient_cluster_*.csv  for n = 1..6

Usage:  uv run python scripts/aml_followup/step01_define_patient_group_by_cell_states.py \
            --config config/followup_config_aml.yaml
"""
from pathlib import Path
import argparse
import itertools
import sys

import anndata
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: config/followup_config_aml.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    input_dir = cfg.paths.input_dir
    output_dir = cfg.paths.output_dir
    for nc in range(1, 7):                     # per-num_ct subdirs 1/ ... 6/
        (output_dir / str(nc)).mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir) + "/"

    ctlist = cfg.celltypes
    group_pairs = [list(p) for p in cfg.group_pairs]

    all_predGEP_adata = anndata.read_h5ad(output_dir / "all_predCTGEP.h5ad")
    patient_info = pd.read_csv(input_dir / cfg.patient_info_csv,
                               sep=",", header=0, index_col=0)

    # Patients in each Leiden cluster of every pair.
    cluster_dic = {}
    for group in group_pairs:
        cluster_dic["_".join(group)] = [
            all_predGEP_adata[all_predGEP_adata.obs["leiden"] == group[0], :].obs["patientID"].tolist(),
            all_predGEP_adata[all_predGEP_adata.obs["leiden"] == group[1], :].obs["patientID"].tolist(),
        ]

    # --- 1 cell type ------------------------------------------------------
    ct1_patient_info = patient_info.loc[:, ["os", "osi"]].copy()
    for i, group in enumerate(group_pairs):
        col_to_add = pd.DataFrame(0, index=patient_info.index, columns=["1ct_" + group[0]])
        sample_dup = list(set(cluster_dic["_".join(group)][0]).intersection(cluster_dic["_".join(group)][1]))
        col_to_add.loc[cluster_dic["_".join(group)][0], :] = 1
        # A patient's GEP point can land in both clusters of a pair; if so, keep
        # the assignment from this cell type's own cluster.
        for s in sample_dup:
            if all_predGEP_adata[s + "_" + ctlist[i]].obs["leiden"].iloc[0] == group[1]:
                col_to_add.loc[s, :] = 0
        ct1_patient_info.loc[:, ["1ct_" + group[0]]] = col_to_add

    for i in range(2, ct1_patient_info.shape[1]):
        df_tosave = ct1_patient_info.iloc[:, [0, 1, i]]
        df_tosave.columns = ["os", "osi", "cluster"]
        df_tosave.to_csv(output_path + "1/TCGA_patient_cluster_{}.csv".format(i-1),
                         sep=",", header=True, index=True)
    ct1_patient_info.to_csv(output_path + "1/TCGA_patient_cluster_ct1.csv",
                            sep=",", header=True, index=True)
    print(f"ct1: wrote {ct1_patient_info.shape[1] - 2} definitions")

    # --- 2 to 6 cell types ------------------------------------------------
    group_pairs_arr = np.array(group_pairs)
    for num_ct in range(2, 7):
        ct_patient_info = patient_info.loc[:, ["os", "osi"]].copy()
        for ct_comb in itertools.combinations(range(6), num_ct):
            groups = group_pairs_arr[list(ct_comb)]
            for g_comb in itertools.product(*groups):
                group1 = set(all_predGEP_adata[all_predGEP_adata.obs["leiden"] == g_comb[0], :].obs["patientID"].tolist())
                for i in range(1, num_ct):  # patients shared by this cluster combination
                    group1 = set(all_predGEP_adata[all_predGEP_adata.obs["leiden"] == g_comb[i], :].obs["patientID"].tolist()).intersection(group1)
                if not group1:
                    print("Group", g_comb, " is empty.")
                    continue
                group1 = sorted(list(group1))
                col_to_add = pd.DataFrame(0, index=patient_info.index,
                                          columns=[str(num_ct) + "ct_" + "-".join(g_comb)])
                col_to_add.loc[group1, :] = 1
                ct_patient_info.loc[:, list(col_to_add.columns)] = col_to_add

        for i in range(2, ct_patient_info.shape[1]):
            df_tosave = ct_patient_info.iloc[:, [0, 1, i]]
            df_tosave.columns = ["os", "osi", "cluster"]
            df_tosave.to_csv(output_path + "{}/TCGA_patient_cluster_{}.csv".format(num_ct, i-1),
                             sep=",", header=True, index=True)
        ct_patient_info.to_csv(output_path + str(num_ct) + "/TCGA_patient_cluster_ct{}.csv".format(str(num_ct)),
                               sep=",", header=True, index=True)
        print(f"ct{num_ct}: wrote {ct_patient_info.shape[1] - 2} definitions")


if __name__ == "__main__":
    main()
