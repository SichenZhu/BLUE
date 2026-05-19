"""step03 -- multiple-testing filter and consensus clustering of definitions.

Reads the per-definition survival p-values (from step02), applies a
Benjamini-Hochberg FDR cut, drops definitions whose smaller patient group is too
small, then links the survivors by patient-set IoU into a graph. Each connected
component is collapsed (orientation-aligned majority vote) into one consensus
binary patient grouping.

Input  (paths.output_dir): <n>/TCGA_patient_cluster_ct<n>.csv  and
                           <n>/TCGA_patient_cluster_ct<n>_pvals.txt   (n = 1..6)
Output (paths.output_dir): tcga_new_cluster.csv   (consumed by step04)
With --plots: IoU-histogram and consensus-graph PNGs saved to paths.output_dir.

Usage:  uv run python scripts/aml_followup/step03_multiple_hypothesis_testing.py \
            --config config/followup_config_aml.yaml [--plots]
"""
from pathlib import Path
from copy import deepcopy
import argparse
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import false_discovery_control

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: config/followup_config_aml.yaml)")
    p.add_argument("--plots", action="store_true",
                   help="save the IoU-histogram and consensus-graph PNGs and display them")
    return p.parse_args()


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir) + "/"
    ht = cfg.hypothesis_testing

    all_definitions = None
    for ct in range(1, 7):
        definitions = pd.read_csv(output_path + "{}/TCGA_patient_cluster_ct{}.csv".format(ct, ct), index_col=0)
        if all_definitions is None:
            all_definitions = definitions
        else:
            all_definitions = pd.concat(
                [all_definitions, definitions.loc[:, ~definitions.columns.isin(["os", "osi"])]], axis=1)

    all_pval_lst, all_group_names, all_group_idx = [], [], []
    all_patients1, all_patients2 = [], []
    all_patients_num1, all_patients_num2 = [], []
    for i in range(1, 7):
        pvals = np.genfromtxt(output_path + "{}/TCGA_patient_cluster_ct{}_pvals.txt".format(str(i), str(i))).tolist()
        groupnames = pd.read_csv(output_path + "{}/TCGA_patient_cluster_ct{}.csv".format(str(i), str(i)),
                                 sep=",", header=0, index_col=0)
        for gname in groupnames.columns[2:]:
            p1_lst = sorted(list(groupnames[groupnames[gname] == 1].index))
            p2_lst = sorted(list(groupnames[groupnames[gname] == 0].index))
            patient_group1 = ",".join(p1_lst)
            patient_group2 = ",".join(p2_lst)
            if len(p1_lst) <= len(p2_lst):
                all_patients1.append(patient_group1)
                all_patients2.append(patient_group2)
                all_patients_num1.append(len(p1_lst))
                all_patients_num2.append(len(p2_lst))
            else:
                all_patients1.append(patient_group2)
                all_patients2.append(patient_group1)
                all_patients_num1.append(len(p2_lst))
                all_patients_num2.append(len(p1_lst))
        all_pval_lst.extend(pvals)
        all_group_names.extend(list(groupnames.columns[2:]))
        all_group_idx.extend([str(i) + "_" + str(j) for j in range(1, len(pvals) + 1)])

    pval_df = pd.DataFrame({"survp": all_pval_lst,
                            "groupname": all_group_names,
                            "groupidx": all_group_idx,
                            "group1_num": all_patients_num1,
                            "group2_num": all_patients_num2,
                            "patientID_in_group1": all_patients1,
                            "patientID_in_group2": all_patients2}).sort_values(by=["survp"], axis=0)

    pval_df_grouped = pval_df.groupby(["patientID_in_group1", "patientID_in_group2"], as_index=False).agg(list)
    for idx in range(pval_df_grouped.shape[0]):
        survp = pval_df_grouped["survp"][idx]
        if len(set(survp)) > 1:
            print("survp is not consistent for patient group: {}".format(idx))

    pval_df_grouped["p_val"] = pval_df_grouped["survp"].apply(lambda x: x[0])
    pval_df_grouped["p_val_ranking"] = pval_df_grouped["p_val"].rank(method="min", ascending=True).astype(int)
    pval_df_grouped["p_val_bh"] = false_discovery_control(pval_df_grouped["p_val"], method="bh")
    print("# definitions with BH-FDR < {}:".format(ht.fdr_rate),
          pval_df_grouped[pval_df_grouped["p_val_bh"] < ht.fdr_rate].shape)

    pval_df_grouped_significant = pval_df_grouped[(pval_df_grouped["p_val_bh"] < ht.fdr_rate)]
    pval_df_grouped_significant = pval_df_grouped_significant[
        pval_df_grouped_significant["group1_num"].apply(lambda x: x[0] > ht.min_group_size)]

    groups_v2 = pval_df_grouped_significant.copy()
    groups_v2.reset_index(drop=True, inplace=True)

    # IoU between every pair of significant definitions.
    adj_mtx = np.zeros(shape=(groups_v2.shape[0], groups_v2.shape[0]))
    for i in range(groups_v2.shape[0]):
        A1 = groups_v2.iloc[i, :]["patientID_in_group1"].split(",")
        A2 = groups_v2.iloc[i, :]["patientID_in_group2"].split(",")
        if len(A1) > len(A2):
            print("Something is wrong in patientID groups.")
        for j in range(i, groups_v2.shape[0]):
            B1 = groups_v2.iloc[j, :]["patientID_in_group1"].split(",")
            B2 = groups_v2.iloc[j, :]["patientID_in_group2"].split(",")
            if len(B1) > len(B2):
                print("Something is wrong in patientID groups.")
            iou1 = len(set(A1).intersection(set(B1))) / len(set(A1).union(set(B1)))
            iou2 = len(set(A1).intersection(set(B2))) / len(set(A1).union(set(B2)))
            adj_mtx[i, j] = max(iou1, iou2)
    for i in range(adj_mtx.shape[0]):
        for j in range(i):
            adj_mtx[i, j] = adj_mtx[j, i]

    # Binary adjacency: link two definitions when IoU >= threshold.
    adj_mtx_new = np.copy(adj_mtx)
    adj_mtx_new[adj_mtx_new < ht.iou_threshold] = 0
    adj_mtx_new[adj_mtx_new >= ht.iou_threshold] = 1
    for i in range(adj_mtx_new.shape[0]):
        adj_mtx_new[i, i] = 0
    print("adjacency symmetric:", check_symmetric(adj_mtx_new))

    # Consensus graph. Only edges are added, so definitions with no IoU >=
    # threshold neighbour (isolated nodes) are intentionally excluded.
    rows, cols = np.where(adj_mtx_new == 1)
    graph = nx.Graph()
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))

    if args.plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(adj_mtx.flatten(), log=True)
        plt.xlabel("IoU"); plt.ylabel("count (log scale)"); plt.title("definition-pair IoU")
        plt.savefig(output_dir / "iou_histogram.png", dpi=150, bbox_inches="tight")
        plt.figure()
        nx.draw(graph, node_size=10)
        plt.title("consensus graph")
        plt.savefig(output_dir / "consensus_graph.png", dpi=150, bbox_inches="tight")
        print(f"wrote plot PNGs to {output_dir}")
        plt.show()

    cluster_lst = []
    for c in nx.connected_components(graph):
        print(c)
        cluster_lst.append(list(c))

    all_groups_pd = all_definitions.copy()
    consensus_patients_groups = {}
    for c_idx in range(len(cluster_lst)):
        v_ = cluster_lst[c_idx]
        print("########## {} ##############".format(c_idx))

        ref_v_idx = groups_v2.loc[v_, "p_val_bh"].idxmin()
        v = deepcopy(v_)
        v.remove(ref_v_idx)

        curr_group1 = np.array(all_groups_pd[groups_v2.loc[ref_v_idx, "groupname"][0]])
        curr_group2 = np.abs(curr_group1 - 1)  # flip group assignment
        # align every other definition to each orientation of the reference
        similar_groups_1 = [curr_group1]
        similar_groups_2 = [curr_group2]
        for v_idx in range(len(v)):
            other_group1 = np.array(all_groups_pd[groups_v2.loc[v[v_idx], "groupname"][0]])
            other_group2 = np.abs(other_group1 - 1)  # flip group assignment
            distance11 = np.sum((curr_group1 + other_group1) % 2)
            distance12 = np.sum((curr_group1 + other_group2) % 2)
            distance21 = np.sum((curr_group2 + other_group1) % 2)
            distance22 = np.sum((curr_group2 + other_group2) % 2)
            if distance11 < distance12:
                similar_groups_1.append(other_group1)
            else:
                similar_groups_1.append(other_group2)
            if distance21 < distance22:
                similar_groups_2.append(other_group1)
            else:
                similar_groups_2.append(other_group2)

        similar_groups_assignment_1 = np.sum(np.array(similar_groups_1), axis=0)
        similar_groups_assignment_2 = np.sum(np.array(similar_groups_2), axis=0)
        consensus_patients_1 = all_groups_pd.index[np.where(similar_groups_assignment_1 > np.floor(len(v_) / 2))]
        consensus_patients_2 = all_groups_pd.index[np.where(similar_groups_assignment_2 > np.floor(len(v_) / 2))]

        if len(consensus_patients_1) > len(consensus_patients_2):
            consensus_patients = list(consensus_patients_1)
            similar_groups_assignment_flipped = np.sum(np.abs(np.array(similar_groups_1) - 1), axis=0)
            other_patients = list(all_groups_pd.index[np.where(similar_groups_assignment_flipped > np.floor(len(v_) / 2))])
            print("Assignment 1 wins. # consensus_patients {}, {}".format(len(consensus_patients), len(other_patients)))
            print("Pass sanity check: {}".format(set(consensus_patients).intersection(set(other_patients)) == set()))
        elif len(consensus_patients_1) < len(consensus_patients_2):
            consensus_patients = list(consensus_patients_2)
            similar_groups_assignment_flipped = np.sum(np.abs(np.array(similar_groups_2) - 1), axis=0)
            other_patients = list(all_groups_pd.index[np.where(similar_groups_assignment_flipped > np.floor(len(v_) / 2))])
            print("Assignment 2 wins. # consensus_patients {}, {}".format(len(consensus_patients), len(other_patients)))
            print("Pass sanity check: {}".format(set(consensus_patients).intersection(set(other_patients)) == set()))
        else:
            print("Two group assignment result in same number of consensus patients")
            consensus_patients = [list(consensus_patients_1), list(consensus_patients_2)]
            similar_groups_assignment_flipped_1 = np.sum(np.abs(np.array(similar_groups_1) - 1), axis=0)
            similar_groups_assignment_flipped_2 = np.sum(np.abs(np.array(similar_groups_2) - 1), axis=0)
            other_patients_1 = list(all_groups_pd.index[np.where(similar_groups_assignment_flipped_1 > np.floor(len(v_) / 2))])
            other_patients_2 = list(all_groups_pd.index[np.where(similar_groups_assignment_flipped_2 > np.floor(len(v_) / 2))])
            other_patients = [other_patients_1, other_patients_2]
            print("2 assignments equal (tie). Kept both labelings; consensus side size {}, complements {} / {}".format(
                len(consensus_patients_1), len(other_patients_1), len(other_patients_2)))

        consensus_patients_groups[c_idx] = {"group1": consensus_patients, "group2": other_patients}
        print("########################\n\n")

    patient_info_new = all_definitions.loc[:, ["os", "osi"]].copy()
    for k, d in consensus_patients_groups.items():
        g1, g2 = d["group1"], d["group2"]
        if g1 and isinstance(g1[0], list):
            # Tie component: the two consensus sides are the same size, so there
            # is no size rule to pick which one is group1. Keep BOTH candidate
            # labelings as separate columns (new_cluster_<k>_opt1 / _opt2); the
            # choice between them is resolved manually in step03.
            for opt, (g1_opt, g2_opt) in enumerate(zip(g1, g2), start=1):
                new_colname = "new_cluster_{}_opt{}".format(k, opt)
                patient_info_new[new_colname] = 0
                patient_info_new.loc[g1_opt, new_colname] = 1
                patient_info_new.loc[g2_opt, new_colname] = 2
        else:
            new_colname = "new_cluster_" + str(k)
            patient_info_new[new_colname] = 0
            patient_info_new.loc[g1, new_colname] = 1
            patient_info_new.loc[g2, new_colname] = 2

    out_csv = output_dir / "tcga_new_cluster.csv"
    patient_info_new.to_csv(out_csv, sep=",", header=True, index=True)
    print(f"wrote {out_csv}  ({len(consensus_patients_groups)} consensus clusters)")


if __name__ == "__main__":
    main()
