"""step04 -- assemble the final 3 patient groups (best / worst / rest).

step03's consensus clusters (the new_cluster_<k> columns of tcga_new_cluster.csv)
are combined by hand into three survival groups. After inspecting each consensus
cluster's survival pattern, `final_groups.best_clusters` lists the clusters whose
"== 2" side is intersected into the best-survival group, and
`final_groups.worst_clusters` the clusters whose "== 2" side is unioned into the
worst-survival group; everyone else is "rest".

Input  (paths.output_dir): tcga_new_cluster.csv   (from step03)
Output (paths.output_dir): TCGA_new_3_groups.csv  (my_cluster: 0=rest, 1=worst, 2=best)
With --plots: Kaplan-Meier curve PNG saved to paths.output_dir and displayed
              (this needs scikit-survival installed; imported lazily).

Usage:  uv run python scripts/aml_followup/step04_def_final_patient_groups.py \
            --config config/followup_config_aml.yaml [--plots]
"""
from pathlib import Path
import argparse
import sys

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
    p.add_argument("--plots", action="store_true",
                   help="save the Kaplan-Meier curve PNG and display it "
                        "(requires scikit-survival)")
    return p.parse_args()


def plot_survival_curve(title, patient_groups, output_dir):
    """Kaplan-Meier curves per final group + log-rank p-value in the title.

    scikit-survival is imported here (not at module scope) so the rest of step03
    runs without it -- it is only needed for --plots.
    """
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.compare import compare_survival

    data_x = pd.DataFrame(patient_groups["my_cluster"])
    data_y = patient_groups.loc[:, "os"].to_frame()
    data_y["osi"] = patient_groups["osi"].replace({1: True, 0: False})

    plt.rcParams["figure.figsize"] = (5, 5)
    plt.figure()
    for value in sorted(data_x["my_cluster"].unique()):
        mask = data_x["my_cluster"] == value
        time_cell, survival_prob_cell, conf_int = kaplan_meier_estimator(
            data_y["osi"][mask], data_y["os"][mask], conf_type="log-log"
        )
        plt.step(time_cell, survival_prob_cell, where="post", label=f"{value} (n = {mask.sum()})")

    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.legend(loc="best")

    structured_array = np.rec.fromarrays(patient_groups[["osi", "os"]].values.T,
                                         dtype=[("status", bool), ("time", int)])
    _, pval, _, _ = compare_survival(structured_array, data_x["my_cluster"], return_stats=True)
    plt.title(f"{title} | p-value: {pval:.3f}")

    out_png = output_dir / "final_3_groups_survival.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")
    plt.show()
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fg = cfg.final_groups

    patient_info_new = pd.read_csv(output_dir / "tcga_new_cluster.csv",
                                   sep=",", header=0, index_col=0)

    # "best"  = patients on the ==2 side of EVERY listed consensus cluster
    best_sets = [set(patient_info_new[patient_info_new[f"new_cluster_{k}"] == 2].index)
                 for k in fg.best_clusters]
    best_group = set.intersection(*best_sets)
    # "worst" = patients on the ==2 side of ANY listed consensus cluster
    worst_sets = [set(patient_info_new[patient_info_new[f"new_cluster_{k}"] == 2].index)
                  for k in fg.worst_clusters]
    worst_group = set.union(*worst_sets)
    rest_group = set(patient_info_new.index) - best_group - worst_group
    # (original run produced best=13, worst=22, rest=132; may shift after the L1/L2 fixes)
    print(f"best={len(best_group)}  worst={len(worst_group)}  rest={len(rest_group)}")

    final_def = patient_info_new[["os", "osi"]].copy()
    final_def["my_cluster"] = 0
    final_def.loc[list(best_group), "my_cluster"] = 2
    final_def.loc[list(worst_group), "my_cluster"] = 1

    if args.plots:
        plot_survival_curve("Final Consensus Cluster", final_def, output_dir)

    out_csv = output_dir / "TCGA_new_3_groups.csv"
    final_def.to_csv(out_csv, sep=",", header=True, index=True)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
