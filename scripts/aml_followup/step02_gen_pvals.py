"""step02 -- per-definition log-rank survival p-values.

Runs after step01. For every binary patient-group definition from step01, fits
Kaplan-Meier survival on the two groups (`Surv(os, osi) ~ cluster`) and records
the log-rank test p-value; step03 reads the resulting
TCGA_patient_cluster_ct<ct>_pvals.txt files.

The p-value is the Mantel-Cox log-rank test (`scipy.stats.logrank`); it was
validated against the R `survminer::surv_pvalue(survfit(...))` result (~1e-14).

Input  (paths.output_dir): <n>/TCGA_patient_cluster_ct<n>.csv        (n = 1..6)
Output (paths.output_dir): <n>/TCGA_patient_cluster_ct<n>_pvals.txt

Usage:  uv run python scripts/aml_followup/step02_gen_pvals.py \
            --config config/followup_config_aml.yaml
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import CensoredData, logrank

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: config/followup_config_aml.yaml)")
    return p.parse_args()


def logrank_pvalue(time, event, cluster):
    """Two-group log-rank test p-value for one binary definition.

    time    : survival time per patient (the `os` column)
    event   : 1 = death observed, 0 = right-censored (the `osi` column)
    cluster : binary group label (0 / 1)

    Returns the two-sided log-rank p-value, or nan if either group is empty.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    cluster = np.asarray(cluster)
    labels = np.unique(cluster)
    if len(labels) != 2:
        return float("nan")
    # scipy CensoredData: `censored` is True where the value is right-censored,
    # i.e. the death was not observed (osi == 0).
    samples = [
        CensoredData.right_censored(time[cluster == g], event[cluster == g] == 0)
        for g in labels
    ]
    return float(logrank(*samples).pvalue)


def compute_pvals(csv_path):
    """Log-rank p-value for every definition column in a combined ct CSV.

    CSV layout: index, os, osi, <def_1>, <def_2>, ...  -- one 0/1 column per
    definition. Returns (column_names, pvalues) in column order, matching the
    `columns[2:]` that step02 consumes.
    """
    df = pd.read_csv(csv_path, index_col=0)
    time, event = df["os"].values, df["osi"].values
    def_cols = list(df.columns[2:])
    pvals = [logrank_pvalue(time, event, df[c].values) for c in def_cols]
    return def_cols, pvals


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.paths.output_dir
    for ct in range(1, 7):
        csv_path = output_dir / str(ct) / f"TCGA_patient_cluster_ct{ct}.csv"
        out_path = output_dir / str(ct) / f"TCGA_patient_cluster_ct{ct}_pvals.txt"
        def_cols, pvals = compute_pvals(csv_path)
        np.savetxt(out_path, np.asarray(pvals), fmt="%.15g")
        print(f"ct{ct}: {len(pvals)} definitions -> {out_path}")


if __name__ == "__main__":
    main()
