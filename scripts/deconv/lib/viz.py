"""Prediction-vs-ground-truth scatter helpers (ported from ori_deconv_algo.py)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def l1_error(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def ccc_score(y_pred, y_true) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    rho = np.corrcoef(y_pred, y_true)[0, 1]
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    num = 2 * rho * y_true.std() * y_pred.std()
    den = var_t + var_p + (mean_t - mean_p) ** 2
    return float(num / den) if den != 0 else float("nan")


def plot_prop_pred_vs_gt(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    celltype_lst: list[str],
    out_path: str,
):
    """One panel per cell type: gt proportion vs predicted proportion."""
    num = len(celltype_lst)
    fig, axs = plt.subplots(nrows=1, ncols=num, figsize=(4 * num, 4))
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.supxlabel("gt")
    fig.supylabel("pred")

    for i, ct in enumerate(celltype_lst):
        x = gt_df.loc[:, ct].to_numpy()
        y = pred_df.loc[:, ct].to_numpy()
        l1 = l1_error(y, x)
        ccc = ccc_score(y, x)
        axs[i].scatter(x, y, s=6, alpha=0.5)
        axs[i].set_title(f"{ct}\n l1 {l1:.4f} | ccc {ccc:.3f}")
        axs[i].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
                    color="silver", linestyle="--")
        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_ylim(-0.05, 1.05)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ctgep_pred_vs_gt_avg(
    gt_gep: np.ndarray,       # (N, n_ct, n_gene)
    pred_gep: np.ndarray,     # (N, n_ct, n_gene)
    celltype_lst: list[str],
    out_path: str,
):
    """Per cell type: mean expression across samples, gt vs pred (KDE-colored)."""
    num = len(celltype_lst)
    fig, axs = plt.subplots(nrows=1, ncols=num, figsize=(4 * num, 4))
    axs = np.atleast_1d(axs)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.supxlabel("gt")
    fig.supylabel("pred")

    for i, ct in enumerate(celltype_lst):
        x = np.mean(gt_gep[:, i, :], axis=0)
        y = np.mean(pred_gep[:, i, :], axis=0)
        x_max = float(np.max(x)) if x.size else 0.0
        y_max = float(np.max(y)) if y.size else 0.0
        xy_max = max(x_max, y_max)

        if x.any():
            try:
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)
                axs[i].scatter(x, y, c=z, s=6)
            except Exception:
                axs[i].scatter(x, y, s=6, alpha=0.5)
            l1 = l1_error(y, x)
            ccc = ccc_score(y, x)
            axs[i].set_title(f"{ct}\n l1 {l1:.4f} | ccc {ccc:.3f}")
            axs[i].set_xlim(-0.1, x_max + 0.1)
            axs[i].set_ylim(-0.1, y_max + 0.1)
        else:
            axs[i].set_title(f"{ct}\n (missing in scref)")

        axs[i].plot(np.linspace(0, xy_max, 100), np.linspace(0, xy_max, 100),
                    color="silver", linestyle="--")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
