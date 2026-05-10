"""Evaluate a held-out-patients run on its test patients' pseudobulks.

Requires the run to have been trained with --split-mode held_out_patients
(test_libraries is read from <run_dir>/params.json).

Reports the four Pearson correlation analyses:
  (1) per-pseudobulk prop r — corr over n_classes CTs, one per pseudobulk
  (2) per-CT prop r — corr over N pseudobulks, one per CT
  (3) per-(pseudobulk, CT) GEP r — corr over n_DEG genes
  (4) per-(CT, gene) GEP r — corr over N pseudobulks; per-CT summary

Outputs land in <run_dir>/heldout_pseudobulk_eval/ckpt{N}/.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402
from lib.model import UNet  # noqa: E402

# import build_arrays_from_pseudobulk from 04_train_unet.py
TRAIN_MOD = SCRIPT_DIR / "04_train_unet.py"
spec = importlib.util.spec_from_file_location("train_unet_mod", TRAIN_MOD)
tmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmod)
build_arrays_from_pseudobulk = tmod.build_arrays_from_pseudobulk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cuda", type=int, default=None)
    p.add_argument("--source", choices=["valbulk", "trainbulk", "both"], default="valbulk")
    return p.parse_args()


def safe_pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def hist_panel(ax, vals, title, bins, xlim=(-1, 1)):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=bins, edgecolor="black")
    med = float(np.median(vals)) if vals.size else float("nan")
    mu = float(np.mean(vals)) if vals.size else float("nan")
    ax.axvline(med, color="red", linestyle="--", linewidth=1)
    ax.axvline(0, color="silver", linestyle=":", linewidth=1)
    ax.set_xlim(xlim)
    ax.set_title(f"{title}\n n={len(vals)}  median={med:.3f}  mean={mu:.3f}")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("count")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = load_celltype_mapping(cfg.paths.celltype_mapping_yaml)
    ctlist = mapping["ctlist"]
    n_classes = len(ctlist)

    ckpt = args.ckpt
    run_dir = ckpt.parent
    params = json.loads((run_dir / "params.json").read_text())
    if params.get("split_mode") != "held_out_patients":
        raise ValueError(f"This script expects split_mode=held_out_patients; got {params.get('split_mode')}")

    preprocess_mode = params["preprocess_mode"]
    test_libs: list[str] = params["test_libraries"]
    pb_dir = Path(params["pseudobulk_dir"])
    ep_num = ckpt.stem.replace("model_ep", "")
    out_dir = run_dir / "heldout_pseudobulk_eval" / f"ckpt{ep_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ckpt              : {ckpt}")
    print(f"preprocess_mode   : {preprocess_mode}")
    print(f"source            : {args.source}")
    print(f"test libraries    : {len(test_libs)}")
    print(f"out               : {out_dir}")

    suffixes = {"valbulk": ["_valbulk.h5ad"], "trainbulk": ["_trainbulk.h5ad"],
                "both": ["_trainbulk.h5ad", "_valbulk.h5ad"]}[args.source]
    paths: list[Path] = []
    for lib in sorted(test_libs):
        for suf in suffixes:
            p = pb_dir / f"{lib}{suf}"
            if p.exists():
                paths.append(p)
    print(f"pseudobulk files  : {len(paths)}")

    deg_genes = sorted([g.strip() for g in
                        (cfg.paths.gene_lists_dir / "all_DEG_lst.txt").read_text().splitlines() if g.strip()])
    n_in = len(deg_genes)

    print("\nLoading pseudobulks…")
    count, gt_prop, gt_gep = build_arrays_from_pseudobulk(
        paths, deg_genes, ctlist, preprocess_mode=preprocess_mode,
    )
    print(f"count {count.shape}  prop {gt_prop.shape}  gep {gt_gep.shape}")

    sample_lib_tag, sample_split_tag = [], []
    import anndata as ad
    for p in paths:
        stem = p.stem
        lib_id, bulk_type = stem.rsplit("_", 1)
        n = ad.read_h5ad(p, backed="r").n_obs
        sample_lib_tag.extend([lib_id] * n)
        sample_split_tag.extend([bulk_type] * n)
    sample_lib_tag = np.array(sample_lib_tag)
    sample_split_tag = np.array(sample_split_tag)

    cuda = args.cuda if args.cuda is not None else cfg.training.cuda
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = UNet(n_in_gene=n_in, n_out_gene=n_in, n_classes=n_classes).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        prop_pred_lst, gep_pred_lst = [], []
        for i in range(0, count.shape[0], 128):
            x = torch.from_numpy(count[i:i+128]).to(device).float()
            p, g = model(x)
            prop_pred_lst.append(p.cpu().numpy())
            gep_pred_lst.append(g.cpu().numpy())
        prop_pred = np.concatenate(prop_pred_lst, axis=0)
        gep_pred = np.concatenate(gep_pred_lst, axis=0)
    print(f"pred prop {prop_pred.shape}  gep {gep_pred.shape}")

    # Pearson analyses
    print("\nComputing Pearson correlations…")
    n_pb = prop_pred.shape[0]

    pps = pd.DataFrame({
        "library_id": sample_lib_tag,
        "bulk_type":  sample_split_tag,
        "row_idx":    np.arange(n_pb),
        "pearson_r_prop": [safe_pearson(gt_prop[i], prop_pred[i]) for i in range(n_pb)],
    })
    pps.to_csv(out_dir / "prop_pearson_per_sample.csv", index=False)

    ppc = pd.DataFrame({
        "celltype": ctlist,
        "pearson_r_prop": [safe_pearson(gt_prop[:, ci], prop_pred[:, ci]) for ci in range(n_classes)],
    })
    ppc.to_csv(out_dir / "prop_pearson_per_celltype.csv", index=False)
    print(ppc.round(3).to_string(index=False))

    gep_corr_sample_ct = np.full((n_pb, n_classes), np.nan, dtype=np.float64)
    for i in range(n_pb):
        for ci in range(n_classes):
            gep_corr_sample_ct[i, ci] = safe_pearson(gt_gep[i, ci, :], gep_pred[i, ci, :])
    gsc_df = pd.DataFrame(gep_corr_sample_ct, columns=ctlist)
    gsc_df.insert(0, "library_id", sample_lib_tag)
    gsc_df.insert(1, "bulk_type", sample_split_tag)
    gsc_df.to_csv(out_dir / "gep_pearson_per_sample_celltype.csv", index=False)

    n_genes = gt_gep.shape[2]
    gep_corr_gene_ct = np.full((n_genes, n_classes), np.nan, dtype=np.float64)
    for ci in range(n_classes):
        a = gt_gep[:, ci, :]; b = gep_pred[:, ci, :]
        a_c = a - a.mean(axis=0, keepdims=True); b_c = b - b.mean(axis=0, keepdims=True)
        a_s = a_c.std(axis=0, ddof=0); b_s = b_c.std(axis=0, ddof=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            num = (a_c * b_c).mean(axis=0)
            r = np.where((a_s * b_s) > 0, num / (a_s * b_s), np.nan)
        gep_corr_gene_ct[:, ci] = r
    pd.DataFrame(gep_corr_gene_ct, index=deg_genes, columns=ctlist).rename_axis("gene_id").to_csv(
        out_dir / "gep_pearson_per_gene_celltype.csv"
    )
    summary = pd.DataFrame({
        "celltype": ctlist,
        "n_genes_with_corr": np.isfinite(gep_corr_gene_ct).sum(axis=0),
        "mean_r":        np.nanmean(gep_corr_gene_ct, axis=0),
        "median_r":      np.nanmedian(gep_corr_gene_ct, axis=0),
        "frac_r_gt_0.5": np.nanmean(gep_corr_gene_ct > 0.5, axis=0),
    })
    summary.to_csv(out_dir / "gep_pearson_per_celltype_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # histograms
    print("\nPlotting histograms…")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5)); plt.subplots_adjust(wspace=0.25)
    hist_panel(axs[0], ppc["pearson_r_prop"].values,
               f"prop — per-CT (across {n_pb} pseudobulks)", bins=np.linspace(-1, 1, 22))
    hist_panel(axs[1], pps["pearson_r_prop"].values,
               f"prop — per-pseudobulk (across {n_classes} CTs)", bins=np.linspace(-1, 1, 22))
    fig.savefig(out_dir / "hist_prop_pearson.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, n_classes, figsize=(4*n_classes, 4.5)); plt.subplots_adjust(wspace=0.3)
    fig.suptitle(f"ctGEP per-gene Pearson r (across {n_pb} pseudobulks)")
    for ci, ct in enumerate(ctlist):
        hist_panel(axs[ci], gep_corr_gene_ct[:, ci], ct, bins=40)
    fig.savefig(out_dir / "hist_GEP_per_gene_perCT.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, n_classes, figsize=(4*n_classes, 4.5)); plt.subplots_adjust(wspace=0.3)
    fig.suptitle(f"ctGEP per-pseudobulk Pearson r (across {n_in} genes)")
    for ci, ct in enumerate(ctlist):
        hist_panel(axs[ci], gep_corr_sample_ct[:, ci], ct, bins=np.linspace(-1, 1, 22))
    fig.savefig(out_dir / "hist_GEP_per_sample_perCT.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
