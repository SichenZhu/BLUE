"""Scatter prediction vs simulated ground truth on training subsample and
all validation pseudobulks. Mirrors the pattern at ori_deconv_algo.py:920-931.

Auto-resolves the latest run under ``cfg.paths.ckpt_root`` (like
``05_predict_bulk.py``) and reads ``ctlist`` / ``deg_path`` / ``pseudobulk_dir``
from that run's ``params.json``, so it works for any dataset without code edits.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config  # noqa: E402
from lib.model import UNet  # noqa: E402
from lib.viz import plot_prop_pred_vs_gt, plot_ctgep_pred_vs_gt_avg  # noqa: E402

# Reuse data loader from the training script.
import importlib.util  # noqa: E402
TRAIN_MOD_PATH = SCRIPT_DIR / "04_train_unet.py"
spec = importlib.util.spec_from_file_location("train_unet_mod", TRAIN_MOD_PATH)
tmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmod)
build_arrays_from_pseudobulk = tmod.build_arrays_from_pseudobulk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Path to model_epN.pth. Defaults to last-saved ckpt of "
                        "the most recent run under paths.ckpt_root.")
    p.add_argument("--stride", type=int, default=1000,
                   help="Training subsampling stride (every N-th sample).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <run_dir>/viz/")
    p.add_argument("--cuda", type=int, default=None)
    return p.parse_args()


def find_best_ckpt(ckpt_root: Path) -> Path:
    runs = sorted([p for p in Path(ckpt_root).iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs under {ckpt_root}")
    latest = runs[-1]
    ckpts = sorted(
        latest.glob("model_ep*.pth"),
        key=lambda p: int(p.stem.replace("model_ep", "")),
    )
    if not ckpts:
        raise FileNotFoundError(f"No model_ep*.pth under {latest}")
    return ckpts[-1]


@torch.no_grad()
def batched_forward(model, x_np: np.ndarray, device, batch: int = 128):
    props, geps = [], []
    for i in range(0, x_np.shape[0], batch):
        x = torch.from_numpy(x_np[i : i + batch]).to(device).float()
        p, g = model(x)
        props.append(p.cpu().numpy())
        geps.append(g.cpu().numpy())
    return np.concatenate(props, axis=0), np.concatenate(geps, axis=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    ckpt = args.ckpt or find_best_ckpt(cfg.paths.ckpt_root)
    run_dir = ckpt.parent
    params = json.loads((run_dir / "params.json").read_text())

    ctlist = params["ctlist"]
    n_classes = len(ctlist)
    # New keys (input/output gene lists), with fallback to legacy single-list key.
    input_gene_path  = Path(params.get("input_gene_list",  params.get("deg_path")))
    output_gene_path = Path(params.get("output_gene_list", params.get("deg_path")))
    pb_dir = Path(params["pseudobulk_dir"])
    preprocess_mode = params.get("preprocess_mode", "deg_first")

    out_dir = args.out_dir or (run_dir / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_id = args.cuda if args.cuda is not None else getattr(cfg.predict, "cuda", 0)

    print(f"ckpt   : {ckpt}")
    print(f"out    : {out_dir}")
    print(f"stride : {args.stride} (training subsample)")
    print(f"ctlist : {ctlist}")

    input_gene_lst  = sorted({g.strip() for g in input_gene_path.read_text().splitlines() if g.strip()})
    output_gene_lst = sorted({g.strip() for g in output_gene_path.read_text().splitlines() if g.strip()})
    n_in  = len(input_gene_lst)
    n_out = len(output_gene_lst)
    if input_gene_path == output_gene_path:
        print(f"Gene list: {n_in} genes (input = output)")
    else:
        print(f"Input gene list:  {n_in} genes ({input_gene_path})")
        print(f"Output gene list: {n_out} genes ({output_gene_path})")

    train_paths = [pb_dir / "all_trainbulk.h5ad"] + sorted(
        p for p in pb_dir.glob("*_trainbulk.h5ad") if p.name != "all_trainbulk.h5ad"
    )
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in [pb_dir / "all_valbulk.h5ad"] if p.exists()]
    if not val_paths:
        val_paths = sorted(p for p in pb_dir.glob("*_valbulk.h5ad"))

    print("\n=== Load training pseudobulks ===")
    train_count, train_prop, train_gep = build_arrays_from_pseudobulk(
        train_paths, input_gene_lst, output_gene_lst, ctlist, preprocess_mode=preprocess_mode,
    )
    print(f"train: count {train_count.shape} prop {train_prop.shape} gep {train_gep.shape}")

    print("\n=== Load validation pseudobulks ===")
    val_count, val_prop, val_gep = build_arrays_from_pseudobulk(
        val_paths, input_gene_lst, output_gene_lst, ctlist, preprocess_mode=preprocess_mode,
    )
    print(f"val: count {val_count.shape} prop {val_prop.shape} gep {val_gep.shape}")

    # subsample training for visualization
    train_count_s = train_count[:: args.stride]
    train_prop_s = train_prop[:: args.stride]
    train_gep_s = train_gep[:: args.stride]
    print(f"\nTrain subsample for viz: {train_count_s.shape[0]} samples")

    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = UNet(n_in_gene=n_in, n_out_gene=n_out, n_classes=n_classes).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # --- train scatter ---
    print("\nPredicting on train subsample…")
    pred_prop_tr, pred_gep_tr = batched_forward(model, train_count_s, device)
    gt_prop_tr_df = pd.DataFrame(train_prop_s, columns=ctlist)
    pred_prop_tr_df = pd.DataFrame(pred_prop_tr, columns=ctlist)
    plot_prop_pred_vs_gt(
        gt_prop_tr_df, pred_prop_tr_df, ctlist,
        out_path=str(out_dir / "train_prop.png"),
    )
    plot_ctgep_pred_vs_gt_avg(
        train_gep_s, pred_gep_tr, ctlist,
        out_path=str(out_dir / "train_GEP.png"),
    )
    print("  wrote train_prop.png and train_GEP.png")

    # --- val scatter ---
    print("\nPredicting on all val samples…")
    pred_prop_v, pred_gep_v = batched_forward(model, val_count, device)
    gt_prop_v_df = pd.DataFrame(val_prop, columns=ctlist)
    pred_prop_v_df = pd.DataFrame(pred_prop_v, columns=ctlist)
    plot_prop_pred_vs_gt(
        gt_prop_v_df, pred_prop_v_df, ctlist,
        out_path=str(out_dir / "val_prop.png"),
    )
    plot_ctgep_pred_vs_gt_avg(
        val_gep, pred_gep_v, ctlist,
        out_path=str(out_dir / "val_GEP.png"),
    )
    print("  wrote val_prop.png and val_GEP.png")

    # --- per-cell-type metrics ---
    from lib.viz import l1_error, ccc_score

    def summary(gt, pred, label):
        return [
            {
                "split": label,
                "celltype": ct,
                "l1_prop": l1_error(pred[:, ci], gt[:, ci]),
                "ccc_prop": ccc_score(pred[:, ci], gt[:, ci]),
            }
            for ci, ct in enumerate(ctlist)
        ]

    rows = summary(train_prop_s, pred_prop_tr, "train_sub") + summary(val_prop, pred_prop_v, "val")
    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "prop_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nPer-cell-type prop metrics → {metrics_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
