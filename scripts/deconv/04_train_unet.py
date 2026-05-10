"""Train the dual-head UNet on pseudobulks (config-driven).

Config keys this script reads (CLI flags override):
  paths.gene_lists_dir, paths.pseudobulk_dir, paths.ckpt_root,
  paths.celltype_mapping_yaml,
  training.{batch_size, epochs, lr, weight_decay, coeff_ct_gep, coeff_prop,
            sched_step, sched_gamma, use_amp, seed, cuda,
            preprocess_mode, split_mode, split_seed}.

Loss = coeff_ct_gep · masked-MSE(ctGEP) + coeff_prop · MSE(prop).
"""
from datetime import datetime
from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import anndata as ad

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402
from lib.model import UNet  # noqa: E402
from lib.loss import masked_ctgep_loss  # noqa: E402
from lib.preprocess import preprocess_counts  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--preprocess-mode", choices=["deg_first", "all_first"], default=None)
    p.add_argument("--coeff-prop", type=float, default=None)
    p.add_argument("--cuda", type=int, default=None)
    p.add_argument("--run-stamp", type=str, default=None,
                   help="Override default YYYYMMDD_HHMMSS run stamp.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-mode", choices=["pseudo", "held_out_patients"], default=None)
    p.add_argument("--split-seed", type=int, default=None)
    return p.parse_args()


class BulkDataset(data.Dataset):
    def __init__(self, count_data: np.ndarray, gt_prop: np.ndarray, ct_gep: np.ndarray):
        assert count_data.shape[0] == gt_prop.shape[0] == ct_gep.shape[0]
        self.count_data = count_data
        self.gt_prop = gt_prop
        self.ct_gep = ct_gep

    def __len__(self):
        return self.count_data.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.count_data[idx]),
            torch.from_numpy(self.gt_prop[idx]),
            torch.from_numpy(self.ct_gep[idx]),
        )


def _subset_and_preprocess(
    X,
    var_names: list[str],
    gene_lst: list[str],
    preprocess_mode: str,
) -> np.ndarray:
    """Subset a counts matrix to ``gene_lst`` and apply libsize+log preprocessing.

    preprocess_mode="deg_first": subset first, then libsize+log (libsize over the subset).
    preprocess_mode="all_first": libsize+log on full transcriptome first, then subset.
    """
    idx_map = {g: i for i, g in enumerate(var_names)}
    cols = np.array([idx_map[g] for g in gene_lst if g in idx_map])
    if preprocess_mode == "deg_first":
        x = np.asarray(X, dtype=np.float64)[:, cols]
        return preprocess_counts(x, up_scaling_coeff=1e6).astype(np.float32)
    else:  # all_first
        x_full = np.asarray(X, dtype=np.float64)
        x_full_proc = preprocess_counts(x_full, up_scaling_coeff=1e6)
        return x_full_proc[:, cols].astype(np.float32)


def build_arrays_from_pseudobulk(
    pseudobulk_paths: list[Path],
    input_gene_lst: list[str],
    output_gene_lst: list[str],
    ctlist: list[str],
    preprocess_mode: str = "deg_first",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pseudobulks and return (count_data, gt_prop, ct_gep), each float32.

    ``count_data`` is aligned to ``input_gene_lst`` (model input);
    ``ct_gep`` is aligned to ``output_gene_lst`` (model output target).
    Pass the same list for both to recover the original same-list behavior.
    """
    assert preprocess_mode in ("deg_first", "all_first")
    count_list, prop_list, gep_list = [], [], []

    for p in pseudobulk_paths:
        a = ad.read_h5ad(p)
        var_names = a.var_names.tolist()

        x_proc = _subset_and_preprocess(a.X, var_names, input_gene_lst, preprocess_mode)

        n_samples = a.n_obs
        gep = np.zeros((n_samples, len(ctlist), len(output_gene_lst)), dtype=np.float32)
        for ci, ct in enumerate(ctlist):
            if ct in a.layers:
                gep[:, ci, :] = _subset_and_preprocess(
                    a.layers[ct], var_names, output_gene_lst, preprocess_mode,
                )

        prop_src = a.uns["cellType_prop"]
        prop = prop_src[ctlist].values.astype(np.float32)
        prop = prop / prop.sum(axis=1, keepdims=True).clip(min=1e-12)

        count_list.append(x_proc)
        gep_list.append(gep)
        prop_list.append(prop)
        print(f"  {p.name}: {prop.shape[0]} samples")

    return (
        np.concatenate(count_list, axis=0),
        np.concatenate(prop_list, axis=0),
        np.concatenate(gep_list, axis=0),
    )


def evaluate(model, loader, coeff_ct_gep, coeff_prop, device):
    model.eval()
    crit = nn.MSELoss()
    tot_loss = 0.0
    tot_prop_loss = 0.0
    tot_gep_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for bulk, gtProp, ctGEP in loader:
            bulk = bulk.to(device, non_blocking=True).float()
            gtProp = gtProp.to(device, non_blocking=True).float()
            ctGEP = ctGEP.to(device, non_blocking=True).float()
            prop_pred, gep_pred = model(bulk)
            gep_loss = masked_ctgep_loss(gep_pred, ctGEP, gtProp, crit)
            prop_loss = crit(prop_pred, gtProp)
            tot_prop_loss += prop_loss.item()
            tot_gep_loss += gep_loss.item()
            tot_loss += (coeff_ct_gep * gep_loss + coeff_prop * prop_loss).item()
            n_batches += 1
    model.train()
    return tot_prop_loss / n_batches, tot_gep_loss / n_batches, tot_loss / n_batches


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = load_celltype_mapping(cfg.paths.celltype_mapping_yaml)
    ctlist = mapping["ctlist"]
    n_classes = len(ctlist)

    # Effective hyperparameters: CLI > config.
    t = cfg.training
    epochs        = args.epochs        if args.epochs        is not None else t.epochs
    seed          = args.seed          if args.seed          is not None else t.seed
    cuda          = args.cuda          if args.cuda          is not None else t.cuda
    coeff_prop    = args.coeff_prop    if args.coeff_prop    is not None else t.coeff_prop
    preprocess_mode = args.preprocess_mode if args.preprocess_mode is not None else t.preprocess_mode
    split_mode    = args.split_mode    if args.split_mode    is not None else t.split_mode
    split_seed    = args.split_seed    if args.split_seed    is not None else t.split_seed
    coeff_ct_gep  = t.coeff_ct_gep
    batch_size    = t.batch_size
    lr            = float(t.lr)
    weight_decay  = t.weight_decay
    sched_step    = t.sched_step
    sched_gamma   = t.sched_gamma
    use_amp       = t.use_amp

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    run_stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(cfg.paths.ckpt_root) / run_stamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    pb_dir = cfg.paths.pseudobulk_dir
    if split_mode == "pseudo":
        train_paths = [pb_dir / "all_trainbulk.h5ad"] + sorted(pb_dir.glob("*_trainbulk.h5ad"))
        train_paths = [p for p in train_paths if p.name != "all_trainbulk.h5ad" or p.exists()]
        # de-dup any overlap from glob
        seen = set(); train_paths = [p for p in train_paths if not (p in seen or seen.add(p))]
        val_paths = [pb_dir / "all_valbulk.h5ad"]
        train_libs: list[str] = []
        test_libs: list[str] = []
    else:  # held_out_patients
        all_train_h5 = sorted(p for p in pb_dir.glob("*_trainbulk.h5ad") if p.stem != "all_trainbulk")
        libs = sorted({p.stem.replace("_trainbulk", "") for p in all_train_h5})
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(libs))
        n_train = len(libs) // 2
        train_libs = sorted([libs[i] for i in order[:n_train]])
        test_libs  = sorted([libs[i] for i in order[n_train:]])
        train_paths, val_paths = [], []
        for lib in train_libs:
            for s in ("_trainbulk.h5ad", "_valbulk.h5ad"):
                p = pb_dir / f"{lib}{s}"
                if p.exists():
                    train_paths.append(p)
        for lib in test_libs:
            for s in ("_trainbulk.h5ad", "_valbulk.h5ad"):
                p = pb_dir / f"{lib}{s}"
                if p.exists():
                    val_paths.append(p)
        print(f"\nHeld-out-patients split (seed {split_seed}):")
        print(f"  train libraries ({len(train_libs)}): {train_libs}")
        print(f"  test  libraries ({len(test_libs)}): {test_libs}")

    # Resolve input/output gene lists (config override → default DEG list).
    default_gene_path = cfg.paths.gene_lists_dir / "all_DEG_lst.txt"
    input_gene_path  = getattr(cfg.paths, "input_gene_list",  None) or default_gene_path
    output_gene_path = getattr(cfg.paths, "output_gene_list", None) or input_gene_path

    input_gene_lst  = sorted({g.strip() for g in Path(input_gene_path ).read_text().splitlines() if g.strip()})
    output_gene_lst = sorted({g.strip() for g in Path(output_gene_path).read_text().splitlines() if g.strip()})
    n_in_gene  = len(input_gene_lst)
    n_out_gene = len(output_gene_lst)
    if input_gene_path == output_gene_path:
        print(f"\nGene list: {n_in_gene} genes (input = output)")
    else:
        print(f"\nInput gene list:  {n_in_gene} genes ({input_gene_path})")
        print(f"Output gene list: {n_out_gene} genes ({output_gene_path})")

    params = {
        "run_stamp": run_stamp,
        "preprocess_mode": preprocess_mode,
        "coeff_prop": coeff_prop,
        "coeff_ct_gep": coeff_ct_gep,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "sched_step": sched_step,
        "sched_gamma": sched_gamma,
        "use_amp": use_amp,
        "seed": seed,
        "cuda": cuda,
        "n_classes": n_classes,
        "ctlist": ctlist,
        # New keys (preferred by step 05 / 06):
        "input_gene_list":  str(input_gene_path),
        "output_gene_list": str(output_gene_path),
        "n_in_gene":  n_in_gene,
        "n_out_gene": n_out_gene,
        # Legacy key, kept for backward compatibility with old runs:
        "deg_path": str(input_gene_path),
        "pseudobulk_dir": str(pb_dir),
        "split_mode": split_mode,
        "split_seed": split_seed,
        "train_libraries": train_libs,
        "test_libraries": test_libs,
    }
    with (checkpoint_dir / "params.json").open("w") as f:
        json.dump(params, f, indent=2)
    print(f"Params written to {checkpoint_dir / 'params.json'}")
    print(json.dumps(params, indent=2))

    print(f"Training pseudobulk files: {len(train_paths)}")
    print(f"Validation pseudobulk files: {len(val_paths)}")

    print("\n=== Loading training data ===")
    train_count, train_prop, train_gep = build_arrays_from_pseudobulk(
        train_paths, input_gene_lst, output_gene_lst, ctlist, preprocess_mode=preprocess_mode,
    )
    print(f"train count {train_count.shape}  prop {train_prop.shape}  gep {train_gep.shape}")

    print("\n=== Loading validation data ===")
    val_count, val_prop, val_gep = build_arrays_from_pseudobulk(
        val_paths, input_gene_lst, output_gene_lst, ctlist, preprocess_mode=preprocess_mode,
    )
    print(f"val count {val_count.shape}  prop {val_prop.shape}  gep {val_gep.shape}")

    train_ds = BulkDataset(train_count, train_prop, train_gep)
    val_ds = BulkDataset(val_count, val_prop, val_gep)
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\nDevice: {device}")
    model = UNet(n_in_gene=n_in_gene, n_out_gene=n_out_gene, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.MSELoss()

    losses = {"train": [], "val_prop": [], "val_gep": [], "val_total": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        model.train()
        epoch_loss = 0.0
        for bulk, gtProp, ctGEP in tqdm(train_loader, total=len(train_loader)):
            bulk = bulk.to(device, non_blocking=True).float()
            gtProp = gtProp.to(device, non_blocking=True).float()
            ctGEP = ctGEP.to(device, non_blocking=True).float()

            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=use_amp):
                prop_pred, gep_pred = model(bulk)
                gep_loss = masked_ctgep_loss(gep_pred, ctGEP, gtProp, criterion)
                prop_loss = criterion(prop_pred, gtProp)
                loss = coeff_ct_gep * gep_loss + coeff_prop * prop_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        val_prop_l, val_gep_l, val_total = evaluate(model, val_loader, coeff_ct_gep, coeff_prop, device)
        losses["train"].append(epoch_loss)
        losses["val_prop"].append(val_prop_l)
        losses["val_gep"].append(val_gep_l)
        losses["val_total"].append(val_total)
        print(f"train {epoch_loss:.4f} | val_total {val_total:.4f} | val_prop {val_prop_l:.6f} | val_gep {val_gep_l:.4f}")
        scheduler.step()

        if val_total < best_val:
            best_val = val_total
            path = checkpoint_dir / f"model_ep{epoch}.pth"
            torch.save(model.state_dict(), path)
            print(f"  saved {path}")

    pd.DataFrame(losses).to_csv(checkpoint_dir / "loss_history.csv", index_label="epoch")
    print(f"\nLoss history -> {checkpoint_dir / 'loss_history.csv'}")


if __name__ == "__main__":
    main()
