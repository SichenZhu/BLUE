"""Predict cell-type proportions and ctGEP for the bulk dataset (config-driven).

Preprocessing mirrors training (preprocess_mode lives in the run's params.json):
the raw bulk is aligned to all_DEG_lst.txt and run through libsize-norm + 1e6
+ log2(+1) under the same convention used during training.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import anndata as ad
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.config import load_config, load_celltype_mapping  # noqa: E402
from lib.model import UNet  # noqa: E402
from lib.preprocess import preprocess_counts, align_to_gene_list  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Path to model_epN.pth. Defaults to last-saved ckpt of "
                        "the most recent run under paths.ckpt_root.")
    p.add_argument("--cuda", type=int, default=None)
    return p.parse_args()


def find_best_ckpt(root: Path) -> Path:
    runs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs under {root}")
    latest = runs[-1]
    candidates = sorted(
        latest.glob("model_ep*.pth"),
        key=lambda p: int(p.stem.replace("model_ep", "")),
    )
    if not candidates:
        raise FileNotFoundError(f"No model_ep*.pth under {latest}")
    return candidates[-1]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    mapping = load_celltype_mapping(cfg.paths.celltype_mapping_yaml)
    ctlist = mapping["ctlist"]
    n_classes = len(ctlist)

    ckpt = args.ckpt or find_best_ckpt(cfg.paths.ckpt_root)
    ckpt_id = ckpt.stem.replace("model_ep", "")
    run_dir = ckpt.parent

    # Read training params for the run to recover preprocess_mode + gene lists.
    params = json.loads((run_dir / "params.json").read_text())
    preprocess_mode = params["preprocess_mode"]
    # New keys (input/output gene lists), with fallback to legacy single-list key.
    input_gene_path  = Path(params.get("input_gene_list",  params.get("deg_path")))
    output_gene_path = Path(params.get("output_gene_list", params.get("deg_path")))

    out_dir = cfg.paths.predictions_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint     : {ckpt}")
    print(f"preprocess_mode: {preprocess_mode}")
    print(f"Output dir     : {out_dir}")

    input_gene_lst  = sorted({g.strip() for g in input_gene_path.read_text().splitlines() if g.strip()})
    output_gene_lst = sorted({g.strip() for g in output_gene_path.read_text().splitlines() if g.strip()})
    n_in  = len(input_gene_lst)
    n_out = len(output_gene_lst)
    if input_gene_path == output_gene_path:
        print(f"Gene list: {n_in} genes (input = output)")
    else:
        print(f"Input gene list:  {n_in} genes ({input_gene_path})")
        print(f"Output gene list: {n_out} genes ({output_gene_path})")

    print("\nLoading raw bulk…")
    df = pd.read_csv(cfg.paths.bulk_tsv, sep="\t", index_col="gene_id")
    sample_ids = df.columns.tolist()
    mat = df.T.values.astype(np.float64)
    src_genes = df.index.astype(str).tolist()

    if preprocess_mode == "deg_first":
        aligned = align_to_gene_list(mat, src_genes, input_gene_lst, fill_value=0.0)
        processed = preprocess_counts(aligned, up_scaling_coeff=1e6)
    elif preprocess_mode == "all_first":
        proc_full = preprocess_counts(mat, up_scaling_coeff=1e6)
        processed = align_to_gene_list(proc_full, src_genes, input_gene_lst, fill_value=0.0)
    else:
        raise ValueError(f"unknown preprocess_mode {preprocess_mode}")
    print(f"  bulk: {processed.shape}  dtype={processed.dtype}")

    cuda = args.cuda if args.cuda is not None else cfg.predict.cuda
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = UNet(n_in_gene=n_in, n_out_gene=n_out, n_classes=n_classes).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(processed).to(device).float()
        prop, ctgep = model(x)
        prop = prop.cpu().numpy()
        ctgep = ctgep.cpu().numpy()
    print(f"prop: {prop.shape}  ctgep: {ctgep.shape}")

    prop_df = pd.DataFrame(prop, index=sample_ids, columns=ctlist)
    prop_df.index.name = "library_id"
    if cfg.paths.bulk_metadata_tsv is not None and cfg.paths.bulk_metadata_tsv.exists():
        meta = pd.read_csv(cfg.paths.bulk_metadata_tsv, sep="\t")
        if "library_id" in meta.columns:
            meta = meta.drop_duplicates("library_id").set_index("library_id")
            prop_df = prop_df.join(meta, how="left")
    prop_csv = out_dir / f"predicted_proportions_ep{ckpt_id}.csv"
    prop_df.to_csv(prop_csv)
    print(f"wrote {prop_csv}")

    adata = ad.AnnData(X=np.zeros((len(sample_ids), n_out), dtype=np.float32))
    adata.obs_names = sample_ids
    adata.var_names = output_gene_lst
    adata.obs["ckpt"] = ckpt.name
    for ci, ct in enumerate(ctlist):
        adata.layers[ct] = ctgep[:, ci, :].astype(np.float32)
    adata.obsm["predicted_prop"] = prop.astype(np.float32)
    gep_h5 = out_dir / f"predicted_ctGEP_ep{ckpt_id}.h5ad"
    adata.write_h5ad(gep_h5)
    print(f"wrote {gep_h5}")

    print("\n--- predicted proportions head ---")
    print(prop_df[ctlist].head().round(3).to_string())
    print("\n--- predicted proportions mean across samples ---")
    print(prop_df[ctlist].mean().round(3).to_string())


if __name__ == "__main__":
    main()
