"""
AML dataset-specific preprocessing: convert per-patient .dem.txt / .anno.txt
pairs into per-patient .h5ad files suitable for the deconvolution pipeline.

Input layout (one pair per patient):
  <data_dir>/<patient>.dem.txt   -- genes x cells count matrix (tab-separated,
                                     first col = "Gene", remaining cols = cell barcodes)
  <data_dir>/<patient>.anno.txt  -- cells x metadata (tab-separated,
                                     first col = "Cell", must contain "CellType" column)

Output:
  <out_dir>/<patient>.h5ad  -- AnnData with
      .X             : raw counts (cells x genes, scipy CSR)
      .obs["CellType"]: fine cell-type label (from anno CellType column)
      .obs["library_id"]: patient/library identifier (stem of the filename)
      .var_names     : gene symbols

Usage:
  python aml_txt_to_h5ad.py --data-dir /path/to/AML_data --out-dir /path/to/output
"""

import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def convert_patient(dem_path: str, anno_path: str, out_path: str) -> None:
    patient_id = os.path.splitext(os.path.basename(dem_path))[0].replace(".dem", "")

    counts = pd.read_csv(dem_path, sep="\t", index_col=0)  # genes × cells
    anno = pd.read_csv(anno_path, sep="\t", index_col=0)   # cells × metadata

    # align cells present in both files
    shared_cells = counts.columns.intersection(anno.index)
    if len(shared_cells) == 0:
        raise ValueError(f"{patient_id}: no shared cell barcodes between dem and anno files")
    if len(shared_cells) < len(counts.columns):
        print(f"  {patient_id}: dropping {len(counts.columns) - len(shared_cells)} cells missing from anno")

    counts = counts[shared_cells]        # genes × cells (aligned)
    anno   = anno.loc[shared_cells]      # cells × metadata (aligned)

    X = sp.csr_matrix(counts.values.T.astype(np.float32))  # cells × genes

    obs = pd.DataFrame(
        {
            "CellType":   anno["CellType"].values,
            "library_id": patient_id,
        },
        index=shared_cells,
    )
    var = pd.DataFrame(index=counts.index)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    adata.write_h5ad(out_path)
    print(f"  {patient_id}: {adata.n_obs} cells × {adata.n_vars} genes → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AML .dem/.anno txt pairs to per-patient h5ad files")
    parser.add_argument("--data-dir", required=True, help="Directory containing *.dem.txt and *.anno.txt files")
    parser.add_argument("--out-dir",  required=True, help="Output directory for *.h5ad files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dem_files = sorted(glob.glob(os.path.join(args.data_dir, "*.dem.txt")))
    if not dem_files:
        raise FileNotFoundError(f"No *.dem.txt files found in {args.data_dir}")

    print(f"Found {len(dem_files)} patient(s) to convert.")
    for dem_path in dem_files:
        stem = os.path.basename(dem_path).replace("-D0.dem.txt", "")
        anno_path = os.path.join(args.data_dir, f"{stem}-D0.anno.txt")
        if not os.path.exists(anno_path):
            print(f"  WARNING: no anno file for {stem}, skipping")
            continue
        out_path = os.path.join(args.out_dir, f"{stem}.h5ad")
        convert_patient(dem_path, anno_path, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
