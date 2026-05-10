# Custom Dataset Deconvolution Pipeline

A step-by-step guide for running the deconvolution pipeline on your own dataset. The pipeline is config-driven and dataset-agnostic; you only need to (a) prepare two input files, (b) edit two YAML configs, and (c) run six numbered scripts in order.

For a worked example, see `AML_README.md` (TCGA-LAML deconvolution against an scRNA reference).


## Input contract

Before running the pipeline, prepare the following files:

1. **Per-patient single-cell h5ads** — one `*.h5ad` file per patient/library, all in the same directory.
   - `.X` must hold **raw counts** (integer-valued). If your raw counts live in a layer instead, point `sc_schema.raw_counts_layer` at it.
   - `obs` must contain a cell-type label column (any name) and a library/patient identifier column (any name) — name them in the config under `sc_schema.fine_celltype_col` / `sc_schema.library_id_col`.
   - `var_names` must be **gene IDs that match your bulk TSV** — use either gene symbols on both sides or Ensembl IDs on both sides.

2. **Bulk RNA-seq TSV** — tab-separated, **genes × samples**, first column named `gene_id`. Values are non-negative numerics (both counts or RPKM/TPM should work — the pipeline applies its own libsize+log2 normalization downstream).

3. **(Optional) bulk metadata TSV** — joined into the prediction CSV by `library_id`. Leave the config path null if you don't have one.

### Brief gotchas

- **Raw counts only.** Step 02 builds pseudobulks by summing raw counts per cell type — pre-normalized matrices in `.X` will silently produce wrong training data.
- **Gene-ID consistency.** A small `sc ∩ bulk` overlap means a weak deconvolution signal; the pipeline's bulk-prep helper warns when overlap drops below thousands of genes (see `aml_tcga_bulk_to_tsv.py` for an example with a `--min-common-genes` check).
- **Cell-type names must round-trip exactly.** Every fine label appearing in your sc h5ads must be listed under exactly one coarse key in `celltype_mapping.yaml`; cells with unmapped labels are dropped silently with a warning.

## Setup: config files

Copy the two example configs and edit them for your project:

```bash
cp config/pipeline_config.yaml    config/pipeline_config_<your_project>.yaml
cp config/celltype_mapping.yaml   config/celltype_mapping_<your_project>.yaml
```

In `pipeline_config_<your_project>.yaml`, edit:

- `paths.sc_h5ad_dir`, `paths.bulk_tsv` — your input files (absolute or repo-relative).
- `paths.combined_sc_h5ad`, `paths.gene_lists_dir`, `paths.pseudobulk_dir`, `paths.ckpt_root`, `paths.predictions_dir` — output locations.
- `paths.celltype_mapping_yaml` — point at the mapping file you copied above.
- `sc_schema.fine_celltype_col`, `sc_schema.library_id_col`, `sc_schema.raw_counts_layer` — match your h5ads' obs/layer names.
- `training.cuda` — GPU id.

In `celltype_mapping_<your_project>.yaml`, define the **coarse → fine mapping** that the model will deconvolve. Keys are the cell-type classes you want; each value is a list of fine labels (as they appear in `obs[fine_celltype_col]`) that collapse into that key. Cells whose fine label is not listed under any key are dropped. 

We recommend to use an appropriate number of cell types to deconvolve based on the scale of annotated cells in training single cell references (i.e. If the total number of cells are small in your training sc references, then pick a small number of cell types to deconvolve.)

```yaml
HSC-like:     [HSC-like]
Prog-like:    [Prog-like]
T_lineage:    [T_naive, T_effector, T_memory]   # multiple fine labels collapse
```

All steps below take `--config config/pipeline_config_<your_project>.yaml`.

## Deconvolution pipeline

### Step 00 — combine per-patient sc h5ads into one labeled reference

```bash
uv run python scripts/deconv/00_combine_sc_h5ads.py --config config/pipeline_config_<your_project>.yaml
```

Reads each per-patient h5ad, applies the fine→coarse cell-type mapping (writing the result to `obs["cell.type.for.deconv"]`), and concatenates them with an inner gene join.

_Output_: `paths.combined_sc_h5ad` (one combined AnnData; cells without a mapped label are dropped).

### Step 01 — build common-gene and DEG lists

```bash
uv run python scripts/deconv/01_build_gene_lists.py --config config/pipeline_config_<your_project>.yaml
```

Computes `sc ∩ bulk` common genes, then runs per-patient one-vs-rest Wilcoxon DEG selection (FDR ≤ `degs.fdr_rate`, top `degs.num_deg_per_ct` per cell type, deduplicated across cell types).

_Outputs_ (in `paths.gene_lists_dir`):
- `all_common_gene_lst.txt` — sc ∩ bulk gene overlap.
- `all_DEG_lst.txt` — DEG union, intersected with common genes.

### Step 02 — simulate pseudobulks

```bash
uv run python scripts/deconv/02_simulate_pseudobulk.py --config config/pipeline_config_<your_project>.yaml
```

Generates Dirichlet-mixed pseudobulks (uniform + dominant α=50/15/5 variants, ~`pseudobulk.n_cells` cells per sample) per patient and pooled across all patients. The alpha parameter in Dirichlet distribution could be adjusted given the number of cell types to deconvolve. 

_Outputs_ (in `paths.pseudobulk_dir`):
- per-patient `<lib>_trainbulk.h5ad`, `<lib>_valbulk.h5ad`
- pooled `all_trainbulk.h5ad`, `all_valbulk.h5ad`

### Step 03 — preprocess real bulk

```bash
uv run python scripts/deconv/03_preprocess_bulk.py --config config/pipeline_config_<your_project>.yaml
```

Aligns the bulk TSV to the common-gene list and applies libsize-normalization → ×1e6 → log2(+1).

_Output_: `<gene_lists_dir>/bulk_preprocessed.h5ad` (samples × common genes).

### Step 04 — train UNet

```bash
uv run python scripts/deconv/04_train_unet.py --config config/pipeline_config_<your_project>.yaml
```

Trains the dual-head UNet on pseudobulks (loss = `coeff_ct_gep × masked-MSE(ctGEP) + coeff_prop × MSE(prop)`). Each run is timestamped under `paths.ckpt_root/<YYYYMMDD_HHMMSS>/`. Can adjust the coefficients before two branchs for a more balanced training. 

_Outputs_:
- `params.json` — full hyperparameter snapshot (consumed by steps 05/06).
- `model_epN.pth` — checkpoint saved each epoch where validation loss improved.
- `loss_history.csv` — per-epoch train/val losses.

### Step 05 — predict on real bulk

```bash
uv run python scripts/deconv/05_predict_bulk.py --config config/pipeline_config_<your_project>.yaml
```

Auto-finds the latest run dir under `paths.ckpt_root` and the highest-epoch `model_ep*.pth`. Override with `--ckpt /path/to/model_epN.pth` for a specific checkpoint.

_Outputs_ (in `paths.predictions_dir/<YYYYMMDD_HHMMSS>/`):
- `predicted_proportions_epN.csv` — samples × cell-type fractions (sums to 1 per sample).
- `predicted_ctGEP_epN.h5ad` — predicted per-cell-type gene expression (one layer per cell type).

### Step 06 — visualize predictions on training/validation pseudobulks

```bash
uv run python scripts/deconv/06_visualize_predictions.py --config config/pipeline_config_<your_project>.yaml
```

Sanity-check on synthetic ground-truth pseudobulks. Reads `ctlist`/`pseudobulk_dir` from the run's `params.json`.

_Outputs_ (in `<run_dir>/viz/`):
- `train_prop.png`, `train_GEP.png` — predictions vs. ground truth on a training subsample.
- `val_prop.png`, `val_GEP.png` — same for the validation pseudobulks.
- `prop_metrics.csv` — per-cell-type L1 error and CCC scores.

## Advanced

### Custom input / output gene lists

By default, both the model input (bulk features) and output (per-cell-type GEP) use `<gene_lists_dir>/all_DEG_lst.txt`. You can decouple them by setting two optional fields in your config:

```yaml
paths:
  ...
  input_gene_list:  /path/to/input_genes.txt    # null → default DEG list
  output_gene_list: /path/to/output_genes.txt   # null → reuse input list
```

In our own experiments, we also tried to use a **wider** input list (e.g., the full `all_common_gene_lst.txt`) so the encoder sees more bulk features, and a **narrower** output list (e.g., the DEGs) to focus the per-cell-type GEP head on informative genes (and it works). Both files must be one-gene-per-line and contain only gene IDs already present in the pseudobulks (subset of `all_common_gene_lst.txt`).

Both paths are recorded in `params.json`, so steps 05 and 06 automatically pick them up — no need to repeat the override on the prediction or visualization commands.

### Held-out-patient evaluation

To benchmark generalization, set `training.split_mode: held_out_patients` in the config (with an optional `training.split_seed`). Step 04 will then partition libraries 50/50 by patient ID — the held-out half is used as the validation set instead of `all_valbulk.h5ad`. After training, run:

```bash
uv run python scripts/deconv/06_eval_held_out_pseudobulks.py --config config/pipeline_config_<your_project>.yaml --ckpt <path_to_model_epN.pth>
```

for per-(sample, cell-type, gene) Pearson correlation analyses on the held-out patients.
