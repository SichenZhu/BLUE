# BLUE

[Deconvolving cell-type-specific gene expression profiles from bulk RNA-seq samples](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014101)


## TO DO LIST
- [x] ~~Release code for applying BLUE to AML~~
- [x] ~~Provide instructions to applying BLUE to any customized dataset~~
- [ ] Downstream analysis for AML after deconvolution (Expected to be completed in the week of May 11, 2026)


## Environment setup

The project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment. Install uv once (per machine), then sync the project to create a virtual environment from `uv.lock`:

```bash
# 1. Install uv (skip if you already have it).
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the repo root, create the .venv and install pinned dependencies.
uv sync
```

`uv sync` installs every package recorded in `uv.lock`, including `torch` built against **CUDA 12.8** ([`pyproject.toml`](./pyproject.toml) pins the `https://download.pytorch.org/whl/cu128` wheel index). Your NVIDIA driver must be ≥ 12.8 — check with `nvidia-smi`. If you need a different CUDA wheel, edit the `tool.uv.index` URL in [`pyproject.toml`](./pyproject.toml) (e.g., `cu124`) and re-run `uv sync`.

Run any pipeline or dataset-specific script through uv (no need to activate the venv manually):

```bash
uv run python scripts/deconv/00_combine_sc_h5ads.py --config config/pipeline_config.yaml
```

**Note**: If uv doesn't work for you, simply create a conda environment with all dependencies in [`pyproject.toml`](./pyproject.toml) installed. 


## To use BLUE on a new project, you'd:
For more details, please refer to [`CustomDS_README.md`](./CustomDS_README.md) and [`AML_README.md`](./AML_README.md)

TL;DR ver:
  1. Place per-patient sc h5ads under `paths.sc_h5ad_dir`,
  2. Drop the bulk TSV at `paths.bulk_tsv` (genes × samples, first column named `gene_id`),
  3. Edit `config/celltype_mapping.yaml` to match your fine labels,
  4. Adjust `paths.*`, `sc_schema.*`, `training.cuda` in `config/pipeline_config.yaml`,
  5. Run scripts `00` → `06` from `scripts/deconv/` in order.
