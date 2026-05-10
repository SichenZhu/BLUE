"""Minimal config loader for the deconvolution pipeline.

All scripts under scripts/deconv/ load `config/pipeline_config.yaml` (or a path
passed via --config) and use the resulting dataclass-like dot-access object.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline_config.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _resolve_paths(d: dict) -> dict:
    """Resolve relative paths in d['paths'] to absolute Paths under REPO_ROOT.

    Keys ending in `_glob` are treated as glob patterns (strings), not paths.
    """
    out = dict(d)
    paths = dict(out.get("paths", {}))
    for k, v in paths.items():
        if v is None or k.endswith("_glob"):
            continue
        p = Path(v)
        if not p.is_absolute():
            p = REPO_ROOT / p
        paths[k] = p
    out["paths"] = paths
    return out


def load_config(path: Path | str | None = None) -> SimpleNamespace:
    """Load the YAML config and return a dot-accessible namespace.

    >>> cfg = load_config()
    >>> cfg.paths.combined_sc_h5ad
    PosixPath('.../outputs/combined_labeled.h5ad')
    >>> cfg.training.coeff_prop
    100
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    raw = _resolve_paths(raw)
    return _to_namespace(raw)


def load_celltype_mapping(path: Path | str) -> dict[str, list[str]]:
    """Read fine→coarse mapping (coarse: list-of-fine) and return it inverted
    as fine→coarse for fast lookup at cell-labeling time.

    Returns {coarse_to_fine: dict, fine_to_coarse: dict, ctlist: list}.
    """
    p = Path(path) if not Path(path).is_absolute() else Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as f:
        coarse_to_fine: dict[str, list[str]] = yaml.safe_load(f)
    fine_to_coarse: dict[str, str] = {}
    for coarse, fines in coarse_to_fine.items():
        for fine in fines:
            if fine in fine_to_coarse:
                raise ValueError(f"fine label '{fine}' assigned to multiple coarse labels")
            fine_to_coarse[fine] = coarse
    return {
        "coarse_to_fine": coarse_to_fine,
        "fine_to_coarse": fine_to_coarse,
        "ctlist": list(coarse_to_fine.keys()),
    }
