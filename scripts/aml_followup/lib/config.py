"""Config loader for the AML downstream-analysis pipeline (scripts/aml_followup/).

Loads config/followup_config_aml.yaml (or a path passed via --config) and returns
a dot-accessible namespace. Entries under `paths` are resolved to absolute Paths
(relative values are anchored at the repo root). Mirrors scripts/deconv/lib/config.py.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "followup_config_aml.yaml"


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _resolve_paths(d: dict) -> dict:
    """Resolve every entry under d['paths'] to an absolute Path (relative values
    are anchored at REPO_ROOT)."""
    out = dict(d)
    paths = dict(out.get("paths", {}))
    for k, v in paths.items():
        if v is None:
            continue
        p = Path(v)
        paths[k] = p if p.is_absolute() else REPO_ROOT / p
    out["paths"] = paths
    return out


def load_config(path: Path | str | None = None) -> SimpleNamespace:
    """Load the YAML config and return a dot-accessible namespace."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    return _to_namespace(_resolve_paths(raw))
