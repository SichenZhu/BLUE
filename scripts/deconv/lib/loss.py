"""Masked MSE losses for the dual-head UNet.

- prop_loss: plain MSE between predicted and true proportions.
- ctGEP_loss: per-cell-type MSE, masked to zero for samples where the cell
  type has zero ground-truth proportion (mimics original implementation).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def masked_ctgep_loss(
    ctgep_pred: torch.Tensor,   # (B, n_classes, n_out_gene)
    ctgep_true: torch.Tensor,   # (B, n_classes, n_out_gene)
    gt_prop: torch.Tensor,      # (B, n_classes)
    criterion: nn.Module,
) -> torch.Tensor:
    """Sum of per-cell-type MSE, zero-masked where gt_prop[:, i] == 0."""
    n_classes = gt_prop.shape[1]
    mask = (gt_prop != 0).to(ctgep_pred.dtype)  # (B, n_classes)
    loss = ctgep_pred.new_zeros(())
    for i in range(n_classes):
        m = mask[:, i, None]
        loss = loss + criterion(ctgep_pred[:, i, :] * m, ctgep_true[:, i, :])
    return loss
