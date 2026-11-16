"""Inputs and outputs for inference from archived front-filling models."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch
    from chord import features, inference


class PreparedSplit(NamedTuple):
    """Saved model inputs, conditions and achieved aspect ratios for one split."""

    x: torch.Tensor
    y: torch.Tensor
    aspect_ratio: torch.Tensor


class PreparedDataset(NamedTuple):
    """Saved splits and their transforms; no statistics are refitted on loading."""

    splits: dict[str, PreparedSplit]
    standardization: features.Standardization
    scaler: inference.ConditionScaler
    condition_names: tuple[str, ...]
    template_boundary_json: str


class DecodedSweep(NamedTuple):
    """Raw coefficients and provenance of decoded anchor interpolants."""

    boundaries: torch.Tensor
    alpha: torch.Tensor
    low_train_index: torch.Tensor
    high_train_index: torch.Tensor
