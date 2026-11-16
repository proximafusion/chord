"""Training loop shared by all four variants.

A variant is a pair of switches rather than a separate code path. Conditioning
is decided by the model's `y_dim`; the chord regularizer is active whenever
`chord_lambda` is positive and a metric predictor is supplied. Training a plain
VAE and training CHORD therefore differ only in configuration.

Both the KL weight and the chord weight are annealed. The KL term is annealed
because a conditional decoder otherwise ignores its conditioning early in
training, and the chord term is delayed because penalizing paths through a
latent that has not yet organised itself only adds noise.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Literal

import torch
from torch import nn
from torch.utils import data as torch_data

from . import feasibility as feasibility_mod
from . import models

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Optimisation, annealing and regularizer settings."""

    n_epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cosine_decay: bool = True

    kl_weight: float = 1.0
    free_bits: float = 0.0
    kl_anneal: Literal["none", "linear", "cyclical"] = "cyclical"
    kl_anneal_epochs: int = 50
    kl_n_cycles: int = 4
    kl_cycle_ratio: float = 0.5

    chord_lambda: float = 0.0
    chord_warmup_start_epoch: int = 0
    chord_warmup_epochs: int = 1
    chord_n_pairs: int = 16


def elbo(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction and KL terms, returned alongside their weighted sum.

    `free_bits` floors the KL of each latent dimension, which stops dimensions
    collapsing to the prior early in training.
    """
    reconstruction = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.mean()
    return reconstruction + kl_weight * kl, reconstruction, kl


def kl_weight_at(epoch: int, config: TrainingConfig) -> float:
    """KL weight for `epoch` under the configured annealing schedule."""
    if config.kl_anneal == "none" or config.kl_anneal_epochs <= 0:
        return config.kl_weight
    if config.kl_anneal == "linear":
        return config.kl_weight * min(1.0, epoch / config.kl_anneal_epochs)
    cycle_length = max(1.0, config.n_epochs / max(1, config.kl_n_cycles))
    position = (epoch % cycle_length) / cycle_length
    if position < config.kl_cycle_ratio:
        return config.kl_weight * (position / config.kl_cycle_ratio)
    return config.kl_weight


def chord_lambda_at(epoch: int, config: TrainingConfig) -> float:
    """Chord weight for `epoch`: zero, then a linear ramp, then constant."""
    if config.chord_lambda <= 0.0:
        return 0.0
    start = config.chord_warmup_start_epoch
    duration = max(1, config.chord_warmup_epochs)
    if epoch < start:
        return 0.0
    if epoch >= start + duration:
        return config.chord_lambda
    return config.chord_lambda * (epoch - start) / duration


def train(
    model: models.CVAE,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    config: TrainingConfig,
    predict_metrics: feasibility_mod.MetricPredictor | None = None,
    constraints: feasibility_mod.ConstraintSet | None = None,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """Train `model` in place and return one metrics record per epoch.

    Supplying `predict_metrics` and `constraints` enables the chord regularizer;
    without them, or with `chord_lambda` at zero, this is ordinary (c)VAE
    training.
    """
    wants_chord = config.chord_lambda > 0.0
    if wants_chord and (predict_metrics is None or constraints is None):
        raise ValueError(
            "chord_lambda is positive but no metric predictor or constraint set "
            "was supplied, so the regularizer cannot be evaluated"
        )

    loader = torch_data.DataLoader(
        torch_data.TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.n_epochs):
        kl_weight = kl_weight_at(epoch, config)
        chord_lambda = chord_lambda_at(epoch, config)
        _set_learning_rate(optimizer, epoch, config)

        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "chord": 0.0}
        n_batches = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            x_recon, mu, logvar = model(x_batch, y_batch)
            loss, reconstruction, kl = elbo(
                x_batch, x_recon, mu, logvar, kl_weight, config.free_bits
            )

            chord = x_batch.new_zeros(())
            if chord_lambda > 0.0:
                assert predict_metrics is not None and constraints is not None
                chord = feasibility_mod.chord_penalty(
                    model,
                    x_batch,
                    y_batch,
                    predict_metrics,
                    constraints,
                    config.chord_n_pairs,
                )
                loss = loss + chord_lambda * chord

            loss.backward()
            optimizer.step()

            # Detached before reading: these are only for the record, and
            # converting a tensor that still carries a gradient to a scalar
            # warns on every batch.
            totals["loss"] += float(loss.detach())
            totals["reconstruction"] += float(reconstruction.detach())
            totals["kl"] += float(kl.detach())
            totals["chord"] += float(chord.detach())
            n_batches += 1

        record = {k: v / max(1, n_batches) for k, v in totals.items()}
        record["epoch"] = float(epoch)
        record["kl_weight"] = kl_weight
        record["chord_lambda"] = chord_lambda
        if x_val is not None and y_val is not None:
            record["val_reconstruction"] = evaluate(model, x_val, y_val)
        history.append(record)

    return history


@torch.no_grad()
def evaluate(model: models.CVAE, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean reconstruction error, with the latent taken at its mean."""
    model.eval()
    x_recon = model.decode(model.encode(x, y), y)
    return float(nn.functional.mse_loss(x_recon, x, reduction="mean"))


def _set_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, config: TrainingConfig
) -> None:
    if not config.cosine_decay or config.n_epochs <= 1:
        return
    progress = epoch / (config.n_epochs - 1)
    lr = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
