"""Differentiable stand-in for the simulator.

The chord regularizer needs metrics for designs decoded during training, and
calling the simulator on each would be far too slow. The surrogate supplies
them cheaply and differentiably. It is frozen whenever it is used to shape a
decoder, so the generator cannot learn to exploit its errors.

Metrics are regressed in standardized units, since they span very different
scales. `StandardizedMetrics` adapts them back to physical units so they can be
compared against constraint bounds directly.

`fit` trains one. Front-filling needs this rather than the released checkpoint: each
gap gets its own surrogate, trained on that gap's two anchors alone, which is what makes
the separation between gaps structural instead of something a filter has to enforce.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils import data as torch_data

from . import features, models

_LOG = logging.getLogger(__name__)


class SurrogateModel(nn.Module):
    """Trunk and a metric-regression head."""

    def __init__(
        self,
        x_dim: int = 81,
        n_metrics: int = 18,
        hidden_dims: list[int] | None = None,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]
        self.x_dim = x_dim
        self.n_metrics = n_metrics
        head_dim = hidden_dims[-1]
        trunk_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims
        self.backbone = models.MLPBlock(
            x_dim, head_dim, trunk_dims, use_layer_norm, use_residual
        )
        self.metrics_head = nn.Linear(head_dim, n_metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized metrics, which `StandardizedMetrics` returns to physical
        units."""
        return self.metrics_head(self.backbone(x))


@runtime_checkable
class StandardizedPredictor(Protocol):
    """A metric predictor carrying the feature transform it was fitted with.

    Both `StandardizedMetrics` and the released-surrogate adapter satisfy this. Code
    that needs the transform as well as the predictions takes this rather than either
    concrete type, since the two are related by composition and not by inheritance.
    """

    metric_names: tuple[str, ...]

    @property
    def standardization(self) -> features.Standardization: ...

    def __call__(self, designs: torch.Tensor) -> torch.Tensor: ...


class StandardizedMetrics:
    """Presents a surrogate as a metric predictor in physical units.

    Instances satisfy `chord.feasibility.MetricPredictor`, so a surrogate can be
    handed directly to the chord regularizer or to any constraint evaluation.
    """

    def __init__(
        self,
        model: SurrogateModel,
        mean: torch.Tensor,
        std: torch.Tensor,
        metric_names: Sequence[str],
        standardization: features.Standardization | None = None,
    ) -> None:
        if mean.shape != std.shape:
            raise ValueError(
                f"normalization statistics disagree in shape: mean {tuple(mean.shape)} "
                f"against std {tuple(std.shape)}"
            )
        if len(metric_names) != mean.shape[-1]:
            raise ValueError(
                f"{len(metric_names)} metric names for {mean.shape[-1]} metrics"
            )
        self.model = model
        self.mean = mean
        self.std = std
        self.metric_names = tuple(metric_names)
        self._standardization = standardization

    @property
    def standardization(self) -> features.Standardization:
        """The feature transform these weights were fitted against.

        Carried with the model because the statistics are not interchangeable: a
        surrogate trained on one set of designs and fed features standardized
        against another returns confident nonsense, with nothing to flag it.
        """
        if self._standardization is None:
            raise AttributeError(
                "this surrogate was built without a feature standardization, so "
                "boundaries cannot be encoded for it; supply the one its training "
                "features were built with"
            )
        return self._standardization

    def predict_boundaries(self, raw: NDArray) -> torch.Tensor:
        """Metrics for raw Fourier coefficients, encoded the way training was."""
        encoded = self.standardization.encode(raw)
        return self(torch.as_tensor(encoded, dtype=self.mean.dtype))

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        return self.model(designs) * self.std + self.mean

    def freeze(self) -> StandardizedMetrics:
        """Put the surrogate in eval mode and stop gradients to its parameters."""
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self


def surrogate_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked squared error on standardized metrics.

    The mask drops entries the simulator never produced, so a missing target contributes
    nothing rather than pulling the prediction toward zero.
    """
    squared_error = (predicted - target) ** 2 * mask
    return squared_error.sum() / mask.sum().clamp_min(1.0)


@dataclasses.dataclass(frozen=True)
class SurrogateConfig:
    """Optimisation settings for fitting a surrogate."""

    n_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cosine_decay: bool = True
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    validation_fraction: float = 0.1


def fit(
    boundaries: NDArray,
    metrics: NDArray,
    metric_names: Sequence[str],
    config: SurrogateConfig | None = None,
    alpha: float = 1.0,
    seed: int = 0,
) -> tuple[StandardizedMetrics, list[dict[str, float]]]:
    """Fit a surrogate on raw boundaries and their simulated metrics.

    The feature transform is fitted here and carried on the result, so the returned
    predictor cannot be separated from the standardization its weights expect. That
    pairing is the one thing about a surrogate that is easy to get wrong and silent when
    it is.

    Rows whose metrics the simulator never produced are kept rather than dropped. They
    still describe a real boundary, and the loss masks the missing entries, so a design
    that failed on one metric still teaches the others.

    Returns the frozen predictor and one record per epoch.
    """
    config = config or SurrogateConfig()
    torch.manual_seed(seed)
    boundaries = np.asarray(boundaries, dtype=np.float64)
    metrics = np.asarray(metrics, dtype=np.float64)
    if len(boundaries) != len(metrics):
        raise ValueError(
            f"{len(boundaries)} boundaries against {len(metrics)} metric rows"
        )
    if metrics.shape[-1] != len(metric_names):
        raise ValueError(
            f"{metrics.shape[-1]} metric columns against {len(metric_names)} names"
        )

    train_rows, validation_rows = _split(
        len(boundaries), config.validation_fraction, seed
    )
    standardization = features.Standardization.fit(boundaries[train_rows], alpha=alpha)
    mean, std = _metric_statistics(metrics[train_rows])

    x = torch.as_tensor(standardization.encode(boundaries), dtype=torch.float32)
    observed = np.isfinite(metrics)
    targets = torch.as_tensor(
        np.where(observed, (metrics - mean) / std, 0.0), dtype=torch.float32
    )
    mask = torch.as_tensor(observed, dtype=torch.float32)
    _LOG.info(
        "fitting on %d rows (%d held out), %d of %d metric entries observed",
        len(train_rows),
        len(validation_rows),
        int(observed.sum()),
        observed.size,
    )

    model = SurrogateModel(
        x_dim=x.shape[-1],
        n_metrics=len(metric_names),
        hidden_dims=list(config.hidden_dims),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = torch_data.DataLoader(
        torch_data.TensorDataset(x[train_rows], targets[train_rows], mask[train_rows]),
        batch_size=config.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.n_epochs):
        _set_learning_rate(optimizer, epoch, config)
        model.train()
        total, batches = 0.0, 0
        for x_batch, target_batch, mask_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = surrogate_loss(model(x_batch), target_batch, mask_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        record = {"epoch": float(epoch), "train": total / max(1, batches)}
        if len(validation_rows):
            record["validation"] = _validation_loss(
                model,
                x[validation_rows],
                targets[validation_rows],
                mask[validation_rows],
            )
        history.append(record)

    predictor = StandardizedMetrics(
        model,
        torch.as_tensor(mean, dtype=torch.float32),
        torch.as_tensor(std, dtype=torch.float32),
        metric_names,
        standardization,
    ).freeze()
    return predictor, history


@torch.no_grad()
def _validation_loss(
    model: SurrogateModel,
    x: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    model.eval()
    return float(surrogate_loss(model(x), targets, mask))


def _metric_statistics(metrics: NDArray) -> tuple[NDArray, NDArray]:
    """Per-metric mean and spread, ignoring entries the simulator never produced."""
    mean = np.nanmean(metrics, axis=0)
    std = np.nanstd(metrics, axis=0)
    # A metric that never varies carries nothing, and dividing by its zero spread
    # would put infinities into the targets.
    return np.nan_to_num(mean), np.where(std > 1e-12, std, 1.0)


def _split(n: int, validation_fraction: float, seed: int) -> tuple[NDArray, NDArray]:
    order = np.random.default_rng(seed).permutation(n)
    n_validation = int(round(validation_fraction * n))
    return order[n_validation:], order[:n_validation]


def _set_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, config: SurrogateConfig
) -> None:
    if not config.cosine_decay or config.n_epochs <= 1:
        return
    progress = epoch / (config.n_epochs - 1)
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
