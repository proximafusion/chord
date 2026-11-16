"""Generate designs inside a gap, at aspect ratios no campaign ran.

Filling a gap means asking for designs at a particular aspect ratio, and the decoder is
conditioned on one. But self-consistent conditioning replaces every conditioning
component with what the decoder actually produced, which would let the aspect ratio
drift to wherever the model finds comfortable and defeat the request.

Pinning it uses the machinery already there. The iteration clamps constrained components
to the interval their constraints imply, so a narrow window around the target aspect
ratio holds it there while every other metric is free to settle. The request survives
the iteration without a second code path.

Targets are spread across the gap rather than piled on its midpoint, because a front is
filled by covering it and the hardest aspect ratios are not always the central ones.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch
from numpy.typing import NDArray

from ... import feasibility as feasibility_mod
from ... import inference as inference_mod
from ... import surrogate as surrogate_mod
from ..stellarator import training as training_mod
from . import gaps as gaps_mod
from . import selection

_LOG = logging.getLogger(__name__)

ASPECT_RATIO = selection.ASPECT_RATIO


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """How many designs to ask for, and where in the gap."""

    n_bins: int = 10
    per_bin: int = 30
    n_iterations: int = 30

    aspect_ratio_half_width: float = 0.02
    """How tightly the requested aspect ratio is held during the iteration.

    Narrow enough that a design lands where it was asked for, wide enough that the clamp
    does not fight the other metrics settling.
    """

    objective_low: float = -3.5
    objective_high: float = -1.5
    """Range the quasi-isodynamic target is drawn from, biasing toward good designs."""


@dataclasses.dataclass(frozen=True)
class Generated:
    """Designs produced for one gap, and the aspect ratio each was asked for."""

    boundaries: NDArray
    requested_aspect_ratio: NDArray
    metrics: NDArray
    feasible: NDArray
    metric_names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.boundaries)

    def achieved_aspect_ratio(self) -> NDArray:
        return self.metrics[:, self.metric_names.index(ASPECT_RATIO)]

    def describe(self) -> str:
        achieved = self.achieved_aspect_ratio()
        return (
            f"{len(self)} designs, {int(self.feasible.sum())} feasible by the "
            f"surrogate, aspect ratio {achieved.min():.2f}..{achieved.max():.2f}"
        )


def aspect_ratio_window(
    target: float, half_width: float
) -> tuple[feasibility_mod.ConstraintSpec, ...]:
    """Constraints pinning the aspect ratio to a narrow interval around `target`.

    Expressed as constraints rather than as a special case so the self-consistent
    iteration holds the request using the clamping it already does.
    """
    return (
        feasibility_mod.ConstraintSpec(ASPECT_RATIO, 1.0, -(target + half_width)),
        feasibility_mod.ConstraintSpec(ASPECT_RATIO, -1.0, target - half_width),
    )


def targets(gap: gaps_mod.Gap, n_bins: int) -> NDArray:
    """Aspect ratios to ask for, spread across the interior of the gap.

    The endpoints are anchors where campaigns already ran, so there is nothing to fill
    there and the bins sit strictly inside.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    edges = np.linspace(gap.low, gap.high, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def fill(
    trained: training_mod.Trained,
    predictor: surrogate_mod.StandardizedPredictor,
    gap: gaps_mod.Gap,
    config: GenerationConfig | None = None,
    seed: int = 0,
) -> Generated:
    """Generate designs across a gap, one batch per requested aspect ratio."""
    config = config or GenerationConfig()
    if ASPECT_RATIO not in trained.condition_columns:
        raise KeyError(
            f"the decoder is not conditioned on {ASPECT_RATIO!r}, so it cannot be "
            f"asked for one; it takes {list(trained.condition_columns)}"
        )
    generator = torch.Generator().manual_seed(seed)

    batches = [
        _fill_one(trained, predictor, float(target), config, generator)
        for target in targets(gap, config.n_bins)
    ]
    generated = Generated(
        boundaries=np.concatenate([b.boundaries for b in batches]),
        requested_aspect_ratio=np.concatenate(
            [b.requested_aspect_ratio for b in batches]
        ),
        metrics=np.concatenate([b.metrics for b in batches]),
        feasible=np.concatenate([b.feasible for b in batches]),
        metric_names=tuple(predictor.metric_names),
    )
    _LOG.info("%s: %s", gap.name, generated.describe())
    return generated


def _fill_one(
    trained: training_mod.Trained,
    predictor: surrogate_mod.StandardizedPredictor,
    target: float,
    config: GenerationConfig,
    generator: torch.Generator,
) -> Generated:
    constraints = feasibility_mod.ConstraintSet(
        (
            *feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS,
            *aspect_ratio_window(target, config.aspect_ratio_half_width),
        ),
        predictor.metric_names,
    )
    z = torch.randn(config.per_bin, trained.model.latent_dim, generator=generator)
    y = _request(trained, target, config, generator)
    y = inference_mod.self_consistent_conditioning(
        trained.model,
        z,
        y,
        predictor,
        trained.scaler,
        trained.condition_columns,
        constraints=constraints,
        n_iterations=config.n_iterations,
    )

    with torch.no_grad():
        standardized = trained.model.decode(z, y)
        metrics = predictor(standardized)
        feasible = constraints.is_feasible(metrics).numpy()
    return Generated(
        boundaries=trained.standardization.decode(standardized.numpy()),
        requested_aspect_ratio=np.full(config.per_bin, target),
        metrics=metrics.numpy(),
        feasible=feasible,
        metric_names=tuple(predictor.metric_names),
    )


def _request(
    trained: training_mod.Trained,
    target: float,
    config: GenerationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Standardized conditioning to start from.

    Every component starts at the corpus mean except the two being steered.
    """
    mean = torch.as_tensor(trained.scaler.mean, dtype=torch.float32)
    raw = mean.reshape(1, -1).repeat(config.per_bin, 1)
    columns = list(trained.condition_columns)
    raw[:, columns.index(ASPECT_RATIO)] = target

    if feasibility_mod.LOG10_QI in columns:
        spread = torch.rand(config.per_bin, generator=generator, dtype=raw.dtype)
        raw[:, columns.index(feasibility_mod.LOG10_QI)] = (
            config.objective_low
            + spread * (config.objective_high - config.objective_low)
        )
    return trained.scaler.encode(raw)
