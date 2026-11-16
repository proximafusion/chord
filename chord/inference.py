"""Self-consistent conditioning, the inference-time half of the method.

A conditional decoder is told what metrics to produce. Nothing guarantees the
request is achievable: interpolating the conditioning vector between two designs
asks for metric values that no latent code near the chord actually attains, and
the decoder answers by drifting off the data manifold. Feasibility collapses
even when the latent interior is well shaped.

The fix is to stop imposing the conditioning and instead let it settle. Decode,
ask the surrogate what was actually produced, condition on that, and repeat. The
fixed point is a request the decoder can meet, because it is one the decoder
made.

Clamping keeps that fixed point inside the feasible set rather than on its
boundary, where a small surrogate error puts the design outside. The margin is
expressed as a tightened `ConstraintSet`, so the interior margin and the
constraint margin are one concept.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from . import feasibility as feasibility_mod
from . import models


class MetricPredictorWithNames(Protocol):
    """A metric predictor that also reports the order of its outputs."""

    metric_names: tuple[str, ...]

    def __call__(self, designs: torch.Tensor) -> torch.Tensor: ...


@dataclasses.dataclass(frozen=True)
class ConditionScaler:
    """Standardization of the conditioning vector.

    The decoder consumes standardized conditioning while constraints are stated in
    physical units, so the iteration crosses between the two every step.
    """

    mean: NDArray
    std: NDArray

    @classmethod
    def fit(cls, raw: NDArray) -> ConditionScaler:
        std = raw.std(axis=0)
        return cls(mean=raw.mean(axis=0), std=np.where(std > 1e-12, std, 1.0))

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self._as_tensor(self.mean, raw)) / self._as_tensor(self.std, raw)

    def decode(self, standardized: torch.Tensor) -> torch.Tensor:
        mean = self._as_tensor(self.mean, standardized)
        return standardized * self._as_tensor(self.std, standardized) + mean

    def _as_tensor(self, values: NDArray, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def bounds_from_constraints(
    constraints: feasibility_mod.ConstraintSet,
) -> dict[str, tuple[float | None, float | None]]:
    """Per-metric `(lower, upper)` interval implied by a constraint set.

    A constraint reads `sign * value + offset <= 0`, so a negative sign states a
    lower bound of `offset` and a positive sign an upper bound of `-offset`.
    """
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for spec in constraints.specs:
        lower, upper = bounds.get(spec.metric_name, (None, None))
        if spec.sign < 0:
            candidate = spec.offset
            lower = candidate if lower is None else max(lower, candidate)
        else:
            candidate = -spec.offset
            upper = candidate if upper is None else min(upper, candidate)
        bounds[spec.metric_name] = (lower, upper)
    return bounds


def self_consistent_conditioning(
    model: models.CVAE,
    z: torch.Tensor,
    y0: torch.Tensor,
    predict_metrics: MetricPredictorWithNames,
    scaler: ConditionScaler,
    condition_columns: Sequence[str],
    constraints: feasibility_mod.ConstraintSet | None = None,
    n_iterations: int = 30,
) -> torch.Tensor:
    """Iterate the conditioning to a fixed point of the decoder.

    `y0` and the return value are standardized, matching what the decoder
    consumes. Passing `constraints` clamps the constrained components to the
    interval those constraints imply, keeping the fixed point off the boundary.
    """
    missing = [c for c in condition_columns if c not in predict_metrics.metric_names]
    if missing:
        raise KeyError(
            f"conditioning columns {missing} are not predicted; the predictor "
            f"reports {list(predict_metrics.metric_names)}"
        )
    columns = [predict_metrics.metric_names.index(c) for c in condition_columns]
    clamps = _clamp_targets(constraints, condition_columns) if constraints else []

    y = y0
    with torch.no_grad():
        for _ in range(n_iterations):
            metrics = predict_metrics(model.decode(z, y))
            raw = metrics[:, columns].clone()
            for position, lower, upper in clamps:
                raw[:, position] = raw[:, position].clamp(min=lower, max=upper)
            y = scaler.encode(raw)
    return y


def linear_conditioning(
    y_a: torch.Tensor, y_b: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Blend two conditioning vectors, the naive baseline the method replaces."""
    return (1.0 - alpha) * y_a + alpha * y_b


def _clamp_targets(
    constraints: feasibility_mod.ConstraintSet, condition_columns: Sequence[str]
) -> list[tuple[int, float | None, float | None]]:
    bounds = bounds_from_constraints(constraints)
    return [
        (position, *bounds[name])
        for position, name in enumerate(condition_columns)
        if name in bounds
    ]
