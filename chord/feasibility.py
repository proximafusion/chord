"""Constraint definitions and the chord regularizer.

Feasibility is a set of scalar inequalities on a design's simulated metrics, each
written as `h_k(metrics) <= 0`. The chord regularizer penalizes violations not at the
training designs, where reconstruction already keeps them satisfied, but at points
decoded from *between* them. That interior is what the ordinary (c)VAE objective leaves
unconstrained.

Metrics are supplied by a caller-provided function rather than a fixed surrogate, so the
same penalty serves a closed-form forward model on the constructed benchmarks and a
learned surrogate on the stellarator problem.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import torch

from . import models

MetricPredictor = Callable[[torch.Tensor], torch.Tensor]
"""Maps a batch of designs to a batch of metrics, in physical units."""


@dataclasses.dataclass(frozen=True)
class ConstraintSpec:
    """One inequality `sign * metrics[metric_name] + offset <= 0`."""

    metric_name: str
    sign: float
    offset: float


N_FIELD_PERIODS = 3
"""Field-period count of the design corpus, which every bound below assumes.

The rotational transform is reported per field period, so a bound on the raw edge value
has to be divided by this before it can be compared.
"""

IOTA_EDGE = "edge_rotational_transform_over_n_field_periods"
MIRROR_RATIO_EDGE = "edge_magnetic_mirror_ratio"
VACUUM_WELL = "vacuum_well"

LOG10_QI = "log10_qi"
"""Base-10 logarithm of constellaration's `qi`, which is how the paper reports it."""

EFFECTIVE_ASPECT_RATIO = "effective_aspect_ratio"
EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO = "effective_aspect_ratio_minus_aspect_ratio"
"""Axis non-planarity, derived from the equilibrium rather than reported by it.

Named here with the other metric names so modules that only need a name do not have to
import the artifact store to get one.
"""

BROAD_SCAN_LOOSE_CONSTRAINTS: tuple[ConstraintSpec, ...] = (
    ConstraintSpec(IOTA_EDGE, -1.0, 0.725 / N_FIELD_PERIODS),
    ConstraintSpec(IOTA_EDGE, 1.0, -0.775 / N_FIELD_PERIODS),
    ConstraintSpec(MIRROR_RATIO_EDGE, 1.0, -0.4),
    ConstraintSpec(VACUUM_WELL, -1.0, 0.0),
)
"""Feasibility of the design corpus the generative models are trained on.

Raw edge rotational transform in `[0.725, 0.775]`, mirror ratio at most `0.4`,
and a non-negative vacuum well. This is broader than the published benchmark's
feasibility, which additionally bounds quasi-isodynamic quality and flux
compression; see `chord.evaluate` for that.
"""


def constraints_with_margin(
    iota_margin: float = 0.0,
    mirror_margin: float = 0.0,
    vacuum_well_margin: float = 0.0,
) -> tuple[ConstraintSpec, ...]:
    """Tighten each bound inward by the given margin.

    A search driven by an imperfect metric predictor will sit exactly on a bound and
    then fail the real simulator. Shrinking the feasible set by a small margin absorbs
    that prediction error.
    """
    return (
        ConstraintSpec(IOTA_EDGE, -1.0, (0.725 + iota_margin) / N_FIELD_PERIODS),
        ConstraintSpec(IOTA_EDGE, 1.0, -(0.775 - iota_margin) / N_FIELD_PERIODS),
        ConstraintSpec(MIRROR_RATIO_EDGE, 1.0, -(0.4 - mirror_margin)),
        ConstraintSpec(VACUUM_WELL, -1.0, vacuum_well_margin),
    )


class ConstraintSet:
    """Constraints bound to a metric ordering, with column indices resolved once."""

    def __init__(
        self, specs: Sequence[ConstraintSpec], metric_names: Sequence[str]
    ) -> None:
        name_to_column = {name: i for i, name in enumerate(metric_names)}
        missing = sorted({s.metric_name for s in specs} - set(name_to_column))
        if missing:
            raise KeyError(
                f"constraints reference metrics absent from the predictor output: "
                f"{missing}; available metrics are {list(metric_names)}"
            )
        self.specs = tuple(specs)
        self._columns = [name_to_column[s.metric_name] for s in specs]
        self._signs = [s.sign for s in specs]
        self._offsets = [s.offset for s in specs]

    def __len__(self) -> int:
        return len(self.specs)

    def violations(self, metrics: torch.Tensor) -> torch.Tensor:
        """Signed slack `h_k` per constraint, positive where violated."""
        columns = metrics[:, self._columns]
        signs = metrics.new_tensor(self._signs)
        offsets = metrics.new_tensor(self._offsets)
        return signs * columns + offsets

    def is_feasible(self, metrics: torch.Tensor) -> torch.Tensor:
        """Boolean mask over the batch, true where every constraint holds."""
        return (self.violations(metrics) <= 0).all(dim=-1)


def chord_penalty(
    model: models.CVAE,
    x: torch.Tensor,
    y: torch.Tensor,
    predict_metrics: MetricPredictor,
    constraints: ConstraintSet,
    n_pairs: int = 16,
) -> torch.Tensor:
    """Mean constraint violation at points decoded from between encoded designs.

    Random pairs from the batch are encoded, their codes mixed at a random blend weight,
    and the decoded result scored against the constraints.

    Returns a scalar, zero when every sampled interior point is feasible.
    """
    batch_size = x.shape[0]
    if batch_size < 2 or n_pairs < 1:
        return x.new_zeros(())

    n_pairs = min(n_pairs, batch_size // 2)
    order = torch.randperm(batch_size, device=x.device)[: 2 * n_pairs]
    left, right = order[:n_pairs], order[n_pairs:]

    # The encoder is detached on purpose. This term should teach the decoder to
    # keep straight latent paths feasible, not reshape the encoded distribution
    # until the paths happen to land well.
    with torch.no_grad():
        z_left = model.encode(x[left], y[left])
        z_right = model.encode(x[right], y[right])

    alpha = torch.rand(n_pairs, 1, device=x.device)
    z = alpha * z_left + (1.0 - alpha) * z_right
    if y.shape[-1] > 0:
        y_mixed = alpha * y[left] + (1.0 - alpha) * y[right]
    else:
        y_mixed = y.new_zeros(n_pairs, 0)

    metrics = predict_metrics(model.decode(z, y_mixed))
    return torch.relu(constraints.violations(metrics)).mean()
