"""Walk the latent chord between stellarator designs from different basins.

This is the paper's headline experiment. Two feasible designs are encoded, the straight
line between their latent codes is decoded step by step, and every decoded design is
checked. A latent whose interior is feasible keeps the whole path feasible; one that is
not fails somewhere in the middle.

The conditioning is the other half of it, and the comparison here is between two ways of
choosing it. Blending the endpoints' metric vectors linearly is the obvious thing and it
asks the decoder for metric values no latent code near the chord attains, so the decoder
drifts off the data manifold. Letting the conditioning settle to a fixed point of the
decoder asks instead for something the decoder can actually produce.

Designs are scored twice. The surrogate scores every step, which is what makes a sweep
of this size affordable at all, and then VMEC is run on a subsample, because a surrogate
agreeing with itself proves nothing.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ... import corpus as corpus_mod
from ... import evaluate as evaluate_mod
from ... import feasibility as feasibility_mod
from ... import inference as inference_mod
from ... import surrogate as surrogate_mod
from . import training

_LOG = logging.getLogger(__name__)

INTERIOR = (0.3, 0.7)
"""Range of the path treated as the interior.

The endpoints are corpus designs and feasible by construction, so including them
flatters every method equally and hides the difference the experiment is about.
"""

LINEAR = "linear"
SELF_CONSISTENT = "self_consistent"
CONDITIONING = (LINEAR, SELF_CONSISTENT)


@dataclasses.dataclass(frozen=True)
class Sweep:
    """Decoded paths between endpoint pairs, with the surrogate's verdict on each."""

    alphas: NDArray
    boundaries: NDArray
    """Raw Fourier coefficients, shaped `(pairs, steps, coefficients)`."""

    metrics: NDArray
    feasible: NDArray
    conditioning: str

    def __len__(self) -> int:
        return len(self.feasible)

    def interior_mask(self) -> NDArray:
        """Include interval endpoints despite one-ULP grid-rounding error."""
        return (self.alphas >= np.nextafter(INTERIOR[0], -np.inf)) & (
            self.alphas <= np.nextafter(INTERIOR[1], np.inf)
        )

    def interior(self) -> float:
        """Fraction of interior steps the surrogate calls feasible."""
        return self._mean(self.feasible[:, self.interior_mask()])

    def full_path(self) -> float:
        return self._mean(self.feasible)

    def fully_feasible(self) -> float:
        """Fraction of pairs for which all evaluated steps are feasible."""
        return self._mean(np.asarray(self.feasible.all(axis=-1)))

    def _mean(self, values: NDArray) -> float:
        return float(values.mean()) if len(self) else float("nan")


@dataclasses.dataclass(frozen=True)
class Validation:
    """What the simulator said about a subsample of a sweep."""

    alphas: NDArray
    converged: NDArray
    feasible: NDArray
    benchmark_feasible: NDArray
    """Feasibility under the published benchmark, which is stricter than the corpus."""

    def interior(self) -> float:
        inside = (self.alphas >= np.nextafter(INTERIOR[0], -np.inf)) & (
            self.alphas <= np.nextafter(INTERIOR[1], np.inf)
        )
        values = self.feasible[:, inside]
        return float(values.mean()) if values.size else float("nan")

    def convergence_rate(self) -> float:
        return float(self.converged.mean()) if self.converged.size else float("nan")


def cross_basin_pairs(
    corpus: corpus_mod.Corpus,
    n_pairs: int = 30,
    split: str = "test",
    seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Pair held-out designs from different basins.

    Endpoints from the same basin are often joined by a path that never leaves it, so a
    model can score well on them without having learned anything about the geometry.
    """
    rows = np.flatnonzero(corpus.split == split)
    basins = np.unique(corpus.basin[rows])
    if len(basins) < 2:
        raise ValueError(
            f"the {split!r} split holds only basin {basins.tolist()}, so there is "
            f"nothing to interpolate across"
        )
    first = rows[corpus.basin[rows] == basins[0]]
    second = rows[corpus.basin[rows] == basins[1]]

    rng = np.random.default_rng(seed)
    available = min(n_pairs, len(first), len(second))
    return (
        rng.choice(first, available, replace=False),
        rng.choice(second, available, replace=False),
    )


def sweep(
    trained: training.Trained,
    predictor: surrogate_mod.StandardizedPredictor,
    corpus: corpus_mod.Corpus,
    rows_a: NDArray,
    rows_b: NDArray,
    conditioning: str = SELF_CONSISTENT,
    n_alpha: int = 21,
    n_iterations: int = 30,
    constraints: feasibility_mod.ConstraintSet | None = None,
) -> Sweep:
    """Decode the chord between each pair and score every step with the surrogate."""
    if conditioning not in CONDITIONING:
        raise ValueError(
            f"conditioning must be one of {list(CONDITIONING)}, got {conditioning!r}"
        )
    constraints = constraints or feasibility_mod.ConstraintSet(
        feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS, predictor.metric_names
    )
    alphas = np.linspace(0.0, 1.0, n_alpha)
    n_pairs = len(rows_a)

    conditions = corpus.conditions(trained.condition_columns)
    z_a = trained.encode(corpus.boundaries[rows_a], conditions[rows_a])
    z_b = trained.encode(corpus.boundaries[rows_b], conditions[rows_b])
    y_a = trained.standardized_conditions(conditions[rows_a])
    y_b = trained.standardized_conditions(conditions[rows_b])

    weights = torch.as_tensor(alphas, dtype=torch.float32).reshape(-1, 1, 1)
    z = ((1.0 - weights) * z_a + weights * z_b).reshape(-1, z_a.shape[-1])
    y = _blend(y_a, y_b, weights, n_alpha, n_pairs)
    if conditioning == SELF_CONSISTENT:
        y = inference_mod.self_consistent_conditioning(
            trained.model,
            z,
            y,
            predictor,
            trained.scaler,
            trained.condition_columns,
            constraints=constraints,
            n_iterations=n_iterations,
        )

    # Scoring the decoder's own output rather than re-encoding the raw
    # coefficients avoids a round trip through float32 that would move the
    # designs slightly before the surrogate ever sees them.
    with torch.no_grad():
        standardized = trained.model.decode(z, y)
        metrics = predictor(standardized)
        feasible = constraints.is_feasible(metrics).numpy()
    boundaries = trained.standardization.decode(standardized.numpy())

    _LOG.info(
        "%s sweep: %d pairs x %d steps, surrogate calls %.0f%% feasible",
        conditioning,
        n_pairs,
        n_alpha,
        100 * feasible.mean(),
    )
    return Sweep(
        alphas=alphas,
        boundaries=_to_pairs(boundaries, n_alpha, n_pairs),
        metrics=_to_pairs(metrics.numpy(), n_alpha, n_pairs),
        feasible=_to_pairs(feasible, n_alpha, n_pairs),
        conditioning=conditioning,
    )


def validate(
    result: Sweep,
    stride: int = 4,
    max_pairs: int | None = 8,
    settings: Any | None = None,
    n_workers: int | None = None,
) -> Validation:
    """Run the simulator on a subsample of a sweep.

    A full sweep is thousands of designs and the simulator costs seconds each, so this
    takes every `stride`-th step of the first `max_pairs` pairs. The surrogate's verdict
    on the rest is only as good as the surrogate.
    """
    steps = np.arange(0, len(result.alphas), stride)
    pairs = len(result) if max_pairs is None else min(max_pairs, len(result))
    selected = result.boundaries[:pairs][:, steps]

    boundaries = [
        evaluate_mod.boundary_from_vector(design)
        for design in selected.reshape(-1, selected.shape[-1])
    ]
    _LOG.info("validating %d designs with the simulator", len(boundaries))
    evaluations = evaluate_mod.evaluate_batch(
        boundaries, settings=settings, n_workers=n_workers
    )

    constraints = feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS
    # Two specs bound the rotational transform from either side, so the metric
    # names repeat. Pulling the same metric out of the simulator twice happens
    # to give the right answer, but it is luck rather than intent.
    names = tuple(dict.fromkeys(spec.metric_name for spec in constraints))
    values, converged = evaluate_mod.metrics_to_tensor(evaluations, names)
    feasible = feasibility_mod.ConstraintSet(constraints, names).is_feasible(values)
    strict = np.array(
        [
            bool(e.converged and evaluate_mod.benchmark_feasible(e.metrics))
            for e in evaluations
        ]
    )
    shape = (pairs, len(steps))
    return Validation(
        alphas=result.alphas[steps],
        converged=converged.numpy().reshape(shape),
        feasible=(feasible.numpy() & converged.numpy()).reshape(shape),
        benchmark_feasible=strict.reshape(shape),
    )


def _blend(
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    weights: torch.Tensor,
    n_alpha: int,
    n_pairs: int,
) -> torch.Tensor:
    # An unconditional decoder takes a zero-width vector, which cannot be
    # reshaped from a flattened stack because it holds no elements to count.
    if y_a.shape[-1] == 0:
        return torch.zeros(n_alpha * n_pairs, 0)
    blended = inference_mod.linear_conditioning(
        y_a.expand(n_alpha, *y_a.shape), y_b.expand(n_alpha, *y_b.shape), weights
    )
    return blended.reshape(-1, y_a.shape[-1])


def _to_pairs(flat: NDArray, n_alpha: int, n_pairs: int) -> NDArray:
    """Flattened step-major results back to `(pairs, steps, ...)`."""
    trailing = flat.shape[1:] if flat.ndim > 1 else ()
    return flat.reshape(n_alpha, n_pairs, *trailing).swapaxes(0, 1)
