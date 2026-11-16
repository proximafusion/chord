"""What a constructed benchmark has to provide.

The two benchmarks differ in geometry and dimension but not in interface, so the
dataset generator, the training harness and the interpolation evaluation are written
once against this protocol rather than duplicated per problem.

Feasibility here is analytical rather than simulated, which is the point of these
benchmarks: every claim about a decoded design can be checked exactly and for free.
That makes them the place to establish that the regularizer does what it is supposed
to, before any of it depends on a surrogate being accurate.

A problem reaches the rest of the package through `StandardizedViolations`, which
presents its analytical constraints under the same interface a learned surrogate
satisfies. The training code therefore cannot tell the two apart, which is what makes
the toy results evidence about the method rather than about a separate implementation
of it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

from ... import feasibility as feasibility_mod
from ... import inference as inference_mod

Objective = Callable[[NDArray], float]
"""Scalar to minimize during dataset generation.

Lower is better.
"""


@runtime_checkable
class Problem(Protocol):
    """A constructed benchmark with analytically known feasibility."""

    name: ClassVar[str]

    @property
    def n_dim(self) -> int: ...

    @property
    def n_basins(self) -> int:
        """How many disconnected regions the feasible set has."""
        ...

    @property
    def constraint_names(self) -> tuple[str, ...]:
        """One name per constraint, in the order `violations` returns them."""
        ...

    def violations(self, x: NDArray) -> NDArray:
        """Signed slack per constraint, positive where violated."""
        ...

    def violations_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable mirror of `violations`, which the regularizer needs."""
        ...

    def conditions(self, x: NDArray) -> NDArray:
        """Metric vector a conditional decoder is conditioned on."""
        ...

    def basin(self, x: NDArray) -> NDArray:
        """Index of the disconnected region each design belongs to."""
        ...

    def seed(self, basin: int, rng: np.random.Generator) -> NDArray:
        """A feasible design drawn from one basin, to start a search from."""
        ...

    def sample_objective(self, basin: int, rng: np.random.Generator) -> Objective:
        """Draw the objective one generation run will minimize."""
        ...


def is_feasible(problem: Problem, x: NDArray) -> NDArray:
    """Mask over a batch, true where every constraint holds."""
    return np.asarray((problem.violations(x) <= 0).all(axis=-1))


def constraint_set(problem: Problem) -> feasibility_mod.ConstraintSet:
    """The problem's constraints, in the form the regularizer consumes.

    A constraint reads `sign * metric + offset <= 0`, and here the metric is the
    violation itself, so each one is the identity with no offset.
    """
    names = problem.constraint_names
    return feasibility_mod.ConstraintSet(
        [feasibility_mod.ConstraintSpec(name, 1.0, 0.0) for name in names], names
    )


class StandardizedViolations:
    """Presents analytical violations the way a learned surrogate is presented.

    Designs reach this already standardized, since that is what the decoder emits, so
    the transform has to be undone before the geometry means anything.
    """

    def __init__(self, problem: Problem, scaler: inference_mod.ConditionScaler) -> None:
        self.problem = problem
        self.scaler = scaler
        self.metric_names: tuple[str, ...] = problem.constraint_names

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        return self.problem.violations_torch(self.scaler.decode(designs))
