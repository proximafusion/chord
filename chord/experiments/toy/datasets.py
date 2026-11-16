"""Generate a training corpus for a constructed benchmark.

Sampling the feasible set uniformly would be the easy thing to do and the wrong one.
Real design corpora come out of optimizer trajectories, so they are dense where the
search lingers and thin near the awkward corners, and a latent model trained on uniform
data would never meet the uneven density it has to cope with in practice.

So each basin is explored by a fan of short CMA-ES runs started from feasible seeds,
every point the optimizer proposes is pooled, and the pool is filtered to what is
actually feasible. The optimizer is steered by the problem's own objective plus a
penalty on constraint violation, which is the same augmented-Lagrangian shape the
stellarator campaigns use.
"""

from __future__ import annotations

import dataclasses
import logging

import cma
import numpy as np
from numpy.typing import NDArray

from . import problem as problem_mod

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """Size and shape of the search that produces the corpus."""

    n_runs_per_basin: int = 20
    initial_step: float = 0.15
    population: int = 8
    n_generations: int = 30

    penalty_weight: float = 50.0
    """Weight on constraint violation in the augmented objective.

    High enough that the search stays near the feasible set, low enough that it still
    probes the boundary, which is where the interesting training data is.
    """


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Feasible designs pooled from the search, with their metrics and basins."""

    designs: NDArray
    conditions: NDArray
    basin: NDArray

    def __len__(self) -> int:
        return len(self.designs)

    def counts(self) -> dict[int, int]:
        labels, counts = np.unique(self.basin, return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}


def generate(
    problem: problem_mod.Problem,
    config: GenerationConfig | None = None,
    seed: int = 0,
) -> Dataset:
    """Search every basin and return the feasible designs the search passed through."""
    config = config or GenerationConfig()
    rng = np.random.default_rng(seed)

    proposed: list[NDArray] = []
    for basin in range(problem.n_basins):
        for _ in range(config.n_runs_per_basin):
            proposed.extend(_explore(problem, basin, config, rng))
    if not proposed:
        raise RuntimeError(f"the search on {problem.name} proposed no designs at all")

    pool = np.concatenate(proposed, axis=0)
    feasible = pool[problem_mod.is_feasible(problem, pool)]
    if not len(feasible):
        raise RuntimeError(
            f"none of the {len(pool)} designs proposed on {problem.name} were "
            f"feasible; the configured geometry may be empty"
        )

    dataset = Dataset(
        designs=feasible.astype(np.float32),
        conditions=problem.conditions(feasible).astype(np.float32),
        basin=problem.basin(feasible),
    )
    _LOG.info(
        "%s: %d feasible of %d proposed, per basin %s",
        problem.name,
        len(dataset),
        len(pool),
        dataset.counts(),
    )
    return dataset


def _explore(
    problem: problem_mod.Problem,
    basin: int,
    config: GenerationConfig,
    rng: np.random.Generator,
) -> list[NDArray]:
    """One CMA-ES run, returning every design it proposed along the way."""
    start = problem.seed(basin, rng)
    objective = problem.sample_objective(basin, rng)
    strategy = cma.CMAEvolutionStrategy(
        start.tolist(),
        config.initial_step,
        {
            "popsize": config.population,
            "maxiter": config.n_generations,
            "verbose": -9,
            "seed": int(rng.integers(1, 2**31 - 1)),
        },
    )

    proposed: list[NDArray] = []
    while not strategy.stop():
        asked = strategy.ask()
        designs = np.asarray(asked, dtype=np.float64)
        strategy.tell(
            asked,
            [_augmented(problem, design, objective, config) for design in designs],
        )
        proposed.append(designs)
    return proposed


def _augmented(
    problem: problem_mod.Problem,
    design: NDArray,
    objective: problem_mod.Objective,
    config: GenerationConfig,
) -> float:
    violation = float(np.maximum(problem.violations(design), 0.0).sum())
    return objective(design) + config.penalty_weight * violation
