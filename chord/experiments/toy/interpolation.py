"""Measure feasibility along latent chords between held-out designs.

The question these benchmarks exist to answer is what happens between two designs.
Encode a held-out pair, walk the straight line between their latent codes, decode every
step, and check each decoded design against the problem's analytical constraints. A
latent that preserves feasibility keeps the whole path inside the feasible set; one that
does not fails somewhere in the middle, which is why the middle is reported separately
from the whole path.

Pairs are drawn across basins. Endpoints from the same basin can often be joined by a
path that never leaves it, so a model can look good on them without having learned
anything about the geometry.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import torch
from numpy.typing import NDArray

from ... import inference as inference_mod
from . import datasets, training
from . import problem as problem_mod

MIDDLE = (0.3, 0.7)
"""Range of the path treated as the interior, away from the endpoints.

The endpoints are training-like designs and are feasible almost by construction, so
including them flatters every model equally and hides the difference.
"""


@dataclasses.dataclass(frozen=True)
class Chord:
    """Decoded paths between endpoint pairs, and whether each step is feasible."""

    alphas: NDArray
    designs: NDArray
    feasible: NDArray

    def __len__(self) -> int:
        return len(self.feasible)

    def full_path(self) -> float:
        """Fraction of all decoded steps that are feasible."""
        return float(self.feasible.mean()) if len(self) else float("nan")

    def middle(self) -> float:
        """Fraction feasible over the interior of the path."""
        if not len(self):
            return float("nan")
        inside = (self.alphas >= np.nextafter(MIDDLE[0], -np.inf)) & (
            self.alphas <= np.nextafter(MIDDLE[1], np.inf)
        )
        return float(self.feasible[:, inside].mean())

    def fully_feasible(self) -> float:
        """Fraction of pairs for which all evaluated steps are feasible."""
        return float(self.feasible.all(axis=-1).mean()) if len(self) else float("nan")


def evaluate(
    problem: problem_mod.Problem,
    trained: training.Trained,
    dataset: datasets.Dataset,
    rows_a: NDArray,
    rows_b: NDArray,
    n_alpha: int = 21,
) -> Chord:
    """Decode the chord between each paired pair of rows and check feasibility."""
    alphas = np.linspace(0.0, 1.0, n_alpha)
    if not len(rows_a):
        empty = np.zeros((0, n_alpha), dtype=bool)
        return Chord(
            alphas=alphas, designs=np.zeros((0, n_alpha, problem.n_dim)), feasible=empty
        )

    z_a = trained.encode(dataset.designs[rows_a], dataset.conditions[rows_a])
    z_b = trained.encode(dataset.designs[rows_b], dataset.conditions[rows_b])
    y_a = trained.standardized_conditions(dataset.conditions[rows_a])
    y_b = trained.standardized_conditions(dataset.conditions[rows_b])

    weights = torch.as_tensor(alphas, dtype=torch.float32).reshape(-1, 1, 1)
    z = ((1.0 - weights) * z_a + weights * z_b).reshape(-1, z_a.shape[-1])
    # An unconditional decoder takes a zero-width vector, which cannot be
    # reshaped from a flattened stack because it holds no elements to count.
    if y_a.shape[-1] == 0:
        y = torch.zeros(n_alpha * len(rows_a), 0)
    else:
        y = inference_mod.linear_conditioning(
            y_a.expand(n_alpha, *y_a.shape), y_b.expand(n_alpha, *y_b.shape), weights
        ).reshape(-1, y_a.shape[-1])

    with torch.no_grad():
        decoded = trained.design_scaler.decode(trained.model.decode(z, y)).numpy()
    designs = decoded.reshape(n_alpha, len(rows_a), problem.n_dim).transpose(1, 0, 2)
    return Chord(
        alphas=alphas,
        designs=designs,
        feasible=problem_mod.is_feasible(problem, designs),
    )


def across_basins(
    problem: problem_mod.Problem,
    trained: training.Trained,
    dataset: datasets.Dataset,
    n_pairs: int = 30,
    n_alpha: int = 21,
    seed: int = 0,
) -> dict[tuple[int, int], Chord]:
    """Evaluate every unordered pair of basins, on held-out rows only."""
    rng = np.random.default_rng(seed)
    held_out = trained.validation
    by_basin = {
        basin: held_out[dataset.basin[held_out] == basin]
        for basin in range(problem.n_basins)
    }

    results: dict[tuple[int, int], Chord] = {}
    for first, second in itertools.combinations(range(problem.n_basins), 2):
        available = min(n_pairs, len(by_basin[first]), len(by_basin[second]))
        rows_a = rng.choice(by_basin[first], available, replace=False)
        rows_b = rng.choice(by_basin[second], available, replace=False)
        results[(first, second)] = evaluate(
            problem, trained, dataset, rows_a, rows_b, n_alpha
        )
    return results


def summarize(results: dict[tuple[int, int], Chord]) -> dict[str, float]:
    """Pool the basin pairs into the numbers the paper reports."""
    chords = [chord for chord in results.values() if len(chord)]
    if not chords:
        return {"full_path": float("nan"), "middle": float("nan"), "n_pairs": 0}
    weights = np.array([len(chord) for chord in chords], dtype=np.float64)
    return {
        "full_path": float(
            np.average([chord.full_path() for chord in chords], weights=weights)
        ),
        "middle": float(
            np.average([chord.middle() for chord in chords], weights=weights)
        ),
        "fully_feasible": float(
            np.average([chord.fully_feasible() for chord in chords], weights=weights)
        ),
        "n_pairs": int(weights.sum()),
    }
