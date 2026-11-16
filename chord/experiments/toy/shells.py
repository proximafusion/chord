"""A higher-dimensional benchmark with several disconnected basins.

The design vector splits into equal slices, one per mechanism. A softmax over slice
energies decides which mechanism is in charge, and feasibility demands that one of them
clearly is. That requirement is what disconnects the basins: a design halfway between
two mechanisms is routed ambiguously and fails, so there is no feasible path from one to
another.

Each mechanism maps its slice through the same smooth nonlinearity and lands it near a
vertex of a regular polygon in output space. An inner forbidden disc around each vertex
turns the per-basin feasible set into an annulus, which matters because it makes even
within-basin interpolation able to fail. Without it a decoder could satisfy the
benchmark by collapsing each basin to its centre.

Where the C-shape is two-dimensional and can be plotted, this one is sixteen by default
and cannot. It is here to show the geometry rather than the dimension is what the method
responds to.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from . import problem as problem_mod

_SEED_ATTEMPTS = 4000
_DOMINANT_SCALE = 2.0
_RECESSIVE_SCALE = 0.05


@dataclasses.dataclass(frozen=True)
class Shells:
    """Disconnected mechanism basins, each feasible on an annulus."""

    n_mechanisms: int = 4
    dims_per_mechanism: int = 4
    offset_radius: float = 0.6
    """Radius of the polygon of basin centres in output space."""

    softmax_temperature: float = 0.5
    purity_threshold: float = 0.85
    """How clearly one mechanism must dominate.

    Above about 0.85 the basins separate.
    """

    output_bound: float = 1.4
    inner_radius: float = 0.6
    """Radius of the forbidden disc at each basin centre.

    Zero removes it.
    """

    name: ClassVar[str] = "shells"

    def __post_init__(self) -> None:
        # The activation folds a slice into a two-component output by summing
        # each half, which needs the halves to be the same size.
        if self.dims_per_mechanism % 2:
            raise ValueError(
                f"dims_per_mechanism must be even, got {self.dims_per_mechanism}"
            )

    @property
    def n_dim(self) -> int:
        return self.n_mechanisms * self.dims_per_mechanism

    @property
    def n_basins(self) -> int:
        return self.n_mechanisms

    @property
    def constraint_names(self) -> tuple[str, ...]:
        names = ["purity", "output_bound"]
        return (*names, "inner_radius") if self.inner_radius > 0 else tuple(names)

    def centers(self) -> NDArray:
        """Basin centres, evenly spaced on a circle in output space."""
        angles = 2.0 * np.pi * np.arange(self.n_mechanisms) / self.n_mechanisms
        return self.offset_radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def routing(self, x: NDArray) -> NDArray:
        """Softmax over slice energies, deciding which mechanism is in charge."""
        energies = (self._slices(x) ** 2).sum(axis=-1) / self.softmax_temperature
        shifted = np.exp(energies - energies.max(axis=-1, keepdims=True))
        return shifted / shifted.sum(axis=-1, keepdims=True)

    def simulate(self, x: NDArray) -> NDArray:
        """Output vector, a routed blend of the per-mechanism outputs."""
        routing = self.routing(x)
        outputs = _activation(self._slices(x)) + self.centers()
        return (routing[..., None] * outputs).sum(axis=-2)

    def violations(self, x: NDArray) -> NDArray:
        routing = self.routing(x)
        output = self.simulate(x)
        parts = [
            self.purity_threshold - routing.max(axis=-1),
            np.linalg.norm(output, axis=-1) - self.output_bound,
        ]
        if self.inner_radius > 0:
            offset = routing @ self.centers()
            parts.append(self.inner_radius - np.linalg.norm(output - offset, axis=-1))
        return np.stack(parts, axis=-1)

    def violations_torch(self, x: torch.Tensor) -> torch.Tensor:
        slices = x.reshape(*x.shape[:-1], self.n_mechanisms, self.dims_per_mechanism)
        energies = (slices**2).sum(dim=-1) / self.softmax_temperature
        routing = torch.softmax(energies, dim=-1)
        centers = x.new_tensor(self.centers())
        outputs = _activation_torch(slices) + centers
        output = (routing.unsqueeze(-1) * outputs).sum(dim=-2)

        parts = [
            self.purity_threshold - routing.max(dim=-1).values,
            torch.linalg.vector_norm(output, dim=-1) - self.output_bound,
        ]
        if self.inner_radius > 0:
            offset = routing @ centers
            parts.append(
                self.inner_radius - torch.linalg.vector_norm(output - offset, dim=-1)
            )
        return torch.stack(parts, dim=-1)

    def conditions(self, x: NDArray) -> NDArray:
        """The output vector itself, which is what a user of this problem asks for."""
        return self.simulate(x)

    def basin(self, x: NDArray) -> NDArray:
        return np.argmax(self.routing(x), axis=-1).astype(np.int64)

    def seed(self, basin: int, rng: np.random.Generator) -> NDArray:
        if basin not in range(self.n_basins):
            raise ValueError(f"basin must be below {self.n_basins}, got {basin}")
        for _ in range(_SEED_ATTEMPTS):
            scales = np.full(self.n_mechanisms, _RECESSIVE_SCALE)
            scales[basin] = _DOMINANT_SCALE if self.inner_radius > 0 else 1.0
            candidate = np.concatenate(
                [
                    rng.normal(scale=scale, size=self.dims_per_mechanism)
                    for scale in scales
                ]
            )
            if (
                bool(problem_mod.is_feasible(self, candidate))
                and int(self.basin(candidate)) == basin
            ):
                return candidate
        raise RuntimeError(
            f"no feasible seed found in basin {basin} after {_SEED_ATTEMPTS} attempts; "
            f"the configured geometry may leave it empty"
        )

    def sample_objective(
        self, basin: int, rng: np.random.Generator
    ) -> problem_mod.Objective:
        """Aim at a point drawn along the ray out to this basin's centre.

        Drawing the target rather than fixing it spreads the search over the annulus
        instead of piling every run onto the same attractor.
        """
        target = rng.uniform(0.4, 1.0) * self.centers()[basin]
        return functools.partial(_distance_to_target, self.simulate, target)

    def _slices(self, x: NDArray) -> NDArray:
        return x.reshape(*x.shape[:-1], self.n_mechanisms, self.dims_per_mechanism)


def _distance_to_target(
    simulate: Callable[[NDArray], NDArray], target: NDArray, x: NDArray
) -> float:
    """Squared distance from a design's output to the point the search aims at.

    Written at module level and bound with `functools.partial` rather than closed over,
    so the objective stays inspectable and can cross a process boundary.
    """
    return float(((simulate(x) - target) ** 2).sum())


def _activation(slices: NDArray) -> NDArray:
    """Fold each slice into a two-component output by summing its halves."""
    halves = slices.reshape(*slices.shape[:-1], 2, slices.shape[-1] // 2)
    return np.tanh(halves.sum(axis=-1))


def _activation_torch(slices: torch.Tensor) -> torch.Tensor:
    halves = slices.reshape(*slices.shape[:-1], 2, slices.shape[-1] // 2)
    return torch.tanh(halves.sum(dim=-1))
