"""The paper's two-dimensional C-shaped annulus.

The annulus excludes a wedge around the positive horizontal axis. Its upper and
lower arms are connected around the left side, but a straight chord between them
can cross the forbidden inner disc. The default feasible set contains no inner disc.

A positive `disc_radius` adds a disconnected feasible component for optional
experiments; it is not part of the manuscript's C-shape diagnostic.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from . import problem as problem_mod

LOWER_ARM = 0
UPPER_ARM = 1
DISC = 2

_SEED_ATTEMPTS = 2000


@dataclasses.dataclass(frozen=True)
class CShape:
    """A C-shaped corridor with an optional disconnected inner disc."""

    inner_radius: float = 0.85
    outer_radius: float = 1.15
    wedge_half_angle: float = 0.45
    """Angular half-width of the forbidden wedge, about 26 degrees."""

    disc_center: tuple[float, float] = (0.0, 0.0)
    disc_radius: float = 0.0
    """A positive radius enables the optional feasible disc."""

    n_dim: int = 2
    nuisance_bound: float = 0.3
    """Bound on each coordinate beyond the first two, when there are any.

    Extra dimensions carry no geometry. They exist to check that the method does not
    depend on the feasible set being visualisable.
    """

    name: ClassVar[str] = "cshape"

    @property
    def n_basins(self) -> int:
        return 3 if self.disc_radius > 0 else 2

    @property
    def constraint_names(self) -> tuple[str, ...]:
        names = ["membership"]
        for index in range(2, self.n_dim):
            names.extend([f"nuisance_{index}_upper", f"nuisance_{index}_lower"])
        return tuple(names)

    def violations(self, x: NDArray) -> NDArray:
        radius = np.linalg.norm(x[..., :2], axis=-1)
        corridor = np.stack(
            [
                self.inner_radius - radius,
                radius - self.outer_radius,
                self.wedge_half_angle - self._absolute_angle(x),
            ],
            axis=-1,
        ).max(axis=-1)
        parts = [self._membership(corridor, self._disc_slack(x))]
        for index in range(2, self.n_dim):
            parts.append(x[..., index] - self.nuisance_bound)
            parts.append(-self.nuisance_bound - x[..., index])
        return np.stack(parts, axis=-1)

    def violations_torch(self, x: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(x[..., :2], dim=-1)
        corridor = (
            torch.stack(
                [
                    self.inner_radius - radius,
                    radius - self.outer_radius,
                    self.wedge_half_angle - torch.atan2(x[..., 1], x[..., 0]).abs(),
                ],
                dim=-1,
            )
            .max(dim=-1)
            .values
        )
        if self.disc_radius > 0:
            center = x.new_tensor(self.disc_center)
            disc = (
                torch.linalg.vector_norm(x[..., :2] - center, dim=-1) ** 2
                - self.disc_radius**2
            )
            membership = torch.minimum(corridor, disc)
        else:
            membership = corridor
        parts = [membership]
        for index in range(2, self.n_dim):
            parts.append(x[..., index] - self.nuisance_bound)
            parts.append(-self.nuisance_bound - x[..., index])
        return torch.stack(parts, dim=-1)

    def conditions(self, x: NDArray) -> NDArray:
        """Radius, absolute angle, squared height, and the nuisance energy if any."""
        parts = [
            np.linalg.norm(x[..., :2], axis=-1),
            self._absolute_angle(x),
            x[..., 1] ** 2,
        ]
        if self.n_dim > 2:
            parts.append((x[..., 2:] ** 2).sum(axis=-1))
        return np.stack(parts, axis=-1)

    def basin(self, x: NDArray) -> NDArray:
        inside_disc = self._disc_slack(x) <= 0
        if self.disc_radius <= 0:
            inside_disc = np.zeros(x.shape[:-1], dtype=bool)
        arm = np.where(x[..., 1] >= 0, UPPER_ARM, LOWER_ARM)
        return np.where(inside_disc, DISC, arm).astype(np.int64)

    def seed(self, basin: int, rng: np.random.Generator) -> NDArray:
        if basin not in range(self.n_basins):
            raise ValueError(f"basin must be below {self.n_basins}, got {basin}")
        for _ in range(_SEED_ATTEMPTS):
            candidate = (
                self._disc_candidate(rng)
                if basin == DISC
                else self._corridor_candidate(basin, rng)
            )
            if bool(problem_mod.is_feasible(self, candidate)):
                return candidate
        raise RuntimeError(
            f"no feasible seed found in basin {basin} after {_SEED_ATTEMPTS} attempts; "
            f"the configured geometry may leave it empty"
        )

    def sample_objective(
        self, basin: int, rng: np.random.Generator
    ) -> problem_mod.Objective:
        """Push outward along the horizontal axis, the same target from every basin.

        A single shared objective is what makes the dataset realistic: every basin is
        explored by a search that wants the same thing and is stopped by different
        geometry, which is how the density ends up uneven.
        """
        del basin, rng
        return _horizontal_extent

    def _membership(self, corridor: NDArray, disc: NDArray) -> NDArray:
        return np.minimum(corridor, disc) if self.disc_radius > 0 else corridor

    def _disc_slack(self, x: NDArray) -> NDArray:
        center = np.asarray(self.disc_center)
        return ((x[..., :2] - center) ** 2).sum(axis=-1) - self.disc_radius**2

    def _absolute_angle(self, x: NDArray) -> NDArray:
        return np.abs(np.arctan2(x[..., 1], x[..., 0]))

    def _corridor_candidate(self, basin: int, rng: np.random.Generator) -> NDArray:
        radius = rng.uniform(self.inner_radius, self.outer_radius)
        angle = rng.uniform(self.wedge_half_angle, np.pi)
        if basin == LOWER_ARM:
            angle = -angle
        return self._with_nuisance(radius * np.cos(angle), radius * np.sin(angle), rng)

    def _disc_candidate(self, rng: np.random.Generator) -> NDArray:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = self.disc_radius * np.sqrt(rng.uniform())
        return self._with_nuisance(
            self.disc_center[0] + radius * np.cos(angle),
            self.disc_center[1] + radius * np.sin(angle),
            rng,
        )

    def _with_nuisance(
        self, first: float, second: float, rng: np.random.Generator
    ) -> NDArray:
        x = np.zeros(self.n_dim, dtype=np.float64)
        x[0], x[1] = first, second
        if self.n_dim > 2:
            x[2:] = rng.normal(scale=0.1, size=self.n_dim - 2)
        return x


def _horizontal_extent(x: NDArray) -> float:
    """Push outward along the horizontal axis."""
    return float(-x[..., 0])
