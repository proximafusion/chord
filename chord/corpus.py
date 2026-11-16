"""Loading the published design corpus.

The corpus is a table of plasma boundaries with their simulated metrics. Each row
carries the raw Fourier coefficients of one boundary, the metrics the simulator returned
for it, which basin it belongs to, and which split it was assigned to. Splits are stored
rather than recomputed so that a reproduction trains on exactly the rows the paper
trained on.

Artifact locations resolve through a single base, overridable by environment variable.
That keeps the published location in one place, which matters when the same code has to
read from an anonymised mirror during double-blind review.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import artifacts, feasibility, features

BOUNDARY_COLUMN = "boundary_coefficients"
SPLIT_COLUMN = "split"
BASIN_COLUMN = "basin"

LOG10_QI = artifacts.LOG10_QI
"""Base-10 logarithm of constellaration's `qi`, which is how the paper reports it."""

EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO = (
    artifacts.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO
)
"""Axis non-planarity, and the one conditioning component with no benchmark equivalent.

The effective aspect ratio is that of an axisymmetric flat-axis configuration carrying
the same external poloidal current, so the difference against the true aspect ratio
grows with axis torsion. Constellaration does not compute it, so unlike the rest of the
conditioning vector this name is ours.
"""

MAX_ELONGATION = "max_elongation"

CONDITION_COLUMNS: tuple[str, ...] = (
    EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO,
    feasibility.IOTA_EDGE,
    LOG10_QI,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
    MAX_ELONGATION,
)
"""Metrics a conditional decoder is conditioned on."""

Split = Literal["train", "val", "test"]


@dataclasses.dataclass(frozen=True)
class Corpus:
    """Boundaries, their metrics, and the split each row belongs to."""

    boundaries: NDArray
    """Raw Fourier coefficients, one row per design."""

    metrics: pd.DataFrame
    split: NDArray
    basin: NDArray

    def __len__(self) -> int:
        return len(self.boundaries)

    def subset(self, split: Split) -> Corpus:
        """Rows belonging to one split, in their original order."""
        selected = self.split == split
        return Corpus(
            boundaries=self.boundaries[selected],
            metrics=self.metrics.loc[selected].reset_index(drop=True),
            split=self.split[selected],
            basin=self.basin[selected],
        )

    def conditions(self, columns: Sequence[str] = CONDITION_COLUMNS) -> NDArray:
        """Conditioning vectors, in the column order given."""
        missing = [c for c in columns if c not in self.metrics.columns]
        if missing:
            raise KeyError(
                f"corpus is missing conditioning columns {missing}; it carries "
                f"{sorted(self.metrics.columns)}"
            )
        return self.metrics[list(columns)].to_numpy(dtype=np.float64)


def load_corpus(name: str = "corpus.parquet") -> Corpus:
    """Read the corpus from the configured artifact store, or from a direct path."""
    path = (
        name if "://" in name or os.path.exists(name) else artifacts.artifact_uri(name)
    )
    return from_frame(pd.read_parquet(path))


def from_frame(frame: pd.DataFrame) -> Corpus:
    """Build a corpus from an already-loaded table."""
    for column in (BOUNDARY_COLUMN, SPLIT_COLUMN, BASIN_COLUMN):
        if column not in frame.columns:
            raise KeyError(
                f"corpus table is missing the {column!r} column; it carries "
                f"{sorted(frame.columns)}"
            )

    boundaries = np.stack(frame[BOUNDARY_COLUMN].tolist()).astype(np.float64)
    if boundaries.shape[-1] != features.FEATURE_DIM:
        raise ValueError(
            f"boundaries have {boundaries.shape[-1]} coefficients, expected "
            f"{features.FEATURE_DIM}"
        )

    metric_columns = [
        c
        for c in frame.columns
        if c not in (BOUNDARY_COLUMN, SPLIT_COLUMN, BASIN_COLUMN)
    ]
    return Corpus(
        boundaries=boundaries,
        metrics=frame[metric_columns].reset_index(drop=True),
        split=frame[SPLIT_COLUMN].to_numpy(),
        basin=frame[BASIN_COLUMN].to_numpy(),
    )
