"""Expand a corpus across aspect ratio by rescaling and relabelling.

A corpus built from optimization campaigns clusters at the aspect ratios those campaigns
targeted, so it carries almost no signal between them. That is precisely where an
aspect-ratio-conditioned model is asked to work, which makes the gaps the thing to fill.

Filling them does not need more campaigns. Scaling the non-axisymmetric Fourier modes of
a boundary by a single factor moves its aspect ratio, and the factor is known in closed
form: shrinking the minor radius by `s` raises the aspect ratio by `1 / s`. One cheap
array multiply per target replaces an optimization run.

The scaled geometry is only half of it, and the dangerous half is the other one. A
rescaled boundary is a different physical configuration, so its metrics are not its
source's. Every rescaled boundary has to go back through the simulator. `Augmentation`
therefore carries no metrics at all; the only way to get them is `label`, which runs the
simulator, so reusing a source's labels is not something a caller can do by accident.

The closed-form factor lands within roughly 0.01 of the requested aspect ratio on
average. That approximation costs nothing, because what a boundary ends up with is read
back from the simulator rather than assumed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ... import corpus as corpus_mod
from ... import evaluate as evaluate_mod
from ... import features
from . import gaps as gaps_mod

_LOG = logging.getLogger(__name__)

ASPECT_RATIO = "aspect_ratio"
TARGET_ASPECT_RATIO = "target_aspect_ratio"
SOURCE_ROW = "source_row"
CONVERGED = "converged"


@dataclasses.dataclass(frozen=True)
class Augmentation:
    """Rescaled boundaries, deliberately without metrics.

    Scaling changes the physics, so a source's labels do not transfer. Leaving them out
    entirely is what makes that impossible to forget.
    """

    boundaries: NDArray
    target_aspect_ratio: NDArray
    """What each boundary was aimed at, not what it achieved."""

    source_row: NDArray
    """Index of the corpus row each boundary came from, kept for provenance."""

    def __len__(self) -> int:
        return len(self.boundaries)


def poloidal_mask() -> NDArray:
    """Coefficients that scale with the minor radius, which is every `m != 0` mode.

    The `m = 0` modes set where the surface sits rather than how big its cross-section
    is, so scaling them would move the major radius and leave the aspect ratio roughly
    where it started.
    """
    m_grid, _ = features.fourier_index_grid()
    return m_grid != 0


def scale_factor(native_aspect_ratio: NDArray, target_aspect_ratio: NDArray) -> NDArray:
    """Factor taking a boundary from its own aspect ratio to the requested one."""
    native = np.asarray(native_aspect_ratio, dtype=float)
    target = np.asarray(target_aspect_ratio, dtype=float)
    if np.any(target <= 0):
        raise ValueError("target aspect ratios must be positive")
    return native / target


def rescale(boundaries: NDArray, factors: NDArray) -> NDArray:
    """Scale the non-axisymmetric modes of each boundary by its own factor."""
    scaled = np.array(boundaries, dtype=np.float64, copy=True)
    mask = poloidal_mask()
    scaled[:, mask] *= np.asarray(factors, dtype=float).reshape(-1, 1)
    return scaled


def sample_targets(
    native_aspect_ratio: NDArray,
    low: float,
    high: float,
    per_source: int = 5,
    local_fraction: float = 0.5,
    local_spread: float = 0.25,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Aspect ratios to aim each source boundary at, shaped `(sources, per_source)`.

    Half the targets sit near the source's own aspect ratio and half are spread across
    the whole interval. The local half keeps continuity around each source, where a
    small rescale stays close to a configuration known to work; the spread half covers
    the interior and deliberately pushes some boundaries off the feasible manifold, so a
    model trained on the result learns where feasibility ends rather than only where it
    holds.

    The first target of every source is its own aspect ratio, which gives an unrescaled
    control in the set.
    """
    if not low < high:
        raise ValueError(f"expected low < high, got [{low}, {high}]")
    if per_source < 1:
        raise ValueError(f"per_source must be at least 1, got {per_source}")
    if rng is None:
        rng = np.random.default_rng(0)
    native = np.asarray(native_aspect_ratio, dtype=float).reshape(-1, 1)

    n_local = max(1, int(round(per_source * local_fraction)))
    local = rng.normal(native, local_spread, size=(len(native), n_local))
    spread = rng.uniform(low, high, size=(len(native), per_source - n_local))
    targets = np.clip(np.concatenate([local, spread], axis=1), low, high)
    targets[:, 0] = np.clip(native[:, 0], low, high)
    return targets


def augment(
    corpus: corpus_mod.Corpus,
    low: float,
    high: float,
    per_source: int = 5,
    max_sources: int | None = None,
    rng: np.random.Generator | None = None,
) -> Augmentation:
    """Rescale a corpus across an aspect-ratio interval.

    The scale factor comes from each row's recorded aspect ratio, so no geometry solver
    is needed and the corpus's own labels supply everything.
    """
    if ASPECT_RATIO not in corpus.metrics.columns:
        raise KeyError(
            f"the corpus has no {ASPECT_RATIO!r} column, so there is nothing to "
            f"rescale from; it carries {sorted(corpus.metrics.columns)}"
        )
    if rng is None:
        rng = np.random.default_rng(0)
    rows = np.arange(len(corpus))
    if max_sources is not None and max_sources < len(rows):
        rows = np.sort(rng.choice(rows, max_sources, replace=False))

    native = corpus.metrics[ASPECT_RATIO].to_numpy()[rows]
    targets = sample_targets(native, low, high, per_source, rng=rng)
    repeated = np.repeat(rows, targets.shape[1])
    flat_targets = targets.reshape(-1)

    boundaries = rescale(
        corpus.boundaries[repeated],
        scale_factor(native.repeat(targets.shape[1]), flat_targets),
    )
    _LOG.info(
        "rescaled %d sources into %d boundaries across [%.2f, %.2f]",
        len(rows),
        len(boundaries),
        low,
        high,
    )
    return Augmentation(
        boundaries=boundaries,
        target_aspect_ratio=flat_targets,
        source_row=repeated,
    )


def augment_for_gap(
    corpus: corpus_mod.Corpus, gap: gaps_mod.Gap, **kwargs: Any
) -> Augmentation:
    """Rescale across one interval, which is all a per-gap model may ever see."""
    return augment(corpus, gap.low, gap.high, **kwargs)


def label(
    augmentation: Augmentation,
    settings: Any | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Run the simulator on rescaled boundaries and return what it found.

    This is the expensive half and there is no way around it. Rescaling changes a
    boundary's physics, so its source's metrics describe a different configuration.

    The aspect ratio in the result is the one achieved, which is what makes the closed-
    form scale factor good enough despite being approximate.
    """
    boundaries = [
        evaluate_mod.boundary_from_vector(design) for design in augmentation.boundaries
    ]
    evaluations = evaluate_mod.evaluate_batch(
        boundaries, settings=settings, n_workers=n_workers
    )

    frame = pd.DataFrame(evaluate_mod.corpus_metrics(evaluations))
    frame[corpus_mod.BOUNDARY_COLUMN] = list(augmentation.boundaries)
    frame[TARGET_ASPECT_RATIO] = augmentation.target_aspect_ratio
    frame[SOURCE_ROW] = augmentation.source_row
    frame[CONVERGED] = [evaluation.converged for evaluation in evaluations]
    _LOG.info(
        "labelled %d boundaries, %d converged",
        len(frame),
        int(frame[CONVERGED].sum()),
    )
    return frame
