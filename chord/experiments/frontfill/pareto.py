"""Scoring a Pareto front over aspect ratio and magnetic gradient scale length.

The question front-filling asks is how much of a front a few campaigns plus generated
designs recover, and hypervolume is how that is measured: the area dominated by a front,
bounded by a fixed reference point, so two fronts are comparable as single numbers.

Designs are better with smaller aspect ratio and larger gradient scale length. The
reference point is the corner both must beat to count at all, and it has to match the
published benchmark's exactly or the numbers mean nothing next to it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

REFERENCE_ASPECT_RATIO = 20.0
"""Aspect ratio a design must beat to contribute any hypervolume."""

REFERENCE_GRADIENT_SCALE_LENGTH = -1.0
"""Gradient scale length a design must exceed to contribute any hypervolume.

Negative, and that sign is the whole of it. The benchmark states its reference in
minimization coordinates, where the first axis is the *negated* scale length, so the
published pair reads `(1.0, 20.0)`. Taking that first entry at face value as a scale
length of `+1.0` shortens every strip of the sweep by 2.0 and understates the front
substantially. The quantity is stored here in its natural orientation so the sign is
visible rather than implied by a coordinate convention.
"""

GRADIENT_SCALE_LENGTH = "minimum_normalized_magnetic_gradient_scale_length"
"""Metric name, under constellaration's convention, which is scaled by field periods.

A front built from raw unscaled values would sit a factor of the field-period count
below this reference and score near zero.
"""


def non_dominated(aspect_ratio: NDArray, gradient_scale_length: NDArray) -> NDArray:
    """Mask over designs that nothing else beats.

    A design is dominated when another is at least as good on both axes and strictly
    better on one. Duplicates therefore do not dominate each other and are all kept,
    which matters because the count of non-dominated designs is reported.

    Quadratic in the number of designs, which is fine for a front of hundreds and would
    not be for a corpus.
    """
    aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    gradient_scale_length = np.asarray(gradient_scale_length, dtype=float)
    if not len(aspect_ratio):
        return np.zeros(0, dtype=bool)

    at_least_as_good = (aspect_ratio[None, :] <= aspect_ratio[:, None]) & (
        gradient_scale_length[None, :] >= gradient_scale_length[:, None]
    )
    strictly_better = (aspect_ratio[None, :] < aspect_ratio[:, None]) | (
        gradient_scale_length[None, :] > gradient_scale_length[:, None]
    )
    return ~np.asarray((at_least_as_good & strictly_better).any(axis=1))


def hypervolume(aspect_ratio: NDArray, gradient_scale_length: NDArray) -> float:
    """Area a set of designs dominates, bounded by the reference point.

    Designs that fail to beat the reference on either axis contribute nothing rather
    than contributing negatively.
    """
    aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    gradient_scale_length = np.asarray(gradient_scale_length, dtype=float)
    inside = (aspect_ratio < REFERENCE_ASPECT_RATIO) & (
        gradient_scale_length > REFERENCE_GRADIENT_SCALE_LENGTH
    )
    if not inside.any():
        return 0.0

    # Sweep from the largest scale length downward, accumulating a strip each time
    # the aspect ratio improves on everything seen so far.
    order = np.argsort(-gradient_scale_length[inside])
    swept_aspect_ratio = aspect_ratio[inside][order]
    swept_scale_length = gradient_scale_length[inside][order]

    area = 0.0
    frontier = REFERENCE_ASPECT_RATIO
    for ratio, length in zip(swept_aspect_ratio, swept_scale_length):
        if ratio < frontier:
            area += (length - REFERENCE_GRADIENT_SCALE_LENGTH) * (frontier - ratio)
            frontier = ratio
    return float(area)


def recovered_fraction(
    anchors_only: float, anchors_plus_generated: float, full_campaign: float
) -> float | None:
    """How much of what the held-out campaigns add is recovered generatively.

    Returns `None` when the held-out campaigns add nothing, since there is then no gap
    to express a fraction of.
    """
    available = full_campaign - anchors_only
    if available <= 0:
        return None
    return (anchors_plus_generated - anchors_only) / available
