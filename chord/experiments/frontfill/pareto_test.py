"""Tests for Pareto front scoring.

The hypervolume here is written in the natural orientation of the two metrics, while the
published benchmark states its reference in minimization coordinates. That rewrite is
exactly where a sign error would hide, so the first test holds it against a direct
transcription of the benchmark's own sweep.
"""

import numpy as np
import pytest

from . import pareto

REFERENCE_IN_MINIMIZATION_SPACE = (1.0, 20.0)


def _benchmark_hypervolume(
    aspect_ratio: np.ndarray, gradient_scale_length: np.ndarray
) -> float:
    """Transcription of the benchmark's sweep, in its own coordinates."""
    reference = np.array(REFERENCE_IN_MINIMIZATION_SPACE)
    points = np.array(
        [(-length, ratio) for ratio, length in zip(aspect_ratio, gradient_scale_length)]
    )
    if not len(points):
        return 0.0
    points = points[np.all(points < reference, axis=1)]
    if not len(points):
        return 0.0
    points = points[np.argsort(points[:, 0])]
    area = 0.0
    frontier = reference[1]
    for negated_length, ratio in points:
        if ratio < frontier:
            area += (reference[0] - negated_length) * (frontier - ratio)
            frontier = ratio
    return float(area)


def _front(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(6.0, 22.0, n), rng.uniform(-3.0, 8.0, n)


@pytest.mark.parametrize("seed", range(8))
def test_hypervolume_matches_the_benchmark_sweep(seed: int) -> None:
    aspect_ratio, gradient_scale_length = _front(64, seed)

    assert pareto.hypervolume(aspect_ratio, gradient_scale_length) == pytest.approx(
        _benchmark_hypervolume(aspect_ratio, gradient_scale_length)
    )


def test_reference_scale_length_is_negative() -> None:
    """Reading the benchmark's first minimization coordinate as a scale length halves
    it."""
    assert pareto.REFERENCE_GRADIENT_SCALE_LENGTH == -REFERENCE_IN_MINIMIZATION_SPACE[0]
    assert pareto.REFERENCE_GRADIENT_SCALE_LENGTH < 0


def test_one_design_covers_its_rectangle() -> None:
    area = pareto.hypervolume(np.array([10.0]), np.array([4.0]))

    width = 4.0 - pareto.REFERENCE_GRADIENT_SCALE_LENGTH
    height = pareto.REFERENCE_ASPECT_RATIO - 10.0
    assert area == pytest.approx(width * height)


def test_a_design_outside_the_reference_contributes_nothing() -> None:
    worse_aspect_ratio = pareto.hypervolume(np.array([25.0]), np.array([4.0]))
    worse_scale_length = pareto.hypervolume(np.array([10.0]), np.array([-2.0]))

    assert worse_aspect_ratio == 0.0
    assert worse_scale_length == 0.0


def test_an_empty_front_scores_zero() -> None:
    assert pareto.hypervolume(np.array([]), np.array([])) == 0.0


def test_a_dominated_design_adds_nothing() -> None:
    alone = pareto.hypervolume(np.array([10.0]), np.array([4.0]))

    with_worse = pareto.hypervolume(np.array([10.0, 12.0]), np.array([4.0, 3.0]))

    assert with_worse == pytest.approx(alone)


def test_a_design_beyond_the_front_adds_area() -> None:
    alone = pareto.hypervolume(np.array([10.0]), np.array([4.0]))

    with_better = pareto.hypervolume(np.array([10.0, 8.0]), np.array([4.0, 2.0]))

    assert with_better > alone


def test_non_dominated_keeps_only_the_frontier() -> None:
    aspect_ratio = np.array([8.0, 10.0, 12.0, 11.0])
    gradient_scale_length = np.array([2.0, 4.0, 6.0, 3.0])

    kept = pareto.non_dominated(aspect_ratio, gradient_scale_length)

    # 11.0/3.0 is beaten by 10.0/4.0 on both axes; the other three trade off.
    assert kept.tolist() == [True, True, True, False]


def test_identical_designs_do_not_dominate_each_other() -> None:
    """The count of non-dominated designs is reported, so duplicates must all
    survive."""
    kept = pareto.non_dominated(np.array([10.0, 10.0]), np.array([4.0, 4.0]))

    assert kept.tolist() == [True, True]


def test_non_dominated_of_nothing_is_empty() -> None:
    assert pareto.non_dominated(np.array([]), np.array([])).tolist() == []


def test_recovered_fraction_is_the_share_of_what_was_available() -> None:
    assert pareto.recovered_fraction(100.0, 140.0, 180.0) == pytest.approx(0.5)


def test_recovered_fraction_is_undefined_when_nothing_was_available() -> None:
    """The held-out campaigns adding nothing leaves no gap to express a fraction of."""
    assert pareto.recovered_fraction(100.0, 120.0, 100.0) is None


def test_the_scale_length_name_is_the_field_period_scaled_one() -> None:
    """A front built from raw values would sit a factor of three low and score near
    zero."""
    assert pareto.GRADIENT_SCALE_LENGTH.startswith("minimum_normalized")
