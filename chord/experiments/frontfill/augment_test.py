"""Tests for aspect-ratio augmentation.

The property worth pinning hardest is that rescaling actually moves the aspect ratio by
the factor it claims to, and that a source's metrics cannot be carried along with it.
"""

import numpy as np
import pandas as pd
import pytest

from ... import corpus as corpus_mod
from ... import features
from . import augment, gaps

N_ROWS = 24


def _corpus(aspect_ratio: float = 7.0) -> corpus_mod.Corpus:
    rng = np.random.default_rng(0)
    boundaries = rng.normal(scale=1e-2, size=(N_ROWS, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            corpus_mod.SPLIT_COLUMN: "train",
            corpus_mod.BASIN_COLUMN: "planar",
            augment.ASPECT_RATIO: np.full(N_ROWS, aspect_ratio),
            **{name: rng.normal(size=N_ROWS) for name in corpus_mod.CONDITION_COLUMNS},
        }
    )
    return corpus_mod.from_frame(frame)


def test_only_non_axisymmetric_modes_scale() -> None:
    """Scaling the m=0 modes would move the major radius instead of the cross-
    section."""
    m_grid, _ = features.fourier_index_grid()

    mask = augment.poloidal_mask()

    assert bool(np.all(mask == (m_grid != 0)))
    assert not mask[np.flatnonzero(m_grid == 0)].any()


def test_rescaling_leaves_the_axisymmetric_modes_alone() -> None:
    boundaries = np.ones((3, features.FEATURE_DIM))

    scaled = augment.rescale(boundaries, np.array([0.5, 2.0, 1.0]))

    axisymmetric = ~augment.poloidal_mask()
    np.testing.assert_allclose(scaled[:, axisymmetric], 1.0)


def test_rescaling_multiplies_the_cross_section_modes() -> None:
    boundaries = np.ones((2, features.FEATURE_DIM))

    scaled = augment.rescale(boundaries, np.array([0.5, 2.0]))

    poloidal = augment.poloidal_mask()
    np.testing.assert_allclose(scaled[0, poloidal], 0.5)
    np.testing.assert_allclose(scaled[1, poloidal], 2.0)


def test_rescaling_does_not_touch_the_input() -> None:
    boundaries = np.ones((2, features.FEATURE_DIM))

    augment.rescale(boundaries, np.array([0.5, 2.0]))

    np.testing.assert_allclose(boundaries, 1.0)


def test_scale_factor_shrinks_to_raise_the_aspect_ratio() -> None:
    """Aspect ratio is major over minor, so a smaller cross-section means a larger
    ratio."""
    factor = augment.scale_factor(np.array([7.0]), np.array([14.0]))

    assert float(factor[0]) == pytest.approx(0.5)


def test_scale_factor_is_one_when_nothing_changes() -> None:
    factor = augment.scale_factor(np.array([9.0]), np.array([9.0]))

    assert float(factor[0]) == pytest.approx(1.0)


def test_a_non_positive_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        augment.scale_factor(np.array([7.0]), np.array([0.0]))


def test_targets_stay_inside_the_interval() -> None:
    native = np.full(16, 7.0)

    targets = augment.sample_targets(native, 9.0, 11.0, per_source=6)

    assert targets.shape == (16, 6)
    assert targets.min() >= 9.0
    assert targets.max() <= 11.0


def test_targets_cover_the_interval_rather_than_clustering() -> None:
    """Half the targets are spread so a model sees the interior, not just the
    anchors."""
    native = np.full(256, 10.0)

    targets = augment.sample_targets(native, 9.0, 11.0, per_source=6)

    assert targets.max() - targets.min() > 1.5


def test_the_first_target_is_the_source_itself() -> None:
    """An unrescaled control in every group keeps the augmentation honest."""
    native = np.array([9.5, 10.5])

    targets = augment.sample_targets(native, 9.0, 11.0, per_source=4)

    np.testing.assert_allclose(targets[:, 0], native)


def test_an_inverted_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="low < high"):
        augment.sample_targets(np.array([7.0]), 11.0, 9.0)


def test_augmentation_produces_one_boundary_per_target() -> None:
    corpus = _corpus()

    augmented = augment.augment(corpus, 9.0, 11.0, per_source=3)

    assert len(augmented) == N_ROWS * 3
    assert augmented.boundaries.shape == (N_ROWS * 3, features.FEATURE_DIM)


def test_augmentation_records_where_each_boundary_came_from() -> None:
    corpus = _corpus()

    augmented = augment.augment(corpus, 9.0, 11.0, per_source=3)

    assert set(augmented.source_row) <= set(range(N_ROWS))
    assert len(augmented.source_row) == len(augmented)


def test_augmentation_carries_no_metrics() -> None:
    """Rescaling changes the physics, so a source's labels describe something else."""
    fields = {field for field in augment.Augmentation.__dataclass_fields__}

    assert "metrics" not in fields
    assert fields == {"boundaries", "target_aspect_ratio", "source_row"}


def test_augmentation_keeps_the_structural_zeros() -> None:
    """Scaling must not put anything where stellarator symmetry forbids it."""
    corpus = _corpus()

    augmented = augment.augment(corpus, 9.0, 11.0, per_source=2)

    zeros = augmented.boundaries[:, features.structurally_zero_mask()]
    assert bool(np.all(zeros == 0.0))


def test_a_corpus_without_an_aspect_ratio_cannot_be_rescaled() -> None:
    corpus = _corpus()
    without = corpus_mod.Corpus(
        boundaries=corpus.boundaries,
        metrics=corpus.metrics.drop(columns=[augment.ASPECT_RATIO]),
        split=corpus.split,
        basin=corpus.basin,
    )

    with pytest.raises(KeyError, match=augment.ASPECT_RATIO):
        augment.augment(without, 9.0, 11.0)


def test_sampling_limits_the_number_of_sources() -> None:
    corpus = _corpus()

    augmented = augment.augment(corpus, 9.0, 11.0, per_source=2, max_sources=5)

    assert len(set(augmented.source_row)) == 5
    assert len(augmented) == 10


def test_augmenting_for_a_gap_uses_its_anchors() -> None:
    corpus = _corpus()
    gap = gaps.all_gaps()[0]

    augmented = augment.augment_for_gap(corpus, gap, per_source=2)

    assert augmented.target_aspect_ratio.min() >= gap.low
    assert augmented.target_aspect_ratio.max() <= gap.high


def test_augmentation_is_reproducible() -> None:
    corpus = _corpus()

    first = augment.augment(corpus, 9.0, 11.0, per_source=3)
    second = augment.augment(corpus, 9.0, 11.0, per_source=3)

    np.testing.assert_array_equal(first.boundaries, second.boundaries)
