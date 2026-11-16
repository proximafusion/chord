"""Tests for per-gap selection.

The case that matters most is the one an aspect-ratio filter gets wrong: a boundary from
a held-out campaign, rescaled onto an anchor's aspect ratio, is still held-out
information and must be excluded.
"""

import numpy as np
import pandas as pd
import pytest

from ... import corpus as corpus_mod
from ... import features
from . import augment, gaps, selection

GAP = gaps.all_gaps()[2]  # [10.04, 10.82], held-out test point 10.43


def _labelled(rows: list[dict]) -> pd.DataFrame:
    """A labelled augmentation table with one entry per supplied row."""
    boundaries = np.zeros((len(rows), features.FEATURE_DIM))
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            augment.SOURCE_ROW: [r["source"] for r in rows],
            augment.CONVERGED: [r.get("converged", True) for r in rows],
            selection.ASPECT_RATIO: [r["achieved"] for r in rows],
            **{name: np.zeros(len(rows)) for name in corpus_mod.CONDITION_COLUMNS},
        }
    )
    return frame


# Source 0 sits on the low anchor, 1 on the high anchor, 2 on the held-out point.
SOURCE_ASPECT_RATIO = np.array([GAP.low, GAP.high, 10.43])


def test_rows_from_both_anchors_are_kept() -> None:
    labelled = _labelled(
        [{"source": 0, "achieved": 10.1}, {"source": 1, "achieved": 10.8}]
    )

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert len(dataset) == 2


def test_a_held_out_source_is_excluded_however_it_was_rescaled() -> None:
    """This is the leak an aspect-ratio window cannot catch."""
    labelled = _labelled(
        [
            {"source": 0, "achieved": 10.43},  # anchor rescaled into the gap: keep
            {"source": 2, "achieved": GAP.low},  # held-out rescaled onto an anchor
        ]
    )

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert len(dataset) == 1
    assert dataset.frame[augment.SOURCE_ROW].tolist() == [0]


def test_an_anchor_rescaled_into_the_gap_is_kept_deliberately() -> None:
    """It needed no new campaign, and it is the baseline a model has to beat."""
    labelled = _labelled([{"source": 1, "achieved": 10.43}])

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert len(dataset) == 1
    assert bool(dataset.in_gap[0])


def test_rows_outside_the_gap_are_not_counted_as_in_gap() -> None:
    labelled = _labelled([{"source": 0, "achieved": GAP.low}])

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert not bool(dataset.in_gap[0])


def test_designs_the_simulator_failed_on_are_dropped() -> None:
    labelled = _labelled(
        [
            {"source": 0, "achieved": 10.2},
            {"source": 0, "achieved": 10.3, "converged": False},
        ]
    )

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert len(dataset) == 1


def test_the_leak_an_aspect_ratio_filter_would_admit_is_measurable() -> None:
    """Reporting the size of the leak beats asserting there is none."""
    labelled = _labelled(
        [
            {"source": 2, "achieved": GAP.low},  # would fool an AR filter
            {"source": 2, "achieved": 10.43},  # an AR filter rejects this too
            {"source": 0, "achieved": GAP.low},  # genuinely an anchor
        ]
    )

    leaks = selection.leaked(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert leaks.tolist() == [True, False, False]


def test_conditioning_includes_aspect_ratio() -> None:
    """Filling a gap means asking for a design at an aspect ratio.

    A decoder not told which one has no way to produce it.
    """
    assert selection.ASPECT_RATIO in selection.CONDITION_COLUMNS
    assert set(corpus_mod.CONDITION_COLUMNS) < set(selection.CONDITION_COLUMNS)


def test_conditions_come_back_in_the_requested_order() -> None:
    labelled = _labelled([{"source": 0, "achieved": 10.2}])

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)
    conditions = dataset.conditions()

    assert conditions.shape == (1, len(selection.CONDITION_COLUMNS))


def test_a_missing_conditioning_column_is_named() -> None:
    labelled = _labelled([{"source": 0, "achieved": 10.2}])
    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    with pytest.raises(KeyError, match="absent_metric"):
        dataset.conditions(("absent_metric",))


def test_boundaries_come_back_stacked() -> None:
    labelled = _labelled(
        [{"source": 0, "achieved": 10.2}, {"source": 1, "achieved": 10.7}]
    )

    dataset = selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)

    assert dataset.boundaries().shape == (2, features.FEATURE_DIM)


def test_a_table_missing_provenance_is_rejected() -> None:
    labelled = _labelled([{"source": 0, "achieved": 10.2}]).drop(
        columns=[augment.SOURCE_ROW]
    )

    with pytest.raises(KeyError, match=augment.SOURCE_ROW):
        selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)


def test_a_source_index_out_of_range_is_rejected() -> None:
    """Silently resolving the wrong provenance would defeat the whole filter."""
    labelled = _labelled([{"source": 99, "achieved": 10.2}])

    with pytest.raises(IndexError, match="only 3 source"):
        selection.select(labelled, SOURCE_ASPECT_RATIO, GAP)


def test_every_gap_selects_only_its_own_anchors() -> None:
    """A model for one gap must never see a neighbouring gap's campaigns."""
    anchors = np.array(gaps.ANCHORS)
    labelled = _labelled(
        [{"source": i, "achieved": float(ar)} for i, ar in enumerate(anchors)]
    )

    for gap in gaps.all_gaps():
        dataset = selection.select(labelled, anchors, gap)
        kept = anchors[dataset.frame[augment.SOURCE_ROW].to_numpy()]
        assert set(np.round(kept, 2)) <= {gap.low, gap.high}
