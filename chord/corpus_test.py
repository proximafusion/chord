"""Tests for corpus loading and artifact resolution."""

import numpy as np
import pandas as pd
import pytest

from . import artifacts, corpus, feasibility, features


def _frame(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            corpus.BOUNDARY_COLUMN: [
                rng.normal(size=features.FEATURE_DIM) for _ in range(n)
            ],
            corpus.SPLIT_COLUMN: ["train"] * (n - 2) + ["val", "test"],
            corpus.BASIN_COLUMN: ["planar"] * (n // 2) + ["torsion"] * (n - n // 2),
            corpus.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO: rng.normal(size=n),
            feasibility.IOTA_EDGE: rng.normal(size=n),
            corpus.LOG10_QI: rng.normal(size=n),
            feasibility.MIRROR_RATIO_EDGE: rng.normal(size=n),
            feasibility.VACUUM_WELL: rng.normal(size=n),
            corpus.MAX_ELONGATION: rng.normal(size=n),
        }
    )


def test_boundaries_are_stacked_to_the_expected_width() -> None:
    loaded = corpus.from_frame(_frame())

    assert loaded.boundaries.shape == (6, features.FEATURE_DIM)
    assert len(loaded) == 6


def test_metric_columns_exclude_structural_ones() -> None:
    loaded = corpus.from_frame(_frame())

    for column in (corpus.BOUNDARY_COLUMN, corpus.SPLIT_COLUMN, corpus.BASIN_COLUMN):
        assert column not in loaded.metrics.columns


def test_subset_selects_one_split() -> None:
    loaded = corpus.from_frame(_frame())

    train = loaded.subset("train")

    assert len(train) == 4
    assert set(train.split) == {"train"}
    assert len(train.metrics) == 4


def test_subset_keeps_rows_aligned() -> None:
    """Boundaries and metrics must stay row-aligned through a subset."""
    frame = _frame()
    frame.loc[5, feasibility.VACUUM_WELL] = 123.0
    loaded = corpus.from_frame(frame)

    test_split = loaded.subset("test")

    assert float(test_split.metrics[feasibility.VACUUM_WELL].iloc[0]) == 123.0
    np.testing.assert_allclose(
        test_split.boundaries[0], np.stack(frame[corpus.BOUNDARY_COLUMN].tolist())[5]
    )


def test_conditions_follow_the_requested_order() -> None:
    loaded = corpus.from_frame(_frame())

    conditions = loaded.conditions()

    assert conditions.shape == (6, len(corpus.CONDITION_COLUMNS))
    np.testing.assert_allclose(
        conditions[:, 1], loaded.metrics[feasibility.IOTA_EDGE].to_numpy()
    )


def test_missing_conditioning_column_is_named() -> None:
    frame = _frame().drop(columns=[corpus.LOG10_QI])
    loaded = corpus.from_frame(frame)

    with pytest.raises(KeyError, match=corpus.LOG10_QI):
        loaded.conditions()


def test_missing_structural_column_is_named() -> None:
    frame = _frame().drop(columns=[corpus.BASIN_COLUMN])

    with pytest.raises(KeyError, match=corpus.BASIN_COLUMN):
        corpus.from_frame(frame)


def test_wrong_coefficient_count_is_rejected() -> None:
    frame = _frame()
    frame[corpus.BOUNDARY_COLUMN] = pd.Series([np.zeros(10) for _ in range(len(frame))])

    with pytest.raises(ValueError, match="expected"):
        corpus.from_frame(frame)


def test_round_trip_through_parquet(tmp_path) -> None:
    path = tmp_path / "corpus.parquet"
    _frame().to_parquet(path)

    loaded = corpus.load_corpus(str(path))

    assert loaded.boundaries.shape == (6, features.FEATURE_DIM)


def test_artifact_base_is_overridable(monkeypatch) -> None:
    """One override point is what makes an anonymised mirror a config change."""
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, "file:///tmp/mirror/")

    assert (
        artifacts.artifact_uri("corpus.parquet") == "file:///tmp/mirror/corpus.parquet"
    )


def test_artifact_base_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(artifacts.ARTIFACT_BASE_ENV, raising=False)

    assert artifacts.artifact_uri("corpus.parquet").startswith(
        artifacts.DEFAULT_ARTIFACT_BASE
    )


def test_conditioning_vector_width_matches_the_paper() -> None:
    assert len(corpus.CONDITION_COLUMNS) == 6
