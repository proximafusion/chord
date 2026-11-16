"""Tests for shaping a gap's rows and for the entry point that fills it."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from ... import corpus as corpus_mod
from ... import features
from . import augment, campaign, gaps, run, selection

GAP = gaps.all_gaps()[2]
N_ROWS = 160


def _labelled(n: int = N_ROWS) -> pd.DataFrame:
    """A labelled table whose rows all descend from this gap's two anchors."""
    rng = np.random.default_rng(0)
    boundaries = rng.normal(scale=1e-2, size=(n, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    sources = rng.integers(0, 2, size=n)
    achieved = rng.uniform(GAP.low, GAP.high, size=n)
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            augment.SOURCE_ROW: sources,
            augment.CONVERGED: True,
            run._SOURCE_ASPECT_RATIO: np.where(sources == 0, GAP.low, GAP.high),
            selection.ASPECT_RATIO: achieved,
            **{
                name: rng.normal(size=n)
                for name in selection.SURROGATE_METRICS
                if name != selection.ASPECT_RATIO
            },
        }
    )
    return frame


def _dataset() -> selection.GapDataset:
    labelled = _labelled()
    source = np.array([GAP.low, GAP.high])
    return selection.select(labelled, source, GAP)


def test_shaping_assigns_every_row_to_a_split() -> None:
    shaped = selection.as_corpus(_dataset())

    assert set(np.unique(shaped.split)) <= {"train", "val"}
    assert len(shaped) == N_ROWS


def test_shaping_holds_some_rows_out() -> None:
    shaped = selection.as_corpus(_dataset(), validation_fraction=0.25)

    assert 0 < int((shaped.split == "val").sum()) < len(shaped)


def test_shaping_records_which_rows_landed_in_the_gap() -> None:
    """Nothing reads it, but it keeps visible where the signal comes from."""
    dataset = _dataset()

    shaped = selection.as_corpus(dataset)

    assert set(np.unique(shaped.basin)) <= {"in_gap", "at_anchor"}
    assert int((shaped.basin == "in_gap").sum()) == int(dataset.in_gap.sum())


def test_shaping_keeps_the_conditioning_columns() -> None:
    shaped = selection.as_corpus(_dataset())

    assert shaped.conditions(selection.CONDITION_COLUMNS).shape == (
        N_ROWS,
        len(selection.CONDITION_COLUMNS),
    )


def test_filling_runs_end_to_end(tmp_path, monkeypatch) -> None:
    """The documented way in, so it is worth one run of its own."""
    dataset_path = tmp_path / "gap.parquet"
    _labelled().to_parquet(dataset_path, index=False)
    results = tmp_path / "results.jsonl"

    # The campaign front is a published artifact; a test must not fetch it.
    frame = pd.DataFrame(
        {
            campaign.ROLE: [campaign.ANCHOR, campaign.ANCHOR, campaign.HELD_OUT],
            campaign.ASPECT_RATIO: [8.0, 12.0, 10.0],
            run.pareto.GRADIENT_SCALE_LENGTH: [2.0, 6.0, 4.0],
            campaign.FEASIBLE: [True, True, True],
        }
    )
    monkeypatch.setattr(campaign, "load", lambda *a, **k: frame)

    run.main(
        [
            "fill",
            "--gap",
            GAP.name,
            "--dataset",
            str(dataset_path),
            "--n-epochs",
            "2",
            "--surrogate-epochs",
            "2",
            "--n-bins",
            "2",
            "--per-bin",
            "3",
            "--variant",
            "cvae",
            "--results-path",
            str(results),
        ]
    )

    assert results.read_text().count("\n") == 1


def test_a_dataset_without_provenance_is_rejected(tmp_path) -> None:
    """Selecting on aspect ratio instead would silently admit held-out campaigns."""
    path = tmp_path / "gap.parquet"
    _labelled().drop(columns=[run._SOURCE_ASPECT_RATIO]).to_parquet(path, index=False)

    with pytest.raises(KeyError, match=run._SOURCE_ASPECT_RATIO):
        run.main(["fill", "--gap", GAP.name, "--dataset", str(path)])


def test_a_dataset_built_for_another_gap_is_rejected(tmp_path) -> None:
    path = tmp_path / "gap.parquet"
    _labelled().to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no rows survive"):
        run.main(["fill", "--gap", "g1", "--dataset", str(path)])


def test_build_selects_campaign_targets_before_simulating(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An achieved anchor value must not admit a held-out campaign's boundary."""
    frame = _labelled(4)
    frame[corpus_mod.SPLIT_COLUMN] = "train"
    frame[corpus_mod.BASIN_COLUMN] = "anchor"
    frame[run._SOURCE_ASPECT_RATIO] = [7.0, 10.43, GAP.low, GAP.high]
    frame[augment.ASPECT_RATIO] = [7.0, GAP.low, GAP.low + 0.04, GAP.high - 0.04]
    source = tmp_path / "anchors.parquet"
    output = tmp_path / "labelled.parquet"
    frame.to_parquet(source, index=False)
    monkeypatch.setattr(augment, "label", _record_simulated_sources)

    run.main(
        [
            "build",
            "--gap",
            GAP.name,
            "--source-corpus",
            str(source),
            "--out",
            str(output),
            "--per-source",
            "2",
        ]
    )

    labelled = pd.read_parquet(output)
    assert labelled[augment.SOURCE_ROW].tolist() == [2, 2, 3, 3]
    assert labelled[run._SOURCE_ASPECT_RATIO].tolist() == [GAP.low] * 2 + [GAP.high] * 2


def test_fixed_ar_source_stops_before_simulation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The released fixed-AR corpus cannot supply front-filling anchors."""
    frame = _labelled(4)
    frame[corpus_mod.SPLIT_COLUMN] = "train"
    frame[corpus_mod.BASIN_COLUMN] = "planar"
    frame[run._SOURCE_ASPECT_RATIO] = 7.0
    frame[augment.ASPECT_RATIO] = 7.0
    source = tmp_path / "fixed_ar.parquet"
    frame.to_parquet(source, index=False)
    monkeypatch.setattr(augment, "label", _simulation_must_not_start)

    with pytest.raises(ValueError, match="no simulations have been started"):
        run.main(
            [
                "build",
                "--gap",
                GAP.name,
                "--source-corpus",
                str(source),
                "--out",
                str(tmp_path / "unused.parquet"),
            ]
        )


def _record_simulated_sources(
    augmented: augment.Augmentation, n_workers: int | None = None
) -> pd.DataFrame:
    return pd.DataFrame({augment.SOURCE_ROW: augmented.source_row})


def _simulation_must_not_start(
    augmented: augment.Augmentation, n_workers: int | None = None
) -> pd.DataFrame:
    raise AssertionError("simulation started for an incompatible source corpus")
