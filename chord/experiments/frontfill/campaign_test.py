"""Tests for reading and scoring the campaign front."""

import numpy as np
import pandas as pd
import pytest

from . import campaign, pareto


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            campaign.ROLE: [
                campaign.ANCHOR,
                campaign.ANCHOR,
                campaign.HELD_OUT,
                campaign.EXCLUDED,
                campaign.ANCHOR,
            ],
            campaign.ASPECT_RATIO: [8.0, 12.0, 10.0, 7.0, 9.0],
            pareto.GRADIENT_SCALE_LENGTH: [2.0, 6.0, 4.0, 1.0, 3.0],
            campaign.FEASIBLE: [True, True, True, True, False],
        }
    )


def test_points_cover_every_role_by_default() -> None:
    ratio, _ = campaign.points(_frame())

    assert len(ratio) == 4


def test_points_filter_to_the_requested_roles() -> None:
    ratio, length = campaign.points(_frame(), [campaign.ANCHOR])

    assert sorted(ratio) == [8.0, 12.0]
    assert sorted(length) == [2.0, 6.0]


def test_infeasible_designs_are_never_scored() -> None:
    """A front built from constraint violations measures nothing."""
    ratio, _ = campaign.points(_frame(), [campaign.ANCHOR])

    assert 9.0 not in ratio


def test_scoring_reports_both_campaign_fronts_without_generated_designs() -> None:
    report = campaign.score(_frame())

    assert set(report) == {"anchors_only", "study_campaign", "full_campaign"}
    assert report["full_campaign"]["n_points"] == 4


def test_study_reference_excludes_campaigns_outside_the_study() -> None:
    """An excluded low-aspect-ratio point changes only the broader reference."""
    report = campaign.score(_frame(), np.array([10.0]), np.array([4.0]))

    assert report["anchors_only"]["hypervolume"] == 68.0
    assert report["study_campaign"]["hypervolume"] == 72.0
    assert report["full_campaign"]["hypervolume"] == 74.0
    assert report["recovered_study"]["fraction_of_available_hypervolume"] == 1.0
    assert report["recovered"]["fraction_of_available_hypervolume"] == pytest.approx(
        2.0 / 3.0
    )


def test_generated_front_count_uses_provenance_at_rounded_anchor_targets() -> None:
    """An anchor just inside a rounded gap must not become a generated design."""
    frame = _frame()
    frame.loc[0, campaign.ASPECT_RATIO] = 8.0 + 2.0**-24

    report = campaign.score(frame, np.array([10.0]), np.array([4.0]))

    assert report["anchors_plus_generated"]["n_non_dominated"] == 3
    assert report["anchors_plus_generated"]["n_generated_non_dominated"] == 1
    assert report["anchors_plus_generated"]["largest_adjacent_aspect_ratio_gap"] == 2.0


def test_front_coverage_ignores_dominated_candidates() -> None:
    report = campaign.score(
        _frame(), np.array([9.0, 10.0, 11.0]), np.array([1.0, 4.0, 3.0])
    )

    assert report["anchors_only"]["largest_adjacent_aspect_ratio_gap"] == 4.0
    assert report["anchors_plus_generated"]["largest_adjacent_aspect_ratio_gap"] == 2.0


def test_the_full_campaign_dominates_the_anchors() -> None:
    """Held-out campaigns can only add area, which is what bounds the recovery."""
    report = campaign.score(_frame())

    assert (
        report["full_campaign"]["hypervolume"] >= report["anchors_only"]["hypervolume"]
    )


def test_generated_designs_add_the_middle_front() -> None:
    report = campaign.score(_frame(), np.array([10.0]), np.array([4.0]))

    assert "anchors_plus_generated" in report
    assert report["anchors_plus_generated"]["n_points"] == 3


def test_recovering_every_non_anchor_design_scores_one() -> None:
    """Generating everything the anchors lack must recover all of what was available."""
    frame = _frame()
    missing_ratio, missing_length = campaign.points(
        frame, [campaign.HELD_OUT, campaign.EXCLUDED]
    )

    report = campaign.score(frame, missing_ratio, missing_length)

    assert report["recovered"]["fraction_of_available_hypervolume"] == pytest.approx(
        1.0
    )


def test_a_useless_generated_design_recovers_nothing() -> None:
    frame = _frame()

    report = campaign.score(frame, np.array([19.0]), np.array([-0.5]))

    assert report["recovered"]["fraction_of_available_hypervolume"] == pytest.approx(
        0.0
    )


def test_added_frontier_designs_are_counted() -> None:
    """Density is the result front-filling actually delivers, so it is reported."""
    frame = _frame()

    report = campaign.score(frame, np.array([10.0]), np.array([4.0]))

    assert report["recovered"]["added_frontier_designs"] >= 1


def test_a_table_missing_a_required_column_is_named(tmp_path) -> None:
    path = tmp_path / "front.parquet"
    _frame().drop(columns=[campaign.ROLE]).to_parquet(path)

    with pytest.raises(KeyError, match=campaign.ROLE):
        campaign.load(str(path))


def test_a_direct_path_bypasses_the_artifact_store(tmp_path) -> None:
    path = tmp_path / "front.parquet"
    _frame().to_parquet(path)

    assert len(campaign.load(str(path))) == 5
