"""Tests for the anchor and gap geometry."""

import pytest

from . import gaps


def test_every_consecutive_pair_of_anchors_is_a_gap() -> None:
    assert len(gaps.all_gaps()) == len(gaps.ANCHORS) - 1


def test_gaps_span_the_anchors_without_overlapping() -> None:
    bounds = [(gap.low, gap.high) for gap in gaps.all_gaps()]

    assert [low for low, _ in bounds] == list(gaps.ANCHORS[:-1])
    assert [high for _, high in bounds] == list(gaps.ANCHORS[1:])


def test_every_held_out_campaign_is_tested_exactly_once() -> None:
    """A point in two gaps would be predicted twice; one in none would be unused."""
    tested = [point for gap in gaps.all_gaps() for point in gap.test_points]

    assert sorted(tested) == sorted(gaps.HELD_OUT)
    assert len(tested) == len(set(tested))


def test_test_points_lie_inside_their_gap() -> None:
    for gap in gaps.all_gaps():
        for point in gap.test_points:
            assert gap.low < point < gap.high


def test_excluded_campaigns_are_never_tested() -> None:
    tested = {point for gap in gaps.all_gaps() for point in gap.test_points}

    assert tested.isdisjoint(gaps.EXCLUDED)


def test_the_near_duplicate_anchor_is_excluded() -> None:
    """It sits 0.04 from an anchor, closer than the matching tolerance can separate."""
    near = min(gaps.EXCLUDED, key=lambda ar: min(abs(ar - a) for a in gaps.ANCHORS))
    separation = min(abs(near - anchor) for anchor in gaps.ANCHORS)

    assert separation < gaps.MATCH_TOLERANCE * 4
    assert near in gaps.EXCLUDED


def test_matching_tolerance_cannot_merge_the_two_closest_campaigns() -> None:
    everything = sorted({*gaps.ANCHORS, *gaps.HELD_OUT, *gaps.EXCLUDED})
    closest = min(b - a for a, b in zip(everything, everything[1:]))

    assert gaps.MATCH_TOLERANCE < closest


def test_anchors_and_held_out_campaigns_are_disjoint() -> None:
    assert set(gaps.ANCHORS).isdisjoint(gaps.HELD_OUT)
    assert set(gaps.ANCHORS).isdisjoint(gaps.EXCLUDED)


def test_contains_excludes_the_anchors_themselves() -> None:
    gap = gaps.all_gaps()[0]

    assert not gap.contains(gap.low)
    assert not gap.contains(gap.high)
    assert gap.contains((gap.low + gap.high) / 2)


def test_width_is_the_span() -> None:
    gap = gaps.all_gaps()[0]

    assert gap.width == pytest.approx(gap.high - gap.low)


def test_gaps_are_lookupable_by_name() -> None:
    assert gaps.gap_by_name("g1") is not None
    assert [gap.name for gap in gaps.all_gaps()] == ["g1", "g2", "g3", "g4"]


def test_an_unknown_gap_lists_the_real_ones() -> None:
    with pytest.raises(ValueError, match="g1"):
        gaps.gap_by_name("g9")
