"""Tests for self-consistent conditioning."""

import numpy as np
import pytest
import torch

from . import feasibility, inference, models

CONDITION_COLUMNS = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
)
IN_RANGE = 0.75 / feasibility.N_FIELD_PERIODS


class _Predictor:
    """Returns a fixed metric vector, so the fixed point is known in advance."""

    metric_names: tuple[str, ...] = CONDITION_COLUMNS

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls = 0

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return designs.new_tensor([self.values]).expand(designs.shape[0], 3)


def _scaler() -> inference.ConditionScaler:
    return inference.ConditionScaler(mean=np.zeros(3), std=np.ones(3))


def _constraints() -> feasibility.ConstraintSet:
    return feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, CONDITION_COLUMNS
    )


def test_scaler_round_trips() -> None:
    raw = np.random.default_rng(0).normal(size=(64, 3)) * 5 + 2
    scaler = inference.ConditionScaler.fit(raw)
    tensor = torch.as_tensor(raw)

    torch.testing.assert_close(scaler.decode(scaler.encode(tensor)), tensor)


def test_scaler_standardizes() -> None:
    raw = np.random.default_rng(0).normal(size=(512, 3)) * 5 + 2
    scaler = inference.ConditionScaler.fit(raw)

    encoded = scaler.encode(torch.as_tensor(raw))

    torch.testing.assert_close(
        encoded.mean(0), torch.zeros(3, dtype=encoded.dtype), atol=1e-6, rtol=0
    )


def test_constant_condition_does_not_divide_by_zero() -> None:
    raw = np.ones((16, 3))

    encoded = inference.ConditionScaler.fit(raw).encode(torch.as_tensor(raw))

    assert bool(torch.isfinite(encoded).all())


def test_bounds_read_both_directions_of_a_constraint() -> None:
    bounds = inference.bounds_from_constraints(_constraints())

    lower, upper = bounds[feasibility.IOTA_EDGE]
    assert lower == pytest.approx(0.725 / feasibility.N_FIELD_PERIODS)
    assert upper == pytest.approx(0.775 / feasibility.N_FIELD_PERIODS)


def test_one_sided_constraints_leave_the_other_side_open() -> None:
    bounds = inference.bounds_from_constraints(_constraints())

    assert bounds[feasibility.MIRROR_RATIO_EDGE][0] is None
    assert bounds[feasibility.VACUUM_WELL][1] is None


def test_iteration_reaches_what_the_decoder_produces() -> None:
    """The fixed point is the predictor's own output, not the initial request."""
    model = models.CVAE(4, 3, 2, hidden_dims=[8, 8])
    predictor = _Predictor([IN_RANGE, 0.3, 0.1])

    y = inference.self_consistent_conditioning(
        model,
        torch.randn(5, 2),
        torch.full((5, 3), 99.0),
        predictor,
        _scaler(),
        CONDITION_COLUMNS,
        n_iterations=3,
    )

    torch.testing.assert_close(y, torch.tensor([[IN_RANGE, 0.3, 0.1]]).expand(5, 3))


def test_iteration_count_is_honoured() -> None:
    model = models.CVAE(4, 3, 2, hidden_dims=[8, 8])
    predictor = _Predictor([IN_RANGE, 0.3, 0.1])

    inference.self_consistent_conditioning(
        model,
        torch.randn(5, 2),
        torch.zeros(5, 3),
        predictor,
        _scaler(),
        CONDITION_COLUMNS,
        n_iterations=7,
    )

    assert predictor.calls == 7


def test_clamping_pulls_the_fixed_point_inside_the_feasible_set() -> None:
    """Without clamping the iteration settles on an infeasible request."""
    model = models.CVAE(4, 3, 2, hidden_dims=[8, 8])
    predictor = _Predictor([0.10, 0.9, -1.0])
    arguments = (
        model,
        torch.randn(5, 2),
        torch.zeros(5, 3),
        predictor,
        _scaler(),
        CONDITION_COLUMNS,
    )

    unclamped = inference.self_consistent_conditioning(*arguments, n_iterations=3)
    clamped = inference.self_consistent_conditioning(
        *arguments, constraints=_constraints(), n_iterations=3
    )

    constraints = _constraints()
    assert not bool(constraints.is_feasible(unclamped).any())
    assert bool(constraints.is_feasible(clamped).all())


def test_clamping_respects_a_margin() -> None:
    """A tighter constraint set must pull the fixed point further inside."""
    model = models.CVAE(4, 3, 2, hidden_dims=[8, 8])
    predictor = _Predictor([0.10, 0.9, -1.0])
    tight = feasibility.ConstraintSet(
        feasibility.constraints_with_margin(vacuum_well_margin=0.05),
        CONDITION_COLUMNS,
    )

    clamped = inference.self_consistent_conditioning(
        model,
        torch.randn(5, 2),
        torch.zeros(5, 3),
        predictor,
        _scaler(),
        CONDITION_COLUMNS,
        constraints=tight,
        n_iterations=2,
    )

    assert float(clamped[0, 2]) == pytest.approx(0.05)


def test_unpredicted_conditioning_column_is_named() -> None:
    model = models.CVAE(4, 3, 2, hidden_dims=[8, 8])
    predictor = _Predictor([IN_RANGE, 0.3, 0.1])

    with pytest.raises(KeyError, match="not_predicted"):
        inference.self_consistent_conditioning(
            model,
            torch.randn(5, 2),
            torch.zeros(5, 3),
            predictor,
            _scaler(),
            ("not_predicted", *CONDITION_COLUMNS[1:]),
        )


def test_linear_conditioning_interpolates_endpoints() -> None:
    y_a = torch.zeros(4, 3)
    y_b = torch.ones(4, 3)

    midpoint = inference.linear_conditioning(y_a, y_b, torch.full((4, 1), 0.5))

    torch.testing.assert_close(midpoint, torch.full((4, 3), 0.5))
