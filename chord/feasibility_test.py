"""Tests for constraint handling and the chord regularizer."""

import pytest
import torch

from . import feasibility, models

METRIC_NAMES = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
)
IOTA_IN_RANGE = 0.75 / feasibility.N_FIELD_PERIODS


def _constraints() -> feasibility.ConstraintSet:
    return feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )


def _always_feasible(designs: torch.Tensor) -> torch.Tensor:
    """Metrics comfortably inside every bound."""
    return designs.new_tensor([[IOTA_IN_RANGE, 0.3, 0.1]]).expand(designs.shape[0], 3)


def _always_infeasible(designs: torch.Tensor) -> torch.Tensor:
    """Metrics outside every bound, whatever the design."""
    return designs.new_tensor([[0.10, 0.9, -1.0]]).expand(designs.shape[0], 3)


def _infeasible_and_differentiable(designs: torch.Tensor) -> torch.Tensor:
    """Infeasible, and with a gradient path back to the design it was given."""
    return designs.sum(dim=-1, keepdim=True).expand(designs.shape[0], 3)


def _zeros(designs: torch.Tensor) -> torch.Tensor:
    return designs.new_zeros(designs.shape[0], 3)


def test_violations_sign_convention() -> None:
    """`h_k` is positive exactly where a constraint is violated."""
    constraints = _constraints()
    feasible = torch.tensor([[IOTA_IN_RANGE, 0.3, 0.1]])
    iota_too_low = torch.tensor([[0.70 / feasibility.N_FIELD_PERIODS, 0.3, 0.1]])

    assert (constraints.violations(feasible) <= 0).all()
    assert constraints.violations(iota_too_low)[0, 0] > 0


def test_is_feasible_matches_each_bound() -> None:
    constraints = _constraints()
    nfp = feasibility.N_FIELD_PERIODS
    batch = torch.tensor(
        [
            [0.75 / nfp, 0.30, 0.10],  # feasible
            [0.70 / nfp, 0.30, 0.10],  # iota below lower bound
            [0.80 / nfp, 0.30, 0.10],  # iota above upper bound
            [0.75 / nfp, 0.50, 0.10],  # mirror ratio too large
            [0.75 / nfp, 0.30, -0.1],  # negative vacuum well
        ]
    )

    expected = torch.tensor([True, False, False, False, False])
    torch.testing.assert_close(constraints.is_feasible(batch), expected)


def test_margin_shrinks_the_feasible_set() -> None:
    """A design on the bound is feasible without a margin and infeasible with one."""
    on_the_bound = torch.tensor([[0.725 / feasibility.N_FIELD_PERIODS, 0.4, 0.0]])

    loose = feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )
    tight = feasibility.ConstraintSet(
        feasibility.constraints_with_margin(
            iota_margin=0.01, mirror_margin=0.01, vacuum_well_margin=0.01
        ),
        METRIC_NAMES,
    )

    assert bool(loose.is_feasible(on_the_bound))
    assert not bool(tight.is_feasible(on_the_bound))


def test_unknown_metric_is_rejected_at_construction() -> None:
    specs = (feasibility.ConstraintSpec("not_a_metric", 1.0, 0.0),)

    with pytest.raises(KeyError, match="not_a_metric"):
        feasibility.ConstraintSet(specs, METRIC_NAMES)


def test_penalty_is_zero_when_interior_is_feasible() -> None:
    model = models.CVAE(4, 2, 3, hidden_dims=[8, 8])
    constraints = _constraints()

    penalty = feasibility.chord_penalty(
        model, torch.randn(8, 4), torch.randn(8, 2), _always_feasible, constraints
    )

    torch.testing.assert_close(penalty, torch.zeros(()))


def test_penalty_is_positive_when_interior_is_infeasible() -> None:
    model = models.CVAE(4, 2, 3, hidden_dims=[8, 8])
    constraints = _constraints()

    penalty = feasibility.chord_penalty(
        model, torch.randn(8, 4), torch.randn(8, 2), _always_infeasible, constraints
    )

    assert penalty > 0


def test_penalty_trains_the_decoder_and_not_the_encoder() -> None:
    """The encoder is detached so the term cannot be satisfied by moving the codes."""
    model = models.CVAE(4, 2, 3, hidden_dims=[8, 8])
    constraints = _constraints()

    penalty = feasibility.chord_penalty(
        model,
        torch.randn(8, 4),
        torch.randn(8, 2),
        _infeasible_and_differentiable,
        constraints,
    )
    penalty.backward()

    assert model.decoder.net.net[0].weight.grad is not None
    assert model.encoder.fc_mu.weight.grad is None


def test_penalty_handles_unconditional_model() -> None:
    model = models.CVAE(4, y_dim=0, latent_dim=3, hidden_dims=[8, 8])
    constraints = _constraints()

    penalty = feasibility.chord_penalty(
        model, torch.randn(8, 4), torch.zeros(8, 0), _zeros, constraints
    )

    assert penalty.shape == ()


@pytest.mark.parametrize("batch_size,n_pairs", [(1, 16), (8, 0)])
def test_penalty_degenerates_to_zero(batch_size: int, n_pairs: int) -> None:
    """Too few designs to pair, or no pairs requested, contributes nothing."""
    model = models.CVAE(4, 2, 3, hidden_dims=[8, 8])
    constraints = _constraints()

    penalty = feasibility.chord_penalty(
        model,
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 2),
        _zeros,
        constraints,
        n_pairs=n_pairs,
    )

    torch.testing.assert_close(penalty, torch.zeros(()))


def test_iota_bound_is_per_field_period() -> None:
    """Raw edge iota of 0.75 is feasible; the same number read per field period is not.

    A name-only mapping between our metrics and constellaration's would swap these two
    quantities and produce plausible but wrong feasibility labels.
    """
    constraints = _constraints()
    nfp = feasibility.N_FIELD_PERIODS

    correctly_scaled = torch.tensor([[0.75 / nfp, 0.3, 0.1]])
    raw_iota_by_mistake = torch.tensor([[0.75, 0.3, 0.1]])

    assert bool(constraints.is_feasible(correctly_scaled))
    assert not bool(constraints.is_feasible(raw_iota_by_mistake))
