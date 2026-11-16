"""Tests for the surrogate model, its loss and the metric-predictor adapter."""

import pytest
import torch

from . import feasibility, models, surrogate

X_DIM = 6
N_METRICS = 3
BATCH = 8
METRIC_NAMES = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
)


def _surrogate() -> surrogate.SurrogateModel:
    return surrogate.SurrogateModel(X_DIM, N_METRICS, hidden_dims=[16, 16])


def _predictor(mean: float = 0.0, std: float = 1.0) -> surrogate.StandardizedMetrics:
    return surrogate.StandardizedMetrics(
        _surrogate(),
        torch.full((N_METRICS,), mean),
        torch.full((N_METRICS,), std),
        METRIC_NAMES,
    )


def test_forward_returns_one_metric_per_name() -> None:
    assert _surrogate()(torch.randn(BATCH, X_DIM)).shape == (BATCH, N_METRICS)


def test_standardized_metrics_applies_denormalization() -> None:
    """Physical metrics must be the standardized output scaled and shifted back."""
    predictor = _predictor(mean=5.0, std=2.0)
    designs = torch.randn(BATCH, X_DIM)

    expected = predictor.model(designs) * 2.0 + 5.0

    torch.testing.assert_close(predictor(designs), expected)


def test_predictor_satisfies_the_regularizer_interface() -> None:
    """The adapter is accepted directly by the chord regularizer."""
    predictor = _predictor()
    constraints = feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )
    model = models.CVAE(X_DIM, 2, 3, hidden_dims=[8, 8])

    penalty = feasibility.chord_penalty(
        model, torch.randn(BATCH, X_DIM), torch.randn(BATCH, 2), predictor, constraints
    )

    assert penalty.shape == ()


def test_mismatched_statistics_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree in shape"):
        surrogate.StandardizedMetrics(
            _surrogate(),
            torch.zeros(N_METRICS),
            torch.ones(N_METRICS + 1),
            METRIC_NAMES,
        )


def test_metric_name_count_must_match() -> None:
    with pytest.raises(ValueError, match="metric names"):
        surrogate.StandardizedMetrics(
            _surrogate(), torch.zeros(N_METRICS), torch.ones(N_METRICS), ("only_one",)
        )


def test_freeze_stops_gradients_and_sets_eval() -> None:
    predictor = _predictor().freeze()

    assert not predictor.model.training
    assert all(not p.requires_grad for p in predictor.model.parameters())


def test_frozen_surrogate_passes_gradient_to_its_input() -> None:
    """Freezing the surrogate must not block the decoder's gradient path."""
    predictor = _predictor().freeze()
    designs = torch.randn(BATCH, X_DIM, requires_grad=True)

    predictor(designs).sum().backward()

    assert designs.grad is not None


def test_loss_ignores_masked_entries() -> None:
    """A fully masked target cannot affect the loss, however wrong it is."""
    wild_target = torch.full((BATCH, N_METRICS), 1e3)

    loss = surrogate.surrogate_loss(
        torch.zeros(BATCH, N_METRICS), wild_target, torch.zeros(BATCH, N_METRICS)
    )

    assert float(loss) == pytest.approx(0.0)


def test_loss_counts_unmasked_error() -> None:
    loss = surrogate.surrogate_loss(
        torch.zeros(BATCH, N_METRICS),
        torch.ones(BATCH, N_METRICS),
        torch.ones(BATCH, N_METRICS),
    )

    assert float(loss) == pytest.approx(1.0)


def test_loss_averages_over_unmasked_entries_only() -> None:
    """Halving the mask must not halve the error; the denominator follows it."""
    mask = torch.ones(BATCH, N_METRICS)
    mask[:, 1:] = 0.0

    loss = surrogate.surrogate_loss(
        torch.zeros(BATCH, N_METRICS), torch.ones(BATCH, N_METRICS), mask
    )

    assert float(loss) == pytest.approx(1.0)
