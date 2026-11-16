"""Tests for the training loop, loss and annealing schedules."""

import pytest
import torch

from . import feasibility, models, train

METRIC_NAMES = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
)


def _model() -> models.CVAE:
    return models.CVAE(4, 2, 3, hidden_dims=[8, 8])


def _constraints() -> feasibility.ConstraintSet:
    return feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )


def _infeasible(designs: torch.Tensor) -> torch.Tensor:
    return designs.new_tensor([[0.10, 0.9, -1.0]]).expand(designs.shape[0], 3)


def test_elbo_splits_into_reconstruction_and_kl() -> None:
    x = torch.randn(6, 4)
    mu = torch.zeros(6, 3)
    logvar = torch.zeros(6, 3)

    total, reconstruction, kl = train.elbo(x, x, mu, logvar, kl_weight=1.0)

    torch.testing.assert_close(reconstruction, torch.zeros(()))
    torch.testing.assert_close(kl, torch.zeros(()))
    torch.testing.assert_close(total, torch.zeros(()))


def test_kl_is_positive_away_from_the_prior() -> None:
    x = torch.randn(6, 4)
    mu = torch.full((6, 3), 2.0)
    logvar = torch.zeros(6, 3)

    _, _, kl = train.elbo(x, x, mu, logvar)

    assert kl > 0


def test_free_bits_floors_the_kl() -> None:
    """At the prior the unfloored KL is zero, so the floor sets the value."""
    x = torch.randn(6, 4)
    mu = torch.zeros(6, 3)
    logvar = torch.zeros(6, 3)

    _, _, kl = train.elbo(x, x, mu, logvar, free_bits=0.1)

    torch.testing.assert_close(kl, torch.full((), 0.1))


@pytest.mark.parametrize("epoch,expected", [(0, 0.0), (25, 0.5), (50, 1.0), (99, 1.0)])
def test_linear_kl_schedule(epoch: int, expected: float) -> None:
    config = train.TrainingConfig(
        kl_anneal="linear", kl_anneal_epochs=50, kl_weight=1.0
    )

    assert train.kl_weight_at(epoch, config) == pytest.approx(expected)


def test_kl_schedule_none_is_constant() -> None:
    config = train.TrainingConfig(kl_anneal="none", kl_weight=0.7)

    assert {train.kl_weight_at(e, config) for e in range(10)} == {0.7}


def test_cyclical_kl_schedule_repeats_and_saturates() -> None:
    config = train.TrainingConfig(
        kl_anneal="cyclical", n_epochs=40, kl_n_cycles=4, kl_cycle_ratio=0.5
    )

    assert train.kl_weight_at(0, config) == pytest.approx(0.0)
    assert train.kl_weight_at(10, config) == pytest.approx(0.0)
    assert train.kl_weight_at(5, config) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "epoch,expected", [(0, 0.0), (9, 0.0), (10, 0.0), (15, 0.5), (20, 1.0), (40, 1.0)]
)
def test_chord_warmup_delays_then_ramps(epoch: int, expected: float) -> None:
    config = train.TrainingConfig(
        chord_lambda=1.0, chord_warmup_start_epoch=10, chord_warmup_epochs=10
    )

    assert train.chord_lambda_at(epoch, config) == pytest.approx(expected)


def test_chord_lambda_zero_disables_the_term() -> None:
    config = train.TrainingConfig(chord_lambda=0.0)

    assert {train.chord_lambda_at(e, config) for e in range(50)} == {0.0}


def test_missing_predictor_with_positive_lambda_is_rejected() -> None:
    config = train.TrainingConfig(n_epochs=1, chord_lambda=1.0)

    with pytest.raises(ValueError, match="no metric predictor"):
        train.train(_model(), torch.randn(8, 4), torch.randn(8, 2), config)


def test_training_reduces_reconstruction_error() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(64, 4)
    y = torch.randn(64, 2)
    config = train.TrainingConfig(
        n_epochs=40, batch_size=16, kl_anneal="none", kl_weight=0.0, cosine_decay=False
    )

    history = train.train(model, x, y, config)

    assert history[-1]["reconstruction"] < history[0]["reconstruction"]


def test_history_records_one_entry_per_epoch() -> None:
    config = train.TrainingConfig(n_epochs=3, batch_size=8)

    history = train.train(_model(), torch.randn(16, 4), torch.randn(16, 2), config)

    assert [record["epoch"] for record in history] == [0.0, 1.0, 2.0]


def test_chord_term_is_recorded_once_warmed_up() -> None:
    config = train.TrainingConfig(
        n_epochs=2, batch_size=8, chord_lambda=1.0, chord_warmup_epochs=1
    )

    history = train.train(
        _model(),
        torch.randn(16, 4),
        torch.randn(16, 2),
        config,
        predict_metrics=_infeasible,
        constraints=_constraints(),
    )

    assert history[0]["chord_lambda"] == 0.0
    assert history[1]["chord_lambda"] == 1.0
    assert history[1]["chord"] > 0.0


def test_validation_error_is_reported_when_data_is_given() -> None:
    config = train.TrainingConfig(n_epochs=2, batch_size=8)

    history = train.train(
        _model(),
        torch.randn(16, 4),
        torch.randn(16, 2),
        config,
        x_val=torch.randn(8, 4),
        y_val=torch.randn(8, 2),
    )

    assert all("val_reconstruction" in record for record in history)
