"""Tests for fitting a surrogate.

Front-filling trains one per gap, so this has to work from a corpus rather than only
from the released checkpoint. The property worth pinning hardest is that the fitted
predictor carries the feature transform its weights expect, since separating those is
silent and ruinous.
"""

import dataclasses

import numpy as np
import pytest
import torch

from . import features, surrogate

METRIC_NAMES = ("first", "second", "third")
N_ROWS = 256


def _data(seed: int = 0, n_rows: int = N_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Boundaries whose metrics are a smooth function of them, so fitting can work."""
    rng = np.random.default_rng(seed)
    boundaries = rng.normal(scale=1e-2, size=(n_rows, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    driver = boundaries[:, 1:4] * 100.0
    metrics = np.stack(
        [driver[:, 0], driver[:, 1] * 2.0 + 1.0, np.tanh(driver[:, 2])], axis=-1
    )
    return boundaries, metrics


def _config(**overrides) -> surrogate.SurrogateConfig:
    base = surrogate.SurrogateConfig(n_epochs=3, batch_size=64, hidden_dims=(32, 32))

    return dataclasses.replace(base, **overrides)


def test_fitting_returns_a_usable_predictor() -> None:
    boundaries, metrics = _data()

    predictor, history = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config())

    assert predictor.metric_names == METRIC_NAMES
    assert len(history) == 3
    assert predictor.predict_boundaries(boundaries[:8]).shape == (8, 3)


def test_the_predictor_carries_the_transform_its_weights_expect() -> None:
    """Separating a surrogate from its standardization is silent and ruinous."""
    boundaries, metrics = _data()

    predictor, _ = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config())

    assert predictor.standardization.n_features == predictor.model.x_dim


def test_the_transform_is_fitted_on_training_rows_only() -> None:
    """Fitting it on everything would leak the held-out rows into the inputs."""
    boundaries, metrics = _data()

    predictor, _ = surrogate.fit(
        boundaries, metrics, METRIC_NAMES, _config(validation_fraction=0.5)
    )
    everything = features.Standardization.fit(boundaries)

    assert not np.allclose(predictor.standardization.mean, everything.mean)


def test_fitting_reduces_the_loss() -> None:
    boundaries, metrics = _data()

    _, history = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config(n_epochs=40))

    assert history[-1]["train"] < history[0]["train"]


def test_a_held_out_loss_is_reported() -> None:
    boundaries, metrics = _data()

    _, history = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config())

    assert all("validation" in record for record in history)


def test_no_holdout_reports_no_validation() -> None:
    boundaries, metrics = _data()

    _, history = surrogate.fit(
        boundaries, metrics, METRIC_NAMES, _config(validation_fraction=0.0)
    )

    assert all("validation" not in record for record in history)


def test_missing_metrics_do_not_poison_the_others() -> None:
    """A design the simulator failed on for one metric still teaches the rest."""
    boundaries, metrics = _data()
    holed = metrics.copy()
    holed[::2, 0] = np.nan

    predictor, history = surrogate.fit(
        boundaries, holed, METRIC_NAMES, _config(n_epochs=20)
    )

    predicted = predictor.predict_boundaries(boundaries)
    assert bool(torch.isfinite(predicted).all())
    assert history[-1]["train"] < history[0]["train"]


def test_a_metric_that_never_varies_does_not_divide_by_zero() -> None:
    boundaries, metrics = _data()
    metrics[:, 2] = 4.0

    predictor, _ = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config())

    assert bool(torch.isfinite(predictor.predict_boundaries(boundaries[:4])).all())


def test_the_fitted_predictor_is_frozen() -> None:
    """It shapes a decoder during training, so it must not learn from that."""
    boundaries, metrics = _data()

    predictor, _ = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config())

    assert not predictor.model.training
    assert all(not p.requires_grad for p in predictor.model.parameters())


def test_mismatched_row_counts_are_rejected() -> None:
    boundaries, metrics = _data()

    with pytest.raises(ValueError, match="against"):
        surrogate.fit(boundaries, metrics[:10], METRIC_NAMES, _config())


def test_mismatched_metric_names_are_rejected() -> None:
    boundaries, metrics = _data()

    with pytest.raises(ValueError, match="metric columns"):
        surrogate.fit(boundaries, metrics, ("only", "two"), _config())


def test_fitting_is_reproducible() -> None:
    boundaries, metrics = _data()

    first, _ = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config(), seed=3)
    second, _ = surrogate.fit(boundaries, metrics, METRIC_NAMES, _config(), seed=3)

    torch.testing.assert_close(
        first.predict_boundaries(boundaries[:16]),
        second.predict_boundaries(boundaries[:16]),
    )
