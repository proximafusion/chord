"""Tests for artifact resolution and the released surrogate's unit conversion."""

import numpy as np
import pytest
import torch

from . import artifacts, feasibility, features, surrogate

RELEASED_NAMES = tuple(
    c.released_name for c in artifacts.RELEASED_SURROGATE_CONVERSIONS
)
N_KEPT = int((~features.structurally_zero_mask()).sum())


def _standardization(x_dim: int = 4) -> features.Standardization:
    """A transform of the right shape; the conversion tests do not depend on values."""
    kept = np.zeros(features.FEATURE_DIM, dtype=bool)
    kept[:x_dim] = True
    return features.Standardization(
        alpha=1.0, kept=kept, mean=np.zeros(x_dim), std=np.ones(x_dim)
    )


class _ConstantMetrics(surrogate.StandardizedMetrics):
    """Emits a fixed metric vector, so a conversion's effect is checkable exactly."""

    def __init__(self, values: list[float]) -> None:
        n = len(RELEASED_NAMES)
        super().__init__(
            surrogate.SurrogateModel(4, n, hidden_dims=[8, 8]),
            torch.zeros(n),
            torch.ones(n),
            RELEASED_NAMES,
            _standardization(),
        )
        self.values = torch.tensor([values])

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        return self.values.expand(designs.shape[0], self.values.shape[1])


def _inner(values: list[float] | None = None) -> surrogate.StandardizedMetrics:
    n = len(RELEASED_NAMES)
    if values is None:
        return surrogate.StandardizedMetrics(
            surrogate.SurrogateModel(4, n, hidden_dims=[8, 8]),
            torch.zeros(n),
            torch.ones(n),
            RELEASED_NAMES,
            _standardization(),
        )
    return _ConstantMetrics(values)


def test_artifact_base_defaults_to_a_published_location() -> None:
    """A fresh checkout must read something a reader can actually reach.

    An internal bucket would give an external user a permission error, and it would
    carry the authors' names into the code besides.
    """
    base = artifacts.DEFAULT_ARTIFACT_BASE

    assert base.startswith(("hf://", "https://"))


def test_artifact_base_is_overridable(monkeypatch) -> None:
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, "hf://datasets/anon/mirror/")

    assert (
        artifacts.artifact_uri("surrogate.pt")
        == "hf://datasets/anon/mirror/surrogate.pt"
    )


def test_conversion_renames_to_this_packages_convention() -> None:
    converted = artifacts.ConstellarationUnits(_inner())

    assert feasibility.IOTA_EDGE in converted.metric_names
    assert "iota_edge" not in converted.metric_names


def test_rotational_transform_is_divided_by_field_periods() -> None:
    """The released model reports raw iota; this package reports it per field period."""
    values = [0.0] * len(RELEASED_NAMES)
    values[RELEASED_NAMES.index("iota_edge")] = 0.75
    converted = artifacts.ConstellarationUnits(_inner(values))

    predicted = converted(torch.randn(2, 4))
    column = converted.metric_names.index(feasibility.IOTA_EDGE)

    assert float(predicted[0, column]) == pytest.approx(
        0.75 / feasibility.N_FIELD_PERIODS
    )


def test_gradient_scale_length_is_multiplied_by_field_periods() -> None:
    values = [0.0] * len(RELEASED_NAMES)
    values[RELEASED_NAMES.index("min_normalized_magnetic_gradient_scale_length")] = 4.0
    converted = artifacts.ConstellarationUnits(_inner(values))

    predicted = converted(torch.randn(2, 4))
    column = converted.metric_names.index(
        "minimum_normalized_magnetic_gradient_scale_length"
    )

    assert float(predicted[0, column]) == pytest.approx(
        4.0 * feasibility.N_FIELD_PERIODS
    )


def test_unscaled_metrics_pass_through() -> None:
    values = [0.0] * len(RELEASED_NAMES)
    values[RELEASED_NAMES.index("vacuum_well")] = 0.3
    converted = artifacts.ConstellarationUnits(_inner(values))

    predicted = converted(torch.randn(2, 4))
    column = converted.metric_names.index(feasibility.VACUUM_WELL)

    assert float(predicted[0, column]) == pytest.approx(0.3)


def test_smoothed_curvature_variant_is_not_treated_as_flux_compression() -> None:
    """Only the bad-curvature profile is the benchmark's flux compression.

    The released model emits a smoothed variant under a nearly identical name that
    measures something else, so it must not appear in the conversion table.
    """
    released = {c.released_name for c in artifacts.RELEASED_SURROGATE_CONVERSIONS}

    assert "landreman_2025_bad_curvature_rho2_049" in released
    assert "landreman_2025_fine_tuned_smooth_rho2_049" not in released


def test_conversion_satisfies_the_inference_interface() -> None:
    converted = artifacts.ConstellarationUnits(_inner())

    assert isinstance(converted.metric_names, tuple)
    assert converted(torch.randn(3, 4)).shape == (3, len(converted.metric_names))


def test_surrogate_missing_an_expected_output_is_named() -> None:
    n = len(RELEASED_NAMES)
    model = surrogate.SurrogateModel(4, n, hidden_dims=[8, 8])
    wrong = surrogate.StandardizedMetrics(
        model,
        torch.zeros(n),
        torch.ones(n),
        ("unexpected",) + RELEASED_NAMES[1:],
        _standardization(),
    )

    with pytest.raises(KeyError, match=RELEASED_NAMES[0]):
        artifacts.ConstellarationUnits(wrong)


def test_checkpoint_missing_statistics_is_rejected(tmp_path, monkeypatch) -> None:
    """Weights without their statistics cannot be loaded as a usable predictor."""
    path = tmp_path / "surrogate.pt"
    torch.save({"state_dict": {}}, path)
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, str(tmp_path))

    with pytest.raises(KeyError, match="metric_names"):
        artifacts.load_surrogate("surrogate.pt")


def test_surrogate_round_trips_through_a_checkpoint(tmp_path, monkeypatch) -> None:
    n = len(RELEASED_NAMES)
    model = surrogate.SurrogateModel(4, n, hidden_dims=[8, 8])
    path = tmp_path / "surrogate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metric_names": list(RELEASED_NAMES),
            "metrics_mean": torch.zeros(n),
            "metrics_std": torch.ones(n),
            "feature_mean": np.zeros(4),
            "feature_std": np.ones(4),
            "feature_kept": _standardization().kept,
            "feature_alpha": 1.0,
            "x_dim": 4,
            "hidden_dims": [8, 8],
        },
        path,
    )
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, str(tmp_path))

    loaded = artifacts.load_surrogate("surrogate.pt")

    designs = torch.randn(2, 4)
    model.eval()
    assert loaded.metric_names == RELEASED_NAMES
    assert not any(p.requires_grad for p in loaded.model.parameters())
    torch.testing.assert_close(loaded(designs), model(designs))


def test_checkpoint_without_feature_statistics_is_rejected(
    tmp_path, monkeypatch
) -> None:
    """Metric statistics alone are not enough to encode a boundary for the weights."""
    n = len(RELEASED_NAMES)
    path = tmp_path / "surrogate.pt"
    torch.save(
        {
            "state_dict": {},
            "metric_names": list(RELEASED_NAMES),
            "metrics_mean": torch.zeros(n),
            "metrics_std": torch.ones(n),
        },
        path,
    )
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, str(tmp_path))

    with pytest.raises(KeyError, match="feature_mean"):
        artifacts.load_surrogate("surrogate.pt")


def test_a_surrogate_without_a_standardization_refuses_raw_boundaries() -> None:
    """Silently encoding against the wrong statistics is the failure to prevent."""
    n = len(RELEASED_NAMES)
    bare = surrogate.StandardizedMetrics(
        surrogate.SurrogateModel(4, n, hidden_dims=[8, 8]),
        torch.zeros(n),
        torch.ones(n),
        RELEASED_NAMES,
    )

    with pytest.raises(AttributeError, match="feature standardization"):
        bare.predict_boundaries(np.zeros((2, features.FEATURE_DIM)))


def test_released_conventions_reach_raw_boundaries(tmp_path, monkeypatch) -> None:
    """Predicting from boundaries must agree with encoding by hand then predicting."""
    n = len(RELEASED_NAMES)
    model = surrogate.SurrogateModel(N_KEPT, n, hidden_dims=[8, 8])
    standardization = features.Standardization(
        alpha=1.0,
        kept=~features.structurally_zero_mask(),
        mean=np.zeros(N_KEPT),
        std=np.ones(N_KEPT),
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metric_names": list(RELEASED_NAMES),
            "metrics_mean": torch.zeros(n),
            "metrics_std": torch.ones(n),
            "feature_mean": standardization.mean,
            "feature_std": standardization.std,
            "feature_kept": standardization.kept,
            "feature_alpha": standardization.alpha,
            "x_dim": N_KEPT,
            "hidden_dims": [8, 8],
        },
        tmp_path / "surrogate.pt",
    )
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, str(tmp_path))
    released = artifacts.released_surrogate()

    raw = np.random.default_rng(0).normal(size=(3, features.FEATURE_DIM)) * 1e-3
    encoded = torch.as_tensor(released.standardization.encode(raw), dtype=torch.float32)

    torch.testing.assert_close(released.predict_boundaries(raw), released(encoded))
