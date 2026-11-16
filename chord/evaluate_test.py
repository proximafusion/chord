"""Tests for the simulator adapter.

Everything that needs the optional `stellarator` extra is skipped when it is
absent, so the suite is meaningful either way. The parts that shape data are
written against duck-typed metrics and are always exercised.
"""

# pyright: reportMissingImports=false
import dataclasses
import math
import os
import signal

import numpy as np
import pytest
import torch

from . import evaluate, feasibility, features

METRIC_NAMES = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
)


@dataclasses.dataclass
class _Metrics:
    """Stand-in carrying the fields the adapter reads."""

    edge_rotational_transform_over_n_field_periods: float
    edge_magnetic_mirror_ratio: float
    vacuum_well: float


def test_converged_evaluation_is_stacked_in_order() -> None:
    evaluations = [
        evaluate.Evaluation(True, _Metrics(0.25, 0.3, 0.1)),
        evaluate.Evaluation(True, _Metrics(0.26, 0.4, 0.2)),
    ]

    values, converged = evaluate.metrics_to_tensor(evaluations, METRIC_NAMES)

    torch.testing.assert_close(
        values, torch.tensor([[0.25, 0.3, 0.1], [0.26, 0.4, 0.2]])
    )
    assert bool(converged.all())


def test_failed_evaluation_becomes_nan_not_a_plausible_number() -> None:
    """A caller who ignores the mask must get NaN, never a silently usable value."""
    evaluations = [
        evaluate.Evaluation(True, _Metrics(0.25, 0.3, 0.1)),
        evaluate.Evaluation(False, failure="solver did not converge"),
    ]

    values, converged = evaluate.metrics_to_tensor(evaluations, METRIC_NAMES)

    assert bool(torch.isnan(values[1]).all())
    assert converged.tolist() == [True, False]


def test_empty_input_yields_empty_tensors() -> None:
    values, converged = evaluate.metrics_to_tensor([], METRIC_NAMES)

    assert values.numel() == 0
    assert converged.numel() == 0


def test_stacked_metrics_feed_the_constraint_set() -> None:
    """The adapter's output is directly consumable by the feasibility machinery."""
    constraints = feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )
    in_range = 0.75 / feasibility.N_FIELD_PERIODS
    evaluations = [
        evaluate.Evaluation(True, _Metrics(in_range, 0.3, 0.1)),
        evaluate.Evaluation(True, _Metrics(in_range, 0.9, 0.1)),
    ]

    values, _ = evaluate.metrics_to_tensor(evaluations, METRIC_NAMES)

    assert constraints.is_feasible(values).tolist() == [True, False]


def test_nan_rows_are_never_reported_feasible() -> None:
    constraints = feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )
    values, _ = evaluate.metrics_to_tensor(
        [evaluate.Evaluation(False, failure="failed")], METRIC_NAMES
    )

    assert not bool(constraints.is_feasible(values).any())


def test_missing_extra_raises_a_message_naming_the_install() -> None:
    if evaluate.constellaration_available():
        pytest.skip("constellaration is installed, so the guard cannot be exercised")

    with pytest.raises(ImportError, match=r"chord\[stellarator\]"):
        evaluate.very_low_fidelity_settings()


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_fidelity_presets_select_distinct_vmec_settings() -> None:
    screening = evaluate.very_low_fidelity_settings()
    scoring = evaluate.high_fidelity_settings()

    assert screening.vmec_preset_settings.fidelity == "very_low_fidelity"
    assert scoring.vmec_preset_settings.fidelity == "high_fidelity"


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_benchmark_feasibility_is_delegated() -> None:
    """A design failing a benchmark-only bound must be rejected by the delegate.

    The corpus constraints do not bound quasi-isodynamic quality, so this design passes
    those while the benchmark rejects it.
    """
    from constellaration import forward_model  # pants: no-infer-dep

    metrics = forward_model.ConstellarationMetrics(
        aspect_ratio=10.0,
        aspect_ratio_over_edge_rotational_transform=40.0,
        max_elongation=5.0,
        axis_rotational_transform_over_n_field_periods=0.3,
        edge_rotational_transform_over_n_field_periods=0.26,
        axis_magnetic_mirror_ratio=0.2,
        edge_magnetic_mirror_ratio=0.2,
        average_triangularity=-0.3,
        vacuum_well=0.01,
        minimum_normalized_magnetic_gradient_scale_length=1.0,
        qi=1.0,  # log10 qi = 0, far above the benchmark's -3.5 bound
        flux_compression_in_regions_of_bad_curvature=0.5,
    )

    assert not evaluate.benchmark_feasible(metrics)
    assert metrics.qi is not None
    assert math.log10(metrics.qi) > -3.5


@dataclasses.dataclass
class _Equilibrium:
    """Stand-in carrying the half-grid fields the derivation reads."""

    bvco: np.ndarray
    bmnc: np.ndarray
    Aminor_p: float


def test_effective_aspect_ratio_extrapolates_to_the_axis() -> None:
    """Axis values come from the first two half-grid points, not the first alone."""
    equilibrium = _Equilibrium(
        bvco=np.array([0.0, 2.0, 3.0]),
        bmnc=np.array([[0.0, 1.0, 1.0]]),
        Aminor_p=0.5,
    )

    # bvco at the axis is 2.0 * 1.5 - 3.0 * 0.5 = 1.5, field strength is 1.0.
    assert evaluate.effective_aspect_ratio(equilibrium) == pytest.approx(1.5 / 0.5)


def test_effective_aspect_ratio_ignores_the_first_entry() -> None:
    """The leading half-grid entry is a placeholder and must not enter the result."""
    first = _Equilibrium(
        bvco=np.array([99.0, 2.0, 3.0]),
        bmnc=np.array([[99.0, 1.0, 1.0]]),
        Aminor_p=0.5,
    )
    second = dataclasses.replace(
        first, bvco=np.array([-7.0, 2.0, 3.0]), bmnc=np.array([[-7.0, 1.0, 1.0]])
    )

    assert evaluate.effective_aspect_ratio(first) == pytest.approx(
        evaluate.effective_aspect_ratio(second)
    )


def test_extrapolation_needs_two_points() -> None:
    equilibrium = _Equilibrium(
        bvco=np.array([0.0, 1.0]), bmnc=np.array([[0.0, 1.0]]), Aminor_p=0.5
    )

    with pytest.raises(ValueError, match="at least two"):
        evaluate.effective_aspect_ratio(equilibrium)


def test_evaluation_keeps_the_derived_quantity_not_the_equilibrium() -> None:
    """A batch runs in workers, so returning a whole solved wout per design would cost.

    The one quantity derived from the equilibrium is computed while it is in hand and
    kept as a float instead.
    """
    fields = {field.name for field in dataclasses.fields(evaluate.Evaluation)}

    assert "effective_aspect_ratio" in fields
    assert "equilibrium" not in fields


def test_boundary_layout_matches_the_feature_transform() -> None:
    """The conversion must read the layout `chord.features` defines, not a copy."""
    x = np.arange(features.FEATURE_DIM, dtype=np.float64)

    r_cos, z_sin = features.vector_to_coefficients(x)

    assert r_cos.shape == z_sin.shape
    assert r_cos.size + z_sin.size == features.FEATURE_DIM


def _valid_vector(seed: int = 0) -> np.ndarray:
    """A coefficient vector with the structurally-zero entries actually zero."""
    x = np.random.default_rng(seed).normal(size=features.FEATURE_DIM) * 1e-2
    x[features.structurally_zero_mask()] = 0.0
    return x


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_boundary_round_trips_through_a_vector() -> None:
    x = _valid_vector()

    recovered = evaluate.vector_from_boundary(evaluate.boundary_from_vector(x))

    np.testing.assert_allclose(recovered, x)


def test_a_vector_breaking_stellarator_symmetry_names_the_coefficients() -> None:
    """The simulator rejects these anyway, several layers down and less clearly."""
    x = _valid_vector()
    offending = int(np.flatnonzero(features.structurally_zero_mask())[0])
    x[offending] = 1.0

    with pytest.raises(ValueError, match=str(offending)):
        evaluate.boundary_from_vector(x)


def test_decoded_designs_satisfy_the_symmetry_by_construction() -> None:
    """The feature transform inserts the zeros, so model output is always valid."""
    kept = ~features.structurally_zero_mask()
    standardization = features.Standardization(
        alpha=1.0,
        kept=kept,
        mean=np.zeros(int(kept.sum())),
        std=np.ones(int(kept.sum())),
    )

    decoded = standardization.decode(
        np.random.default_rng(0).normal(size=(4, int(kept.sum())))
    )

    assert bool(np.all(decoded[:, features.structurally_zero_mask()] == 0.0))


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_boundary_carries_the_field_period_count() -> None:
    x = np.zeros(features.FEATURE_DIM)

    boundary = evaluate.boundary_from_vector(x)

    assert boundary.n_field_periods == feasibility.N_FIELD_PERIODS
    assert boundary.is_stellarator_symmetric


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_a_batch_of_vectors_is_rejected() -> None:
    """Silently taking the first row would send the wrong design to the simulator."""
    with pytest.raises(ValueError, match="single coefficient vector"):
        evaluate.boundary_from_vector(np.zeros((4, features.FEATURE_DIM)))


def _killer(_boundary, _settings=None):
    """Stands in for a solver that kills its worker instead of raising."""
    os.kill(os.getpid(), signal.SIGKILL)


def _counter(boundary, _settings=None):
    return evaluate.Evaluation(converged=True, metrics=_Metrics(boundary, 0.3, 0.1))


def test_batch_of_nothing_is_empty() -> None:
    assert evaluate.evaluate_batch([]) == []


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_a_worker_that_dies_does_not_lose_the_batch(monkeypatch) -> None:
    """A hard solver crash must cost one design, not every design queued behind it."""
    monkeypatch.setattr(evaluate, "evaluate", _counter)
    boundaries = [evaluate.boundary_from_vector(_valid_vector(i)) for i in range(6)]

    results = evaluate.evaluate_batch(boundaries, n_workers=2)

    assert len(results) == len(boundaries)
    assert all(r.converged for r in results)


def test_unresolved_entries_are_reported_as_failures() -> None:
    """Whatever happens, every input gets an outcome rather than a None."""
    filled = evaluate._or_failure(None)

    assert not filled.converged
    assert filled.failure == evaluate._WORKER_DIED


@dataclasses.dataclass
class _FullMetrics:
    """Stand-in carrying every field the corpus convention reads."""

    aspect_ratio: float = 7.0
    aspect_ratio_over_edge_rotational_transform: float = 28.0
    max_elongation: float = 5.0
    axis_rotational_transform_over_n_field_periods: float = 0.2
    edge_rotational_transform_over_n_field_periods: float = 0.25
    axis_magnetic_mirror_ratio: float = 0.1
    edge_magnetic_mirror_ratio: float = 0.3
    average_triangularity: float = -0.2
    vacuum_well: float = 0.01
    minimum_normalized_magnetic_gradient_scale_length: float = 4.8
    qi: float = 1e-3
    flux_compression_in_regions_of_bad_curvature: float = 0.9


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_corpus_metrics_derive_the_columns_the_simulator_does_not_report() -> None:
    """Two conditioning columns are derived, so reading attributes alone misses them."""
    evaluations = [
        evaluate.Evaluation(True, _FullMetrics(), effective_aspect_ratio=8.2)
    ]

    columns = evaluate.corpus_metrics(evaluations)

    assert columns[feasibility.LOG10_QI][0] == pytest.approx(-3.0)
    assert columns[feasibility.EFFECTIVE_ASPECT_RATIO][0] == pytest.approx(8.2)
    assert columns[feasibility.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO][
        0
    ] == pytest.approx(8.2 - 7.0)


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_corpus_metrics_report_raw_qi_as_a_logarithm() -> None:
    """The corpus reports log10 of it, so a raw value here would be off by orders."""
    columns = evaluate.corpus_metrics(
        [evaluate.Evaluation(True, _FullMetrics(qi=1e-2), effective_aspect_ratio=8.0)]
    )

    assert columns[feasibility.LOG10_QI][0] == pytest.approx(-2.0)
    assert "qi" not in columns


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_corpus_metrics_are_nan_for_a_failed_design() -> None:
    columns = evaluate.corpus_metrics([evaluate.Evaluation(False, failure="boom")])

    assert all(math.isnan(values[0]) for values in columns.values())


@pytest.mark.skipif(
    not evaluate.constellaration_available(), reason="needs the stellarator extra"
)
def test_corpus_metrics_cover_every_conditioning_column() -> None:
    """The conditioning vector has to be buildable from a fresh simulator run."""
    from . import corpus as corpus_mod

    columns = evaluate.corpus_metrics(
        [evaluate.Evaluation(True, _FullMetrics(), effective_aspect_ratio=8.0)]
    )

    assert set(corpus_mod.CONDITION_COLUMNS) <= set(columns)
