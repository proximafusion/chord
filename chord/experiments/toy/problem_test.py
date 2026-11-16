"""Tests shared by every constructed benchmark.

Both problems are exercised through the same cases, so a new benchmark is wired in by
adding it to `PROBLEMS` rather than by writing another suite.
"""

import numpy as np
import pytest
import torch

from ... import inference
from . import cshape, problem, shells

PROBLEMS = [
    cshape.CShape(),
    cshape.CShape(n_dim=5),
    cshape.CShape(disc_radius=0.4),
    shells.Shells(),
    shells.Shells(inner_radius=0.0),
]
IDS = ["cshape", "cshape_5d", "cshape_with_disc", "shells", "shells_no_annulus"]


def _designs(instance: problem.Problem, n: int = 256, scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(0).normal(scale=scale, size=(n, instance.n_dim))


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_torch_mirror_matches_numpy(instance: problem.Problem) -> None:
    """The regularizer differentiates the mirror, so a drift between them is silent."""
    x = _designs(instance)

    mirrored = instance.violations_torch(torch.as_tensor(x)).numpy()

    np.testing.assert_allclose(mirrored, instance.violations(x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_mirror_is_differentiable(instance: problem.Problem) -> None:
    x = torch.as_tensor(_designs(instance, n=8)).requires_grad_(True)

    instance.violations_torch(x).sum().backward()

    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_one_violation_per_constraint_name(instance: problem.Problem) -> None:
    violations = instance.violations(_designs(instance, n=4))

    assert violations.shape == (4, len(instance.constraint_names))


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_seeds_are_feasible_and_land_in_the_requested_basin(
    instance: problem.Problem,
) -> None:
    rng = np.random.default_rng(0)

    for basin in range(instance.n_basins):
        seed = instance.seed(basin, rng)
        assert bool(problem.is_feasible(instance, seed))
        assert int(instance.basin(seed)) == basin


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_basins_are_within_range(instance: problem.Problem) -> None:
    labels = instance.basin(_designs(instance))

    assert labels.min() >= 0
    assert labels.max() < instance.n_basins


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_conditions_are_finite_and_batched(instance: problem.Problem) -> None:
    conditions = instance.conditions(_designs(instance, n=16))

    assert conditions.shape[0] == 16
    assert bool(np.isfinite(conditions).all())


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_sampled_objective_is_a_finite_scalar(instance: problem.Problem) -> None:
    rng = np.random.default_rng(0)
    objective = instance.sample_objective(0, rng)

    value = objective(instance.seed(0, rng))

    assert np.isfinite(value)


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_instances_satisfy_the_protocol(instance: problem.Problem) -> None:
    assert isinstance(instance, problem.Problem)


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_constraint_set_agrees_with_the_problem(instance: problem.Problem) -> None:
    """Feasibility read through the package must match the problem's own verdict."""
    x = _designs(instance)
    constraints = problem.constraint_set(instance)

    through_package = constraints.is_feasible(
        torch.as_tensor(instance.violations(x))
    ).numpy()

    np.testing.assert_array_equal(through_package, problem.is_feasible(instance, x))


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_standardized_violations_undo_the_transform(
    instance: problem.Problem,
) -> None:
    """The decoder emits standardized designs; the geometry only means anything raw."""
    x = _designs(instance)
    scaler = inference.ConditionScaler.fit(x)
    predictor = problem.StandardizedViolations(instance, scaler)

    predicted = predictor(scaler.encode(torch.as_tensor(x))).numpy()

    np.testing.assert_allclose(predicted, instance.violations(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("instance", PROBLEMS, ids=IDS)
def test_standardized_violations_report_the_constraint_order(
    instance: problem.Problem,
) -> None:
    scaler = inference.ConditionScaler.fit(_designs(instance))

    predictor = problem.StandardizedViolations(instance, scaler)

    assert predictor.metric_names == instance.constraint_names


def test_seed_rejects_a_basin_that_does_not_exist() -> None:
    with pytest.raises(ValueError, match="basin must be below"):
        cshape.CShape().seed(9, np.random.default_rng(0))


def test_cshape_default_matches_the_manuscript_annulus() -> None:
    """The origin and positive-axis wedge are forbidden; the far-side optimum is not."""
    instance = cshape.CShape()
    points = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.15, 0.0]])
    expected = np.array([False, False, True, True])

    np.testing.assert_array_equal(problem.is_feasible(instance, points), expected)
    np.testing.assert_array_equal(
        (instance.violations_torch(torch.as_tensor(points)) <= 0).all(dim=-1).numpy(),
        expected,
    )
    assert instance.n_basins == 2


def test_optional_disc_adds_a_disconnected_feasible_component() -> None:
    instance = cshape.CShape(disc_radius=0.4)
    origin = np.zeros(2)

    assert bool(problem.is_feasible(instance, origin))
    assert int(instance.basin(origin)) == cshape.DISC
    assert instance.n_basins == 3


def test_odd_slice_width_is_rejected() -> None:
    """The activation sums halves of a slice, which an odd width cannot provide."""
    with pytest.raises(ValueError, match="must be even"):
        shells.Shells(dims_per_mechanism=3)
