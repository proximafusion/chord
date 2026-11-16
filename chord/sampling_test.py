"""Tests for sampling-based search."""

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch

from . import artifacts, feasibility, inference, models, sampling

METRIC_NAMES = (
    feasibility.IOTA_EDGE,
    feasibility.MIRROR_RATIO_EDGE,
    feasibility.VACUUM_WELL,
    artifacts.LOG10_QI,
)
IN_RANGE = 0.75 / feasibility.N_FIELD_PERIODS
LATENT_DIM = 4


class _TransparentModel(models.CVAE):
    """Decodes the latent code unchanged, so the designs under test are known."""

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return z


class _Predictor:
    """Reads metrics off the design, so feasibility and ranking are controllable.

    The third column drives the vacuum well and so decides feasibility; the fourth
    drives the objective and so decides the ranking.
    """

    metric_names: tuple[str, ...] = METRIC_NAMES

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        metrics = designs.new_empty(designs.shape[0], 4)
        metrics[:, 0] = IN_RANGE
        metrics[:, 1] = 0.2
        metrics[:, 2] = designs[:, 2]
        metrics[:, 3] = designs[:, 3]
        return metrics


class _AlwaysInfeasible(_Predictor):
    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        metrics = super().__call__(designs)
        metrics[:, 2] = -1.0
        return metrics


def _model() -> _TransparentModel:
    return _TransparentModel(LATENT_DIM, len(METRIC_NAMES), LATENT_DIM, hidden_dims=[8])


def _scaler() -> inference.ConditionScaler:
    return inference.ConditionScaler(
        mean=np.array([0.25, 0.2, 0.05, -2.0]), std=np.array([0.01, 0.1, 0.02, 0.5])
    )


def _constraints() -> feasibility.ConstraintSet:
    return feasibility.ConstraintSet(
        feasibility.BROAD_SCAN_LOOSE_CONSTRAINTS, METRIC_NAMES
    )


def _config(**overrides: Any) -> sampling.SamplingConfig:
    base = sampling.SamplingConfig(n_samples=64, n_iterations=2, top_k=8)
    return dataclasses.replace(base, **overrides)


def _generator(seed: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _propose(
    predictor: _Predictor | None = None, **overrides: Any
) -> sampling.Proposal:
    return sampling.propose(
        _model(),
        predictor or _Predictor(),
        _scaler(),
        METRIC_NAMES,
        _constraints(),
        config=_config(**overrides),
        generator=_generator(),
    )


def test_initialization_requests_the_objective_across_its_range() -> None:
    config = _config(n_samples=256)

    y = sampling.initial_conditioning(_scaler(), METRIC_NAMES, config, _generator())

    requested = _scaler().decode(y)[:, METRIC_NAMES.index(artifacts.LOG10_QI)]
    assert float(requested.min()) >= config.objective_low
    assert float(requested.max()) <= config.objective_high
    assert float(requested.max() - requested.min()) > 1.0


def test_initialization_leaves_other_components_at_the_corpus_mean() -> None:
    scaler = _scaler()

    y = sampling.initial_conditioning(scaler, METRIC_NAMES, _config(), _generator())

    other = scaler.decode(y)[:, :3]
    expected = torch.as_tensor(scaler.mean[:3], dtype=other.dtype)
    torch.testing.assert_close(other, expected.expand_as(other))


def test_shortlist_is_ordered_by_the_objective() -> None:
    proposal = _propose()

    objective = proposal.selected_metrics()[:, METRIC_NAMES.index(artifacts.LOG10_QI)]
    assert bool((objective.diff() >= 0).all())


def test_shortlist_is_drawn_only_from_feasible_candidates() -> None:
    proposal = _propose()

    assert bool(proposal.feasible.any())
    assert not bool(proposal.feasible.all())
    assert bool(proposal.feasible[proposal.ranking].all())


def test_shortlist_beats_every_feasible_candidate_it_left_out() -> None:
    proposal = _propose()

    objective = proposal.metrics[:, METRIC_NAMES.index(artifacts.LOG10_QI)]
    dropped = proposal.feasible.clone()
    dropped[proposal.ranking] = False
    assert float(objective[proposal.ranking].max()) <= float(objective[dropped].min())


def test_shortlist_is_capped_at_top_k() -> None:
    proposal = _propose(top_k=5)

    assert proposal.ranking.numel() == 5
    assert proposal.selected().shape == (5, LATENT_DIM)


def test_shortlist_shrinks_to_the_feasible_pool() -> None:
    """Asking for more than the surrogate approves returns only what it approved."""
    proposal = _propose(top_k=1000)

    assert proposal.ranking.numel() == int(proposal.feasible.sum())


def test_a_run_with_nothing_feasible_still_reports_candidates() -> None:
    """An empty shortlist would leave no account of what the run rejected."""
    proposal = _propose(_AlwaysInfeasible())

    assert not bool(proposal.feasible.any())
    assert proposal.ranking.numel() == 8


def test_separation_collapses_coincident_designs() -> None:
    proposal = _propose(min_separation=1e6)

    assert proposal.ranking.numel() == 1


def test_separation_keeps_the_best_of_a_cluster() -> None:
    unfiltered = _propose()
    filtered = _propose(min_separation=1e6)

    assert int(filtered.ranking[0]) == int(unfiltered.ranking[0])


def test_every_candidate_is_returned_not_only_the_shortlist() -> None:
    proposal = _propose(n_samples=64, top_k=8)

    assert proposal.designs.shape[0] == 64
    assert proposal.metrics.shape == (64, len(METRIC_NAMES))


def test_a_seeded_run_is_reproducible() -> None:
    first = _propose()
    second = _propose()

    torch.testing.assert_close(first.designs, second.designs)
    torch.testing.assert_close(first.ranking, second.ranking)


def test_distinct_count_sees_through_duplicates() -> None:
    designs = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])

    assert sampling.distinct_count(designs, torch.arange(3), min_separation=0.1) == 2


def test_distinct_count_keeps_well_separated_designs() -> None:
    designs = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

    assert sampling.distinct_count(designs, torch.arange(3), min_separation=0.1) == 3


def test_an_unpredicted_objective_is_named() -> None:
    with pytest.raises(KeyError, match="not_predicted"):
        _propose(objective="not_predicted")


def test_an_objective_outside_the_conditioning_is_rejected() -> None:
    """The objective is requested through the conditioning, so it has to be there."""
    with pytest.raises(KeyError, match=artifacts.LOG10_QI):
        sampling.propose(
            _model(),
            _Predictor(),
            _scaler(),
            METRIC_NAMES[:3],
            _constraints(),
            config=_config(),
        )


def test_a_scaler_fitted_on_other_columns_is_rejected() -> None:
    """A mismatched scaler would silently standardize the wrong components."""
    with pytest.raises(ValueError, match="conditioning columns"):
        sampling.initial_conditioning(
            inference.ConditionScaler(mean=np.zeros(2), std=np.ones(2)),
            METRIC_NAMES,
            _config(),
        )
