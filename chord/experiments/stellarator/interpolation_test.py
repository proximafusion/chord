"""Tests for the cross-basin interpolation sweep.

These run on a small synthetic corpus and a stub surrogate, so they check the wiring
rather than the result. The simulator validation needs the stellarator extra and is
skipped without it.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from ... import corpus as corpus_mod
from ... import evaluate as evaluate_mod
from ... import feasibility as feasibility_mod
from ... import features
from ... import surrogate as surrogate_mod
from .. import variants
from . import interpolation, training

N_ROWS = 96
METRIC_NAMES = corpus_mod.CONDITION_COLUMNS
KEPT_WIDTH = int((~features.structurally_zero_mask()).sum())


def _standardization() -> features.Standardization:
    kept = ~features.structurally_zero_mask()
    return features.Standardization(
        alpha=1.0, kept=kept, mean=np.zeros(KEPT_WIDTH), std=np.ones(KEPT_WIDTH)
    )


def _predictor() -> surrogate_mod.StandardizedPredictor:
    n = len(METRIC_NAMES)
    return surrogate_mod.StandardizedMetrics(
        surrogate_mod.SurrogateModel(KEPT_WIDTH, n, hidden_dims=[16, 16]),
        torch.zeros(n),
        torch.ones(n),
        METRIC_NAMES,
        _standardization(),
    )


class _AlwaysFeasible(surrogate_mod.StandardizedMetrics):
    """Reports a fixed metric vector comfortably inside every bound."""

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        inside = {
            feasibility_mod.IOTA_EDGE: 0.75 / feasibility_mod.N_FIELD_PERIODS,
            feasibility_mod.MIRROR_RATIO_EDGE: 0.2,
            feasibility_mod.VACUUM_WELL: 0.1,
        }
        row = [inside.get(name, 1.0) for name in self.metric_names]
        return designs.new_tensor([row]).expand(designs.shape[0], len(row))


def _corpus() -> corpus_mod.Corpus:
    rng = np.random.default_rng(0)
    boundaries = rng.normal(scale=1e-2, size=(N_ROWS, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            corpus_mod.SPLIT_COLUMN: np.array(
                ["train"] * (N_ROWS - 32) + ["val"] * 16 + ["test"] * 16
            ),
            corpus_mod.BASIN_COLUMN: np.where(
                np.arange(N_ROWS) % 2 == 0, "planar", "torsion"
            ),
            **{name: rng.normal(size=N_ROWS) for name in METRIC_NAMES},
        }
    )
    return corpus_mod.from_frame(frame)


@pytest.fixture(scope="module")
def corpus() -> corpus_mod.Corpus:
    return _corpus()


@pytest.fixture(scope="module")
def trained(corpus: corpus_mod.Corpus) -> training.Trained:
    config = dataclasses.replace(
        training.training_config(variants.VARIANTS["chord"]),
        n_epochs=2,
        chord_warmup_start_epoch=0,
        chord_warmup_epochs=1,
    )
    return training.train(
        corpus, variants.VARIANTS["chord"], _predictor(), config=config
    )


def test_pairs_come_from_different_basins(corpus: corpus_mod.Corpus) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=5, seed=0)

    assert len(rows_a) == len(rows_b) == 5
    assert set(corpus.basin[rows_a]).isdisjoint(set(corpus.basin[rows_b]))


def test_pairs_are_drawn_from_the_held_out_split(corpus: corpus_mod.Corpus) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=5, seed=0)

    assert set(corpus.split[np.concatenate([rows_a, rows_b])]) == {"test"}


def test_a_single_basin_split_is_rejected() -> None:
    corpus = _corpus()
    corpus.basin[:] = "planar"

    with pytest.raises(ValueError, match="nothing to interpolate"):
        interpolation.cross_basin_pairs(corpus, n_pairs=2)


def test_sweep_is_shaped_by_pairs_and_steps(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=4, seed=0)

    result = interpolation.sweep(
        trained, _predictor(), corpus, rows_a, rows_b, n_alpha=7
    )

    assert result.boundaries.shape == (4, 7, features.FEATURE_DIM)
    assert result.metrics.shape == (4, 7, len(METRIC_NAMES))
    assert result.feasible.shape == (4, 7)


def test_sweep_endpoints_reconstruct_the_endpoint_designs(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    """At alpha 0 and 1 the path must sit on the endpoints, not merely near them."""
    rows, _ = interpolation.cross_basin_pairs(corpus, n_pairs=3, seed=0)

    result = interpolation.sweep(
        trained,
        _predictor(),
        corpus,
        rows,
        rows,
        conditioning=interpolation.LINEAR,
        n_alpha=5,
    )

    np.testing.assert_allclose(
        result.boundaries[:, 0], result.boundaries[:, -1], atol=1e-5
    )


def test_decoded_designs_keep_the_structural_zeros(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    """Anything else is rejected the moment it reaches the simulator."""
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=3, seed=0)

    result = interpolation.sweep(
        trained, _predictor(), corpus, rows_a, rows_b, n_alpha=5
    )

    zeros = result.boundaries[..., features.structurally_zero_mask()]
    assert bool(np.all(zeros == 0.0))


def test_both_conditioning_strategies_run(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=3, seed=0)

    results = {
        name: interpolation.sweep(
            trained, _predictor(), corpus, rows_a, rows_b, conditioning=name, n_alpha=5
        )
        for name in interpolation.CONDITIONING
    }

    assert set(results) == set(interpolation.CONDITIONING)
    assert all(r.conditioning == name for name, r in results.items())


def test_self_consistent_conditioning_changes_the_path(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    """If the two agreed, the experiment would be comparing a method with itself."""
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=4, seed=0)

    linear = interpolation.sweep(
        trained,
        _predictor(),
        corpus,
        rows_a,
        rows_b,
        conditioning=interpolation.LINEAR,
        n_alpha=7,
        n_iterations=5,
    )
    settled = interpolation.sweep(
        trained,
        _predictor(),
        corpus,
        rows_a,
        rows_b,
        conditioning=interpolation.SELF_CONSISTENT,
        n_alpha=7,
        n_iterations=5,
    )

    interior = linear.interior_mask()
    assert not np.allclose(
        linear.boundaries[:, interior], settled.boundaries[:, interior]
    )


def test_unknown_conditioning_is_rejected(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=2, seed=0)

    with pytest.raises(ValueError, match="conditioning must be one of"):
        interpolation.sweep(
            trained, _predictor(), corpus, rows_a, rows_b, conditioning="guess"
        )


def test_interior_excludes_the_endpoints(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    """Endpoints are feasible by construction and must not enter the interior score."""
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=3, seed=0)
    result = interpolation.sweep(
        trained, _predictor(), corpus, rows_a, rows_b, n_alpha=21
    )

    feasible = np.zeros_like(result.feasible)
    feasible[:, 0] = True
    feasible[:, -1] = True
    forced = dataclasses.replace(result, feasible=feasible)

    assert forced.interior() == pytest.approx(0.0)
    assert forced.full_path() > 0.0


def test_a_feasible_surrogate_gives_a_feasible_sweep(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    """Pins the plumbing from surrogate output through to the reported fraction."""
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=3, seed=0)
    n = len(METRIC_NAMES)
    always = _AlwaysFeasible(
        surrogate_mod.SurrogateModel(KEPT_WIDTH, n, hidden_dims=[8]),
        torch.zeros(n),
        torch.ones(n),
        METRIC_NAMES,
        _standardization(),
    )

    result = interpolation.sweep(
        trained, always, corpus, rows_a, rows_b, n_alpha=5, n_iterations=2
    )

    assert result.interior() == pytest.approx(1.0)
    assert result.fully_feasible() == pytest.approx(1.0)


@pytest.mark.skipif(
    not evaluate_mod.constellaration_available(), reason="needs the stellarator extra"
)
def test_validation_covers_the_requested_subsample(
    corpus: corpus_mod.Corpus, trained: training.Trained
) -> None:
    rows_a, rows_b = interpolation.cross_basin_pairs(corpus, n_pairs=2, seed=0)
    result = interpolation.sweep(
        trained, _predictor(), corpus, rows_a, rows_b, n_alpha=5
    )

    checked = interpolation.validate(result, stride=2, max_pairs=1, n_workers=1)

    assert checked.converged.shape == (1, 3)
    assert checked.feasible.shape == checked.converged.shape


def test_two_sided_constraints_name_their_metric_once() -> None:
    """The validation path extracts metrics by name, so a repeat would double-count."""
    names = tuple(
        dict.fromkeys(
            spec.metric_name for spec in feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS
        )
    )

    assert len(names) == len(set(names))
    assert len(names) < len(feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS)
    # The deduplicated ordering still resolves every constraint.
    feasibility_mod.ConstraintSet(feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS, names)


@pytest.mark.parametrize("reverse", [False, True])
def test_midpoint_statistics_include_both_interval_boundaries(reverse: bool) -> None:
    """The two edge samples contribute 2/9, independently of chord orientation."""
    alphas = np.linspace(0.0, 1.0, 21)
    if reverse:
        alphas = 1.0 - alphas
    feasible = np.zeros((1, 21), dtype=bool)
    feasible[:, [6, 14]] = True
    sweep = interpolation.Sweep(
        alphas=alphas,
        boundaries=np.zeros((1, 21, 1)),
        metrics=np.zeros((1, 21, 1)),
        feasible=feasible,
        conditioning=interpolation.LINEAR,
    )
    validation = interpolation.Validation(
        alphas=alphas,
        converged=np.ones_like(feasible),
        feasible=feasible,
        benchmark_feasible=feasible,
    )

    assert sweep.interior_mask().sum() == 9
    assert sweep.interior() == pytest.approx(2 / 9)
    assert validation.interior() == pytest.approx(2 / 9)
