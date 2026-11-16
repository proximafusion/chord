"""Tests for dataset generation, training and the interpolation evaluation.

These run at a fraction of the paper's settings. They check that the harness is wired
correctly, not that the method works; the latter needs a full run and is what
`run.py` is for.
"""

import numpy as np
import pytest

from . import cshape, datasets, interpolation, run, shells, training

SMALL = datasets.GenerationConfig(n_runs_per_basin=2, population=6, n_generations=6)


@pytest.fixture(scope="module")
def problem() -> cshape.CShape:
    return cshape.CShape(disc_radius=0.4)


@pytest.fixture(scope="module")
def dataset(problem: cshape.CShape) -> datasets.Dataset:
    return datasets.generate(problem, SMALL, seed=0)


@pytest.fixture(scope="module")
def trained(problem: cshape.CShape, dataset: datasets.Dataset) -> training.Trained:
    return training.train(
        problem, dataset, training.VARIANTS["chord"], n_epochs=3, seed=0
    )


def test_generated_designs_are_all_feasible(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    """The pool is filtered, so an infeasible row means the filter looked at the wrong
    thing."""
    assert len(dataset) > 0
    assert bool(np.all(problem.violations(dataset.designs) <= 0))


def test_generation_reaches_every_basin(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    assert set(dataset.counts()) == set(range(problem.n_basins))


def test_generated_conditions_line_up_with_designs(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    np.testing.assert_allclose(
        dataset.conditions, problem.conditions(dataset.designs), rtol=1e-6
    )


def test_generation_is_reproducible(problem: cshape.CShape) -> None:
    first = datasets.generate(problem, SMALL, seed=3)
    second = datasets.generate(problem, SMALL, seed=3)

    np.testing.assert_array_equal(first.designs, second.designs)


def test_shells_generates_too() -> None:
    """The generator is problem-agnostic, which only one problem cannot show."""
    instance = shells.Shells()

    generated = datasets.generate(instance, SMALL, seed=0)

    assert len(generated) > 0
    assert bool(np.all(instance.violations(generated.designs) <= 0))


def test_training_holds_out_a_disjoint_split(
    dataset: datasets.Dataset, trained: training.Trained
) -> None:
    assert 0 < len(trained.validation) < len(dataset)
    assert len(set(trained.validation.tolist())) == len(trained.validation)


def test_conditional_variant_consumes_its_conditioning(
    problem: cshape.CShape, dataset: datasets.Dataset, trained: training.Trained
) -> None:
    assert trained.conditional
    assert trained.model.y_dim == dataset.conditions.shape[-1]


def test_unconditional_variant_has_no_conditioning(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    plain = training.train(
        problem, dataset, training.VARIANTS["vae"], n_epochs=2, seed=0
    )

    assert not plain.conditional
    assert plain.model.y_dim == 0


def test_the_regularizer_is_only_active_for_chord_variants() -> None:
    assert training.training_config(training.VARIANTS["chord"]).chord_lambda > 0
    assert training.training_config(training.VARIANTS["chord_unc"]).chord_lambda > 0
    assert training.training_config(training.VARIANTS["cvae"]).chord_lambda == 0
    assert training.training_config(training.VARIANTS["vae"]).chord_lambda == 0


def test_the_regularizer_contributes_once_it_has_warmed_up(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    """A chord term that stays at zero would make the variants identical."""
    config = training.training_config(training.VARIANTS["chord"])
    warm = training.train(
        problem,
        dataset,
        training.VARIANTS["chord"],
        n_epochs=config.chord_warmup_start_epoch + 5,
        seed=0,
    )

    assert warm.history[-1]["chord_lambda"] > 0


def test_encode_decode_round_trips_through_raw_designs(
    dataset: datasets.Dataset, trained: training.Trained
) -> None:
    """Standardization is applied on the way in and undone on the way out."""
    designs = dataset.designs[:16]
    conditions = dataset.conditions[:16]

    decoded = trained.decode(trained.encode(designs, conditions), conditions)

    assert decoded.shape == designs.shape
    assert bool(np.isfinite(decoded).all())


def test_chord_endpoints_reconstruct_their_designs(
    dataset: datasets.Dataset, trained: training.Trained, problem: cshape.CShape
) -> None:
    """At alpha 0 and 1 the path must sit on the endpoints, not merely near them."""
    rows = trained.validation[:4]
    chord = interpolation.evaluate(problem, trained, dataset, rows, rows, n_alpha=5)

    np.testing.assert_allclose(chord.designs[:, 0], chord.designs[:, -1], atol=1e-5)


def test_chord_has_one_row_per_pair_and_one_column_per_step(
    dataset: datasets.Dataset, trained: training.Trained, problem: cshape.CShape
) -> None:
    rows_a = trained.validation[:5]
    rows_b = trained.validation[5:10]

    chord = interpolation.evaluate(problem, trained, dataset, rows_a, rows_b, n_alpha=7)

    assert chord.designs.shape == (5, 7, problem.n_dim)
    assert chord.feasible.shape == (5, 7)


def test_middle_ignores_the_endpoints(
    dataset: datasets.Dataset, trained: training.Trained, problem: cshape.CShape
) -> None:
    """Endpoints are feasible by construction, so they must not enter the interior
    score."""
    rows_a = trained.validation[:4]
    rows_b = trained.validation[4:8]
    chord = interpolation.evaluate(
        problem, trained, dataset, rows_a, rows_b, n_alpha=21
    )

    forced = interpolation.Chord(
        alphas=chord.alphas,
        designs=chord.designs,
        feasible=np.zeros_like(chord.feasible),
    )
    forced.feasible[:, 0] = True
    forced.feasible[:, -1] = True

    assert forced.middle() == pytest.approx(0.0)
    assert forced.full_path() > 0.0


def test_unconditional_models_interpolate_too(
    problem: cshape.CShape, dataset: datasets.Dataset
) -> None:
    """A plain VAE has zero-width conditioning, which the chord still has to carry."""
    plain = training.train(
        problem, dataset, training.VARIANTS["vae"], n_epochs=2, seed=0
    )
    rows = plain.validation[:4]

    chord = interpolation.evaluate(problem, plain, dataset, rows, rows[::-1], n_alpha=5)

    assert chord.designs.shape == (4, 5, problem.n_dim)
    assert bool(np.isfinite(chord.designs).all())


def test_every_basin_pair_is_evaluated(
    dataset: datasets.Dataset, trained: training.Trained, problem: cshape.CShape
) -> None:
    results = interpolation.across_basins(
        problem, trained, dataset, n_pairs=3, n_alpha=5, seed=0
    )

    assert set(results) == {(0, 1), (0, 2), (1, 2)}


def test_summary_pools_the_pairs(
    dataset: datasets.Dataset, trained: training.Trained, problem: cshape.CShape
) -> None:
    results = interpolation.across_basins(
        problem, trained, dataset, n_pairs=3, n_alpha=5, seed=0
    )

    summary = interpolation.summarize(results)

    assert 0.0 <= summary["middle"] <= 1.0
    assert summary["n_pairs"] == sum(len(chord) for chord in results.values())


def test_summary_of_nothing_is_not_a_crash() -> None:
    summary = interpolation.summarize({})

    assert np.isnan(summary["middle"])
    assert summary["n_pairs"] == 0


def test_entry_point_runs_end_to_end(tmp_path, monkeypatch) -> None:
    """The CLI is the documented way in, so it is worth one run of its own."""
    monkeypatch.setattr(datasets, "GenerationConfig", lambda **kw: SMALL)
    results = tmp_path / "results.jsonl"

    run.main(
        [
            "--problem",
            "cshape",
            "--variants",
            "cvae",
            "--n-epochs",
            "2",
            "--n-pairs",
            "2",
            "--results-path",
            str(results),
        ]
    )

    assert results.read_text().count("\n") == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_middle_includes_both_interval_boundaries(reverse: bool) -> None:
    """The inclusive interval must not lose an edge to linspace rounding."""
    alphas = np.linspace(0.0, 1.0, 21)
    if reverse:
        alphas = 1.0 - alphas
    feasible = np.zeros((1, 21), dtype=bool)
    feasible[:, [6, 14]] = True
    chord = interpolation.Chord(
        alphas=alphas, designs=np.zeros((1, 21, 2)), feasible=feasible
    )

    assert chord.middle() == pytest.approx(2 / 9)
