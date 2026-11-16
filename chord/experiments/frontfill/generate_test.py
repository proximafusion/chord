"""Tests for generating designs inside a gap.

The property the whole thing turns on is that a requested aspect ratio survives the
self-consistent iteration. Without the clamp it would be overwritten by whatever the
decoder happened to produce, and the gap would not get filled where it was asked.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from ... import corpus as corpus_mod
from ... import feasibility as feasibility_mod
from ... import features
from ... import surrogate as surrogate_mod
from .. import variants
from ..stellarator import training as training_mod
from . import gaps, generate, selection

GAP = gaps.all_gaps()[2]
N_ROWS = 96
KEPT = int((~features.structurally_zero_mask()).sum())


def _standardization() -> features.Standardization:
    kept = ~features.structurally_zero_mask()
    return features.Standardization(
        alpha=1.0, kept=kept, mean=np.zeros(KEPT), std=np.ones(KEPT)
    )


class _Echo(surrogate_mod.StandardizedMetrics):
    """Reports a fixed metric vector, so what the clamp does is visible."""

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        fixed = {
            feasibility_mod.IOTA_EDGE: 0.75 / feasibility_mod.N_FIELD_PERIODS,
            feasibility_mod.MIRROR_RATIO_EDGE: 0.2,
            feasibility_mod.VACUUM_WELL: 0.1,
            # Far outside the gap, so only a clamp can bring it back.
            generate.ASPECT_RATIO: 30.0,
        }
        row = [fixed.get(name, 1.0) for name in self.metric_names]
        return designs.new_tensor([row]).expand(designs.shape[0], len(row))


def _predictor(
    cls=surrogate_mod.StandardizedMetrics,
) -> surrogate_mod.StandardizedMetrics:
    n = len(selection.CONDITION_COLUMNS)
    return cls(
        surrogate_mod.SurrogateModel(KEPT, n, hidden_dims=[16, 16]),
        torch.zeros(n),
        torch.ones(n),
        selection.CONDITION_COLUMNS,
        _standardization(),
    )


def _corpus() -> corpus_mod.Corpus:
    rng = np.random.default_rng(0)
    boundaries = rng.normal(scale=1e-2, size=(N_ROWS, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            corpus_mod.SPLIT_COLUMN: ["train"] * (N_ROWS - 16) + ["val"] * 16,
            corpus_mod.BASIN_COLUMN: "at_anchor",
            **{name: rng.normal(size=N_ROWS) for name in selection.CONDITION_COLUMNS},
        }
    )
    return corpus_mod.from_frame(frame)


def _trained() -> training_mod.Trained:
    config = dataclasses.replace(
        training_mod.training_config(variants.VARIANTS["chord"]),
        n_epochs=2,
        chord_warmup_start_epoch=0,
        chord_warmup_epochs=1,
    )
    return training_mod.train(
        _corpus(),
        variants.VARIANTS["chord"],
        _predictor(),
        config=config,
        condition_columns=selection.CONDITION_COLUMNS,
    )


def test_targets_sit_strictly_inside_the_gap() -> None:
    """The endpoints are anchors where campaigns already ran."""
    spread = generate.targets(GAP, n_bins=10)

    assert len(spread) == 10
    assert spread.min() > GAP.low
    assert spread.max() < GAP.high


def test_targets_cover_the_gap_rather_than_clustering() -> None:
    spread = generate.targets(GAP, n_bins=10)

    assert spread.max() - spread.min() > 0.6 * GAP.width


def test_no_bins_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        generate.targets(GAP, n_bins=0)


def test_the_window_pins_the_aspect_ratio_from_both_sides() -> None:
    specs = generate.aspect_ratio_window(10.4, 0.02)

    assert len(specs) == 2
    assert {spec.sign for spec in specs} == {1.0, -1.0}
    assert all(spec.metric_name == generate.ASPECT_RATIO for spec in specs)


def test_the_window_admits_the_target_and_rejects_a_drift() -> None:
    names = (generate.ASPECT_RATIO,)
    constraints = feasibility_mod.ConstraintSet(
        generate.aspect_ratio_window(10.4, 0.02), names
    )

    at_target = constraints.is_feasible(torch.tensor([[10.4]]))
    drifted = constraints.is_feasible(torch.tensor([[10.9]]))

    assert bool(at_target[0])
    assert not bool(drifted[0])


def test_generation_produces_one_batch_per_bin() -> None:
    generated = generate.fill(
        _trained(),
        _predictor(),
        GAP,
        generate.GenerationConfig(n_bins=4, per_bin=5, n_iterations=2),
    )

    assert len(generated) == 20
    assert len(np.unique(generated.requested_aspect_ratio)) == 4


def test_the_requested_aspect_ratio_survives_the_iteration() -> None:
    """Without the clamp the decoder's own value would overwrite the request."""
    config = generate.GenerationConfig(n_bins=3, per_bin=4, n_iterations=5)

    generated = generate.fill(_trained(), _predictor(_Echo), GAP, config)

    # The predictor insists on 30.0; the clamp must hold the request instead.
    requested = generated.requested_aspect_ratio
    assert requested.min() > GAP.low
    assert requested.max() < GAP.high
    assert not np.allclose(generated.achieved_aspect_ratio(), requested)


def test_decoded_designs_keep_the_structural_zeros() -> None:
    generated = generate.fill(
        _trained(),
        _predictor(),
        GAP,
        generate.GenerationConfig(n_bins=2, per_bin=3, n_iterations=2),
    )

    zeros = generated.boundaries[:, features.structurally_zero_mask()]
    assert bool(np.all(zeros == 0.0))


def test_a_decoder_not_conditioned_on_aspect_ratio_is_rejected() -> None:
    """Asking for an aspect ratio a decoder cannot be told is a silent no-op."""
    trained = training_mod.train(
        _corpus(),
        variants.VARIANTS["cvae"],
        _predictor(),
        config=dataclasses.replace(
            training_mod.training_config(variants.VARIANTS["cvae"]), n_epochs=1
        ),
        condition_columns=tuple(
            c for c in selection.CONDITION_COLUMNS if c != generate.ASPECT_RATIO
        ),
    )

    with pytest.raises(KeyError, match=generate.ASPECT_RATIO):
        generate.fill(trained, _predictor(), GAP)


def test_generation_is_reproducible() -> None:
    config = generate.GenerationConfig(n_bins=2, per_bin=4, n_iterations=2)
    trained = _trained()
    predictor = _predictor()

    first = generate.fill(trained, predictor, GAP, config, seed=5)
    second = generate.fill(trained, predictor, GAP, config, seed=5)

    np.testing.assert_allclose(first.boundaries, second.boundaries)
