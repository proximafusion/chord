"""Tests for training on the stellarator corpus.

These run on a small synthetic corpus rather than the published one, so they check the
wiring without reaching the network. The property that matters most is that the
decoder's output space comes from the surrogate; a model fitted against its own
statistics would silently feed the surrogate inputs it was not trained on.
"""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
from chord import artifacts, models

from ... import corpus as corpus_mod
from ... import features
from ... import surrogate as surrogate_mod
from ... import train as train_mod
from .. import variants
from . import training

N_ROWS = 96
METRIC_NAMES = corpus_mod.CONDITION_COLUMNS


def _standardization() -> features.Standardization:
    kept = ~features.structurally_zero_mask()
    width = int(kept.sum())
    # Statistics deliberately unlike anything fitted on the corpus below, so a
    # model that refits its own would disagree with this visibly.
    return features.Standardization(
        alpha=1.0,
        kept=kept,
        mean=np.full(width, 7.0),
        std=np.full(width, 3.0),
    )


def _predictor() -> surrogate_mod.StandardizedPredictor:
    n = len(METRIC_NAMES)
    kept_width = int((~features.structurally_zero_mask()).sum())
    return surrogate_mod.StandardizedMetrics(
        surrogate_mod.SurrogateModel(kept_width, n, hidden_dims=[16, 16]),
        torch.zeros(n),
        torch.ones(n),
        METRIC_NAMES,
        _standardization(),
    )


def _corpus() -> corpus_mod.Corpus:
    rng = np.random.default_rng(0)
    boundaries = rng.normal(scale=1e-2, size=(N_ROWS, features.FEATURE_DIM))
    boundaries[:, features.structurally_zero_mask()] = 0.0
    split = np.array(["train"] * (N_ROWS - 32) + ["val"] * 16 + ["test"] * 16)
    frame = pd.DataFrame(
        {
            corpus_mod.BOUNDARY_COLUMN: list(boundaries),
            corpus_mod.SPLIT_COLUMN: split,
            corpus_mod.BASIN_COLUMN: np.where(
                np.arange(N_ROWS) % 2 == 0, "planar", "torsion"
            ),
            **{name: rng.normal(size=N_ROWS) for name in METRIC_NAMES},
        }
    )
    return corpus_mod.from_frame(frame)


def _short(name: str, n_epochs: int = 2) -> train_mod.TrainingConfig:
    """A config brief enough for a test, with the regularizer warmed up immediately."""
    return dataclasses.replace(
        training.training_config(variants.VARIANTS[name]),
        n_epochs=n_epochs,
        chord_warmup_start_epoch=0,
        chord_warmup_epochs=1,
    )


def _train(name: str, n_epochs: int = 2) -> training.Trained:
    return training.train(
        _corpus(), variants.VARIANTS[name], _predictor(), config=_short(name, n_epochs)
    )


def test_the_decoder_inherits_the_surrogates_feature_space() -> None:
    """The pairing is the whole point; refitting here would break it silently."""
    predictor = _predictor()

    trained = training.train(
        _corpus(), variants.VARIANTS["chord"], predictor, config=_short("chord")
    )

    np.testing.assert_array_equal(
        trained.standardization.mean, predictor.standardization.mean
    )
    np.testing.assert_array_equal(
        trained.standardization.std, predictor.standardization.std
    )


def test_a_refit_standardization_would_differ() -> None:
    """Guards the test above from passing because the two happen to coincide."""
    corpus = _corpus()
    refit = features.Standardization.fit(corpus.boundaries, alpha=1.0)

    assert not np.allclose(refit.mean, _standardization().mean)


def test_conditional_variants_consume_every_conditioning_column() -> None:
    trained = _train("cvae")

    assert trained.conditional
    assert trained.model.y_dim == len(corpus_mod.CONDITION_COLUMNS)
    assert trained.condition_columns == corpus_mod.CONDITION_COLUMNS


def test_unconditional_variants_have_no_conditioning() -> None:
    trained = _train("vae")

    assert not trained.conditional
    assert trained.model.y_dim == 0


def test_training_reports_one_record_per_epoch() -> None:
    trained = _train("cvae", n_epochs=3)

    assert len(trained.history) == 3
    assert all(np.isfinite(record["reconstruction"]) for record in trained.history)


def test_validation_split_is_scored() -> None:
    trained = _train("cvae")

    assert "val_reconstruction" in trained.history[-1]


def test_latent_width_is_the_paper_setting() -> None:
    trained = _train("vae")

    assert trained.model.latent_dim == training.LATENT_DIM


def test_encode_then_decode_returns_raw_coefficients() -> None:
    corpus = _corpus()
    trained = _train("cvae")
    conditions = corpus.conditions()

    z = trained.encode(corpus.boundaries, conditions)
    decoded = trained.decode(z, trained.standardized_conditions(conditions))

    assert decoded.shape == corpus.boundaries.shape
    assert bool(np.isfinite(decoded).all())


def test_decoded_designs_keep_the_structural_zeros() -> None:
    """Anything else would be rejected when it reaches the simulator."""
    corpus = _corpus()
    trained = _train("cvae")
    conditions = corpus.conditions()

    decoded = trained.decode(
        trained.encode(corpus.boundaries, conditions),
        trained.standardized_conditions(conditions),
    )

    assert bool(np.all(decoded[:, features.structurally_zero_mask()] == 0.0))


def test_the_regularizer_is_active_only_for_chord_variants() -> None:
    assert training.training_config(variants.VARIANTS["chord"]).chord_lambda > 0
    assert training.training_config(variants.VARIANTS["cvae"]).chord_lambda == 0


def test_conditioning_the_surrogate_cannot_predict_is_rejected() -> None:
    """Self-consistent conditioning later needs every column predicted."""
    with pytest.raises(KeyError, match="unpredicted"):
        training.train(
            _corpus(),
            variants.VARIANTS["cvae"],
            _predictor(),
            config=_short("cvae", 1),
            condition_columns=("unpredicted",),
        )


def test_too_few_epochs_for_the_regularizer_is_rejected() -> None:
    """Below the warmup start the chord variants are silently identical to the plain
    ones."""
    config = training.training_config(variants.VARIANTS["chord"])

    with pytest.raises(ValueError, match="does not start until epoch"):
        training.train(
            _corpus(),
            variants.VARIANTS["chord"],
            _predictor(),
            config=dataclasses.replace(
                config, n_epochs=config.chord_warmup_start_epoch
            ),
        )


def test_variants_without_the_regularizer_may_train_briefly() -> None:
    """The guard must not block the baselines, which have no warmup to wait for."""
    trained = training.train(
        _corpus(), variants.VARIANTS["cvae"], _predictor(), config=_short("cvae", 1)
    )

    assert len(trained.history) == 1


@pytest.mark.parametrize("conditional", [True, False])
def test_archived_model_preserves_boundary_and_condition_transforms(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, conditional: bool
) -> None:
    """Saved feature and condition transforms preserve the decoded boundary."""
    transform = _standardization()
    names = tuple(reversed(METRIC_NAMES))
    count = len(names)
    model = models.CVAE(
        x_dim=transform.n_features,
        y_dim=count if conditional else 0,
        latent_dim=2,
        hidden_dims=[8, 8],
    ).eval()
    path = "fixed_ar/reference.pt"
    folder = tmp_path / "fixed_ar"
    folder.mkdir()
    torch.save(model.state_dict(), tmp_path / path)
    (folder / "models.json").write_text(
        json.dumps(
            {
                path: {
                    "conditional": conditional,
                    "latent_dim": 2,
                    "hidden_dims": [8, 8],
                    "use_layer_norm": True,
                    "use_residual": True,
                }
            }
        )
    )
    mean = np.arange(count, dtype=np.float64)
    std = np.full(count, 2.0)
    np.savez(
        folder / "metadata.npz",
        feature_alpha=transform.alpha,
        feature_kept=transform.kept,
        feature_mean=transform.mean,
        feature_std=transform.std,
        condition_names=np.asarray(names),
        condition_mean=mean,
        condition_std=std,
    )
    monkeypatch.setenv(artifacts.ARTIFACT_BASE_ENV, str(tmp_path))
    loaded = training.load_archived_model(path)
    x = np.arange(3 * transform.n_features, dtype=np.float64).reshape(3, -1) / 8
    y = np.arange(3 * count, dtype=np.float64).reshape(3, -1) / 4
    boundaries = transform.decode(x)
    conditions = y * std + mean
    conditions.setflags(write=False)
    with torch.no_grad():
        expected_z = model.encode(
            torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        expected_x = model.decode(expected_z, torch.tensor(y, dtype=torch.float32))
    actual_z = loaded.encode(boundaries, conditions)
    actual_boundaries = loaded.decode(
        actual_z, loaded.standardized_conditions(conditions)
    )
    assert loaded.condition_columns == names
    assert not loaded.model.training
    torch.testing.assert_close(actual_z, expected_z, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        actual_boundaries, transform.decode(expected_x.numpy()), rtol=0.0, atol=0.0
    )
