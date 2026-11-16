"""Train a variant on the published stellarator corpus.

Unlike the constructed benchmarks, feasibility here cannot be evaluated analytically or
cheaply. The chord regularizer needs metrics for designs decoded during training, and a
VMEC call per decoded design would make training take years, so a surrogate stands in.

That makes one thing load-bearing. The surrogate reads standardized designs, and its
statistics are not interchangeable with any other set: it was fitted on a five-million
row superset of this corpus, whose per-column feature means differ from the corpus's by
more than the features themselves vary. Feeding it corpus-standardized designs leaves
it agreeing with the simulator on feasibility about a quarter of the time instead of
over ninety percent, with nothing raising an error.

So the decoder's output space is taken from the surrogate rather than fitted here. The
two share a feature transform by construction, and the pairing cannot be got wrong by
forgetting to match them.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch
from numpy.typing import NDArray

from ... import corpus as corpus_mod
from ... import feasibility as feasibility_mod
from ... import features, models
from ... import inference as inference_mod
from ... import surrogate as surrogate_mod
from ... import train as train_mod
from .. import variants as variants_mod

_LOG = logging.getLogger(__name__)

VARIANTS = variants_mod.VARIANTS

LATENT_DIM = 16
HIDDEN_DIMS = [512, 512, 512]
FREE_BITS = 0.1


def training_config(
    variant: variants_mod.Variant, n_epochs: int = 300
) -> train_mod.TrainingConfig:
    """Settings the stellarator results were produced with."""
    return train_mod.TrainingConfig(
        n_epochs=n_epochs,
        free_bits=FREE_BITS,
        chord_lambda=variant.chord_lambda,
        chord_warmup_start_epoch=30,
        chord_warmup_epochs=30,
    )


@dataclasses.dataclass(frozen=True)
class Trained:
    """A fitted model and the transforms needed to use it on raw boundaries."""

    model: models.CVAE
    standardization: features.Standardization
    """Shared with the surrogate, which is why it is carried rather than refitted."""

    scaler: inference_mod.ConditionScaler
    condition_columns: tuple[str, ...]
    history: list[dict[str, float]]

    @property
    def conditional(self) -> bool:
        return self.model.y_dim > 0

    def encode(self, boundaries: NDArray, conditions: NDArray) -> torch.Tensor:
        """Latent codes for raw Fourier coefficients."""
        with torch.no_grad():
            return self.model.encode(
                _as_tensor(self.standardization.encode(boundaries)),
                self.standardized_conditions(conditions),
            )

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> NDArray:
        """Raw Fourier coefficients for latent codes and standardized conditioning."""
        with torch.no_grad():
            return self.standardization.decode(self.model.decode(z, y).numpy())

    def standardized_conditions(self, conditions: NDArray) -> torch.Tensor:
        """Conditioning as the decoder takes it, zero-width when unconditional."""
        if not self.conditional:
            return torch.zeros(len(conditions), 0)
        return self.scaler.encode(_as_tensor(conditions))


def train(
    corpus: corpus_mod.Corpus,
    variant: variants_mod.Variant,
    predictor: surrogate_mod.StandardizedPredictor,
    config: train_mod.TrainingConfig | None = None,
    latent_dim: int = LATENT_DIM,
    condition_columns: tuple[str, ...] = corpus_mod.CONDITION_COLUMNS,
    seed: int = 0,
) -> Trained:
    """Fit one variant on the corpus, in the surrogate's feature space."""
    torch.manual_seed(seed)
    standardization = predictor.standardization
    constraints = feasibility_mod.ConstraintSet(
        feasibility_mod.BROAD_SCAN_LOOSE_CONSTRAINTS, predictor.metric_names
    )
    _check_conditioning_is_predicted(condition_columns, predictor)

    config = config or training_config(variant)
    if variant.chord_lambda > 0 and config.n_epochs <= config.chord_warmup_start_epoch:
        raise ValueError(
            f"the chord regularizer does not start until epoch "
            f"{config.chord_warmup_start_epoch}, so training for {config.n_epochs} "
            f"would produce a model identical to the same variant without it"
        )

    train_rows = corpus.split == "train"
    scaler = inference_mod.ConditionScaler.fit(
        corpus.conditions(condition_columns)[train_rows]
    )
    x = _as_tensor(standardization.encode(corpus.boundaries))
    y = scaler.encode(_as_tensor(corpus.conditions(condition_columns)))
    if not variant.conditional:
        y = torch.zeros(len(corpus), 0)

    model = models.CVAE(
        x_dim=x.shape[-1],
        y_dim=y.shape[-1],
        latent_dim=latent_dim,
        hidden_dims=list(HIDDEN_DIMS),
    )
    validation_rows = corpus.split == "val"
    _LOG.info(
        "training on %d rows, validating on %d, %d features",
        int(train_rows.sum()),
        int(validation_rows.sum()),
        x.shape[-1],
    )
    history = train_mod.train(
        model,
        x[train_rows],
        y[train_rows],
        config,
        predict_metrics=predictor,
        constraints=constraints,
        x_val=x[validation_rows],
        y_val=y[validation_rows],
    )
    return Trained(
        model=model.eval(),
        standardization=standardization,
        scaler=scaler,
        condition_columns=tuple(condition_columns),
        history=history,
    )


def _check_conditioning_is_predicted(
    condition_columns: tuple[str, ...],
    predictor: surrogate_mod.StandardizedPredictor,
) -> None:
    """Self-consistent conditioning later needs every conditioning column predicted."""
    missing = [c for c in condition_columns if c not in predictor.metric_names]
    if missing:
        raise KeyError(
            f"the surrogate does not predict {missing}, so the decoder cannot be "
            f"conditioned on them; it predicts {list(predictor.metric_names)}"
        )


def _as_tensor(values: NDArray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)
