"""Train the four variants the paper compares, on a constructed benchmark.

A variant is two independent switches, not four code paths: whether the decoder is
conditioned on metrics, and whether the chord regularizer is on. Everything else is
shared, which is the point of comparing them.

Training runs through `chord.train` unchanged. The only thing this module supplies is
the adapter that lets an analytically specified problem stand where a learned surrogate
normally does, so a result here says something about the method rather than about a
separate implementation of it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from numpy.typing import NDArray

from ... import inference as inference_mod
from ... import models
from ... import train as train_mod
from .. import variants as variants_mod
from . import datasets
from . import problem as problem_mod

VARIANTS = variants_mod.VARIANTS
"""Re-exported so running an experiment needs one import, not two."""


def training_config(
    variant: variants_mod.Variant, n_epochs: int = 200
) -> train_mod.TrainingConfig:
    """Settings the constructed-benchmark results were produced with.

    These are smaller and gentler than the stellarator defaults. The corpora here are a
    few thousand points rather than a hundred thousand, so a long cyclical KL schedule
    would spend most of training annealing.
    """
    return train_mod.TrainingConfig(
        n_epochs=n_epochs,
        batch_size=256,
        learning_rate=5e-4,
        cosine_decay=False,
        kl_weight=0.5,
        free_bits=0.05,
        kl_anneal="linear",
        kl_anneal_epochs=30,
        chord_lambda=variant.chord_lambda,
        chord_warmup_start_epoch=30,
        chord_warmup_epochs=30,
        chord_n_pairs=32,
    )


@dataclasses.dataclass(frozen=True)
class Trained:
    """A fitted model and everything needed to use it on raw designs."""

    model: models.CVAE
    design_scaler: inference_mod.ConditionScaler
    condition_scaler: inference_mod.ConditionScaler
    validation: NDArray
    """Indices of the rows held out, which interpolation is evaluated on."""

    history: list[dict[str, float]]

    @property
    def conditional(self) -> bool:
        return self.model.y_dim > 0

    def encode(self, designs: NDArray, conditions: NDArray) -> torch.Tensor:
        """Latent codes for raw designs, standardizing on the way in."""
        with torch.no_grad():
            return self.model.encode(
                self.design_scaler.encode(_as_tensor(designs)),
                self._conditioning(conditions, len(designs)),
            )

    def decode(self, z: torch.Tensor, conditions: NDArray) -> NDArray:
        """Raw designs for latent codes, undoing standardization on the way out."""
        with torch.no_grad():
            decoded = self.model.decode(z, self._conditioning(conditions, len(z)))
        return self.design_scaler.decode(decoded).numpy()

    def standardized_conditions(self, conditions: NDArray) -> torch.Tensor:
        """Conditioning as the decoder consumes it, empty for an unconditional model."""
        return self._conditioning(conditions, len(conditions))

    def _conditioning(self, conditions: NDArray, batch_size: int) -> torch.Tensor:
        if not self.conditional:
            return torch.zeros(batch_size, 0)
        return self.condition_scaler.encode(_as_tensor(conditions))


def train(
    problem: problem_mod.Problem,
    dataset: datasets.Dataset,
    variant: variants_mod.Variant,
    n_epochs: int = 200,
    latent_dim: int = 2,
    hidden_width: int = 64,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Trained:
    """Fit one variant and return it alongside the transforms it was fitted with."""
    torch.manual_seed(seed)
    train_rows, validation_rows = _split(len(dataset), validation_fraction, seed)

    design_scaler = inference_mod.ConditionScaler.fit(dataset.designs[train_rows])
    condition_scaler = inference_mod.ConditionScaler.fit(dataset.conditions[train_rows])
    x = design_scaler.encode(_as_tensor(dataset.designs))
    y = condition_scaler.encode(_as_tensor(dataset.conditions))
    if not variant.conditional:
        y = torch.zeros(len(dataset), 0)

    model = models.CVAE(
        x_dim=problem.n_dim,
        y_dim=y.shape[-1],
        latent_dim=latent_dim,
        hidden_dims=[hidden_width] * 3,
        use_residual=False,
    )
    predictor = problem_mod.StandardizedViolations(problem, design_scaler)
    config = training_config(variant, n_epochs)
    history = train_mod.train(
        model,
        x[train_rows],
        y[train_rows],
        config,
        predict_metrics=predictor,
        constraints=problem_mod.constraint_set(problem),
        x_val=x[validation_rows],
        y_val=y[validation_rows],
    )
    return Trained(
        model=model.eval(),
        design_scaler=design_scaler,
        condition_scaler=condition_scaler,
        validation=validation_rows,
        history=history,
    )


def _split(n: int, validation_fraction: float, seed: int) -> tuple[NDArray, NDArray]:
    order = np.random.default_rng(seed).permutation(n)
    n_validation = int(round(validation_fraction * n))
    return order[n_validation:], order[:n_validation]


def _as_tensor(values: NDArray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)
