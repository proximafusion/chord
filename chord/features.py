"""Invertible transform between Fourier coefficients and model features.

A plasma boundary is a truncated Fourier series. Its coefficients span several
orders of magnitude, since high-order modes are small by construction, so a
network trained on them directly would be dominated by the low-order terms.
Three steps fix that, and the decoder has to undo all three before a predicted
vector can be handed to the simulator:

1. Spectral whitening, dividing each coefficient by `exp(-alpha * max(|m|, |n|))`
   so every mode lands in a comparable range. This matches the scaling the
   optimization campaigns use for their variable bounds.
2. Dropping coefficients that are structurally zero for a stellarator-symmetric
   surface, which carry no information and would otherwise have zero variance.
3. Per-column standardization using statistics from the training split alone.

`Standardization` holds the fitted parameters and implements both directions.
Round-tripping matters more than it looks: every design the model proposes
reaches the simulator through `decode`, so an error there is indistinguishable
from the model generating bad designs.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

POLOIDAL_MODES = 4
"""Highest poloidal mode number retained, giving rows `m = 0..4`."""

TOROIDAL_MODES = 4
"""Highest toroidal mode number retained, giving columns `n = -4..+4`."""

_POLOIDAL_ROWS = POLOIDAL_MODES + 1
_TOROIDAL_COLS = 2 * TOROIDAL_MODES + 1
COEFFICIENTS_PER_FIELD = _POLOIDAL_ROWS * _TOROIDAL_COLS
FEATURE_DIM = 2 * COEFFICIENTS_PER_FIELD
"""Length of a raw coefficient vector, `r_cos` followed by `z_sin`."""


def fourier_index_grid() -> tuple[NDArray, NDArray]:
    """Mode numbers `(m, n)` for each entry of a raw coefficient vector."""
    m_2d = np.arange(_POLOIDAL_ROWS).reshape(-1, 1).repeat(_TOROIDAL_COLS, axis=1)
    n_2d = (
        np.arange(_TOROIDAL_COLS).reshape(1, -1).repeat(_POLOIDAL_ROWS, axis=0)
        - TOROIDAL_MODES
    )
    m_flat = m_2d.flatten()
    n_flat = n_2d.flatten()
    return (
        np.concatenate([m_flat, m_flat]).astype(np.int32),
        np.concatenate([n_flat, n_flat]).astype(np.int32),
    )


def spectrum_scale(alpha: float = 1.0) -> NDArray:
    """Per-coefficient scale `exp(-alpha * max(|m|, |n|))`."""
    m_grid, n_grid = fourier_index_grid()
    return np.exp(-alpha * np.maximum(np.abs(m_grid), np.abs(n_grid)))


def structurally_zero_mask() -> NDArray:
    """Coefficients forced to zero by stellarator symmetry.

    For `r_cos` these are the `m = 0, n < 0` entries, and for `z_sin` the
    `m = 0, n <= 0` entries, since those duplicate or vanish under the symmetry.
    """
    m_grid, n_grid = fourier_index_grid()
    is_r_cos = np.arange(FEATURE_DIM) < COEFFICIENTS_PER_FIELD
    return (is_r_cos & (m_grid == 0) & (n_grid < 0)) | (
        ~is_r_cos & (m_grid == 0) & (n_grid <= 0)
    )


def vector_to_coefficients(x: NDArray) -> tuple[NDArray, NDArray]:
    """Split a raw vector into `(r_cos, z_sin)` arrays of mode numbers."""
    if x.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"expected last dimension {FEATURE_DIM}, got shape {tuple(x.shape)}"
        )
    shape = (*x.shape[:-1], _POLOIDAL_ROWS, _TOROIDAL_COLS)
    r_cos = x[..., :COEFFICIENTS_PER_FIELD].reshape(shape)
    z_sin = x[..., COEFFICIENTS_PER_FIELD:].reshape(shape)
    return r_cos, z_sin


def coefficients_to_vector(r_cos: NDArray, z_sin: NDArray) -> NDArray:
    """Flatten `(r_cos, z_sin)` arrays back into a raw vector."""
    if r_cos.shape != z_sin.shape:
        raise ValueError(
            f"coefficient arrays disagree in shape: {tuple(r_cos.shape)} "
            f"against {tuple(z_sin.shape)}"
        )
    leading = r_cos.shape[:-2]
    return np.concatenate(
        [r_cos.reshape(*leading, -1), z_sin.reshape(*leading, -1)], axis=-1
    )


@dataclasses.dataclass(frozen=True)
class Standardization:
    """Fitted parameters of the transform, and both directions of it."""

    alpha: float
    kept: NDArray
    """Boolean mask over raw coefficients, false where structurally zero."""

    mean: NDArray
    std: NDArray

    @classmethod
    def fit(cls, raw: NDArray, alpha: float = 1.0) -> Standardization:
        """Fit standardization statistics on raw training coefficients."""
        if raw.ndim != 2 or raw.shape[-1] != FEATURE_DIM:
            raise ValueError(
                f"expected an (n, {FEATURE_DIM}) array, got shape {tuple(raw.shape)}"
            )
        kept = ~structurally_zero_mask()
        whitened = (raw / spectrum_scale(alpha))[:, kept]
        mean = whitened.mean(axis=0)
        std = whitened.std(axis=0)
        # A column of constant value carries nothing, and dividing by its zero
        # standard deviation would put infinities into the training data.
        std = np.where(std > 1e-12, std, 1.0)
        return cls(alpha=alpha, kept=kept, mean=mean, std=std)

    @property
    def n_features(self) -> int:
        """Width of the model's input, after dropped columns."""
        return int(self.kept.sum())

    def encode(self, raw: NDArray) -> NDArray:
        """Raw coefficients to model features."""
        whitened = raw / spectrum_scale(self.alpha)
        return ((whitened[..., self.kept] - self.mean) / self.std).astype(np.float32)

    def decode(self, features: NDArray) -> NDArray:
        """Model features back to raw coefficients.

        Columns dropped as structurally zero are restored as zeros, which is their only
        admissible value.
        """
        if features.shape[-1] != self.n_features:
            raise ValueError(
                f"expected last dimension {self.n_features}, got shape "
                f"{tuple(features.shape)}"
            )
        whitened = features * self.std + self.mean
        raw = np.zeros((*features.shape[:-1], FEATURE_DIM), dtype=np.float64)
        raw[..., self.kept] = whitened
        return raw * spectrum_scale(self.alpha)
