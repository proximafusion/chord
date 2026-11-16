"""Tests for the Fourier feature transform."""

import numpy as np
import pytest

from . import features


def _raw_batch(n: int = 32, seed: int = 0) -> np.ndarray:
    """Coefficients with structurally zero entries actually zeroed, as real data is.

    A fresh generator per call rather than one shared at module level, so what a test
    sees does not depend on how many draws the tests before it happened to make.
    """
    raw = np.random.default_rng(seed).normal(scale=0.1, size=(n, features.FEATURE_DIM))
    raw[:, features.structurally_zero_mask()] = 0.0
    return raw


def test_index_grid_covers_every_coefficient() -> None:
    m_grid, n_grid = features.fourier_index_grid()

    assert m_grid.shape == (features.FEATURE_DIM,)
    assert n_grid.shape == (features.FEATURE_DIM,)
    assert m_grid.min() == 0 and m_grid.max() == features.POLOIDAL_MODES
    assert n_grid.min() == -features.TOROIDAL_MODES
    assert n_grid.max() == features.TOROIDAL_MODES


def test_grid_repeats_across_the_two_fields() -> None:
    """`r_cos` and `z_sin` share a mode layout, so the grid is a repeat."""
    m_grid, n_grid = features.fourier_index_grid()
    half = features.COEFFICIENTS_PER_FIELD

    np.testing.assert_array_equal(m_grid[:half], m_grid[half:])
    np.testing.assert_array_equal(n_grid[:half], n_grid[half:])


def test_spectrum_scale_decays_with_mode_number() -> None:
    m_grid, n_grid = features.fourier_index_grid()
    scale = features.spectrum_scale(alpha=1.0)
    order = np.maximum(np.abs(m_grid), np.abs(n_grid))

    np.testing.assert_allclose(scale, np.exp(-order))
    assert scale[order == 0].min() > scale[order == features.POLOIDAL_MODES].max()


def test_zero_alpha_disables_whitening() -> None:
    np.testing.assert_allclose(
        features.spectrum_scale(alpha=0.0), np.ones(features.FEATURE_DIM)
    )


def test_structurally_zero_mask_counts_match_symmetry() -> None:
    """Nine entries vanish: four in `r_cos` and five in `z_sin`."""
    mask = features.structurally_zero_mask()
    half = features.COEFFICIENTS_PER_FIELD

    assert mask[:half].sum() == features.TOROIDAL_MODES
    assert mask[half:].sum() == features.TOROIDAL_MODES + 1
    assert mask.sum() == 2 * features.TOROIDAL_MODES + 1


def test_vector_and_coefficients_round_trip() -> None:
    raw = _raw_batch()

    r_cos, z_sin = features.vector_to_coefficients(raw)
    restored = features.coefficients_to_vector(r_cos, z_sin)

    np.testing.assert_allclose(restored, raw)


def test_vector_split_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="expected last dimension"):
        features.vector_to_coefficients(np.zeros((4, 7)))


def test_encode_decode_round_trips() -> None:
    """The decoder renders every proposed design, so this must be exact."""
    raw = _raw_batch()
    standardization = features.Standardization.fit(raw)

    restored = standardization.decode(standardization.encode(raw))

    np.testing.assert_allclose(restored, raw, atol=1e-6)


def test_decode_restores_structural_zeros() -> None:
    raw = _raw_batch()
    standardization = features.Standardization.fit(raw)

    restored = standardization.decode(standardization.encode(raw))

    assert np.all(restored[:, features.structurally_zero_mask()] == 0.0)


def test_encoded_features_are_standardized() -> None:
    raw = _raw_batch(512)
    standardization = features.Standardization.fit(raw)

    encoded = standardization.encode(raw)

    np.testing.assert_allclose(encoded.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(encoded.std(axis=0), 1.0, atol=1e-4)


def test_feature_width_excludes_dropped_columns() -> None:
    standardization = features.Standardization.fit(_raw_batch())

    assert standardization.n_features == features.FEATURE_DIM - 9


def test_constant_column_does_not_produce_infinities() -> None:
    """A column with no variance must not be divided by a zero standard deviation."""
    raw = _raw_batch()
    kept = ~features.structurally_zero_mask()
    first_kept = int(np.flatnonzero(kept)[0])
    raw[:, first_kept] = 0.5

    encoded = features.Standardization.fit(raw).encode(raw)

    assert np.all(np.isfinite(encoded))


def test_fit_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected an"):
        features.Standardization.fit(np.zeros((4, 7)))


def test_decode_rejects_wrong_width() -> None:
    standardization = features.Standardization.fit(_raw_batch())

    with pytest.raises(ValueError, match="expected last dimension"):
        standardization.decode(np.zeros((4, 3)))


def test_statistics_come_from_the_fitted_split_only() -> None:
    """Encoding unseen data must reuse the fitted statistics, not refit them."""
    train = _raw_batch(256)
    shifted = _raw_batch(256) + 1.0
    shifted[:, features.structurally_zero_mask()] = 0.0
    standardization = features.Standardization.fit(train)

    encoded = standardization.encode(shifted)

    assert abs(float(encoded.mean())) > 0.1
