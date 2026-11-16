"""Tests for the conditional VAE architecture."""

import pytest
import torch

from . import models

X_DIM = 7
Y_DIM = 3
LATENT_DIM = 4
BATCH = 5


def test_cvae_forward_shapes() -> None:
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    x = torch.randn(BATCH, X_DIM)
    y = torch.randn(BATCH, Y_DIM)

    x_recon, mu, logvar = model(x, y)

    assert x_recon.shape == (BATCH, X_DIM)
    assert mu.shape == (BATCH, LATENT_DIM)
    assert logvar.shape == (BATCH, LATENT_DIM)


def test_unconditional_vae_ignores_y() -> None:
    """With `y_dim=0` the decoder must not depend on `y`, so both variants share a
    path."""
    model = models.CVAE(X_DIM, y_dim=0, latent_dim=LATENT_DIM, hidden_dims=[16, 16])
    z = torch.randn(BATCH, LATENT_DIM)

    first = model.decode(z, torch.randn(BATCH, Y_DIM))
    second = model.decode(z, torch.zeros(BATCH, Y_DIM))

    torch.testing.assert_close(first, second)


def test_conditional_decoder_depends_on_y() -> None:
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    z = torch.randn(BATCH, LATENT_DIM)

    first = model.decode(z, torch.zeros(BATCH, Y_DIM))
    second = model.decode(z, torch.ones(BATCH, Y_DIM))

    assert not torch.allclose(first, second)


def test_encode_is_deterministic() -> None:
    """`encode` returns the latent mean, so repeated calls must agree."""
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    x = torch.randn(BATCH, X_DIM)
    y = torch.randn(BATCH, Y_DIM)

    torch.testing.assert_close(model.encode(x, y), model.encode(x, y))


def test_reparameterize_matches_distribution() -> None:
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    mu = torch.full((20000, 2), 3.0)
    logvar = torch.full((20000, 2), -1.0)

    samples = model.reparameterize(mu, logvar)

    torch.testing.assert_close(samples.mean(0), mu[0], atol=0.02, rtol=0)
    expected_std = torch.exp(torch.tensor(-0.5))
    torch.testing.assert_close(
        samples.std(0), expected_std.expand(2), atol=0.02, rtol=0
    )


def test_zero_variance_reparameterize_returns_mean() -> None:
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    mu = torch.randn(BATCH, LATENT_DIM)
    logvar = torch.full((BATCH, LATENT_DIM), -60.0)

    torch.testing.assert_close(model.reparameterize(mu, logvar), mu)


@pytest.mark.parametrize("hidden_dims", [[16, 16, 16], [32, 16], []])
def test_mlp_block_shapes_across_widths(hidden_dims: list[int]) -> None:
    """Uniform widths take the residual path; mixed or empty widths fall back."""
    block = models.MLPBlock(X_DIM, Y_DIM, hidden_dims)

    assert block(torch.randn(BATCH, X_DIM)).shape == (BATCH, Y_DIM)


def test_residual_block_is_identity_at_zero_weights() -> None:
    block = models.ResidualBlock(X_DIM)
    torch.nn.init.zeros_(block.linear.weight)
    torch.nn.init.zeros_(block.linear.bias)
    x = torch.randn(BATCH, X_DIM)

    torch.testing.assert_close(block(x), x)


def test_gradients_reach_encoder_and_decoder() -> None:
    model = models.CVAE(X_DIM, Y_DIM, LATENT_DIM, hidden_dims=[16, 16])
    x = torch.randn(BATCH, X_DIM)
    y = torch.randn(BATCH, Y_DIM)

    x_recon, _, _ = model(x, y)
    x_recon.sum().backward()

    assert model.encoder.fc_mu.weight.grad is not None
    assert model.decoder.net.net[0].weight.grad is not None
