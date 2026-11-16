"""Conditional VAE architecture for feasibility-preserving latent spaces.

The same module defines all four variants compared in the paper. A variant is
selected by two independent choices: whether the decoder is conditioned on a
metric vector `y`, and whether the chord regularizer is applied during training
(see `chord.train`). Setting `y_dim=0` yields an unconditional VAE, in which
case `y` arguments are accepted and ignored so that callers need no branching.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block: `Linear` then `LayerNorm` then `ReLU`."""

    def __init__(self, dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.linear(x)))


class MLPBlock(nn.Module):
    """Multi-layer perceptron with optional layer norm and residual connections.

    Residual connections require every hidden layer to share a width, so the
    input is first projected to `hidden_dims[0]` and residual blocks are stacked
    on top. When the widths differ, or `use_residual` is false, a plain MLP is
    built instead.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        uniform_width = len(hidden_dims) > 0 and len(set(hidden_dims)) == 1

        if use_residual and uniform_width:
            width = hidden_dims[0]
            layers.append(nn.Linear(input_dim, width))
            if use_layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU())
            layers.extend(
                ResidualBlock(width, use_layer_norm)
                for _ in range(len(hidden_dims) - 1)
            )
            layers.append(nn.Linear(width, output_dim))
        else:
            prev = input_dim
            for width in hidden_dims:
                layers.append(nn.Linear(prev, width))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers.append(nn.ReLU())
                prev = width
            layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Maps a design `x` and conditioning vector `y` to latent `(mu, logvar)`."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        input_dim = x_dim + y_dim
        head_dim = hidden_dims[-1] if hidden_dims else input_dim
        trunk_dims = hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims
        self.backbone = MLPBlock(
            input_dim, head_dim, trunk_dims, use_layer_norm, use_residual
        )
        self.fc_mu = nn.Linear(head_dim, latent_dim)
        self.fc_logvar = nn.Linear(head_dim, latent_dim)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(torch.cat([x, y], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Maps a latent code `z` and conditioning vector `y` back to a design `x`."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.net = MLPBlock(
            latent_dim + y_dim,
            x_dim,
            list(reversed(hidden_dims)),
            use_layer_norm,
            use_residual,
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, y], dim=-1))


class CVAE(nn.Module):
    """Conditional variational autoencoder over designs and their metrics.

    With `y_dim=0` the model degenerates to a plain VAE: `y` is replaced by a
    zero-width tensor before every encode and decode, so the same call signature
    serves both the conditional and unconditional variants.
    """

    def __init__(
        self,
        x_dim: int = 2,
        y_dim: int = 2,
        latent_dim: int = 2,
        hidden_dims: list[int] | None = None,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            x_dim, y_dim, latent_dim, hidden_dims, use_layer_norm, use_residual
        )
        self.decoder = Decoder(
            x_dim, y_dim, latent_dim, hidden_dims, use_layer_norm, use_residual
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self._y_or_empty(y, x.shape[0])
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Latent mean of `x`, without sampling."""
        y = self._y_or_empty(y, x.shape[0])
        mu, _ = self.encoder(x, y)
        return mu

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Design decoded from latent code `z`."""
        y = self._y_or_empty(y, z.shape[0])
        return self.decoder(z, y)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from `N(mu, exp(logvar))` with a gradient path to the parameters."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _y_or_empty(self, y: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.y_dim == 0:
            return y.new_zeros(batch_size, 0)
        return y
