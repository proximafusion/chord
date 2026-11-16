"""Load saved front models and interpolate their encoded training anchors.

These artifacts preserve the paper's eight-component conditioning and separate
generator/surrogate normalization. They are distinct from the fixed-AR corpus and the
exploratory `frontfill.run fill` workflow. Input NPZs contain numeric or Unicode arrays;
loading never enables NumPy pickle deserialization.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import torch
from chord import artifacts, features, inference, models, surrogate
from chord.experiments.frontfill import gaps, prepared_types
from chord.experiments.stellarator import training


def load_dataset(name: str) -> prepared_types.PreparedDataset:
    """Load the archived arrays without changing their splits or normalization.

    Args:
        name: NPZ path relative to `CHORD_ARTIFACT_BASE`.

    Returns:
        Standardized training, validation and held-out inputs with their transforms.
    """
    with (
        artifacts.open_artifact(name) as handle,
        np.load(handle, allow_pickle=False) as data,
    ):
        splits = {
            split: prepared_types.PreparedSplit(
                torch.tensor(data[f"x_{split}"], dtype=torch.float32),
                torch.tensor(data[f"y_{split}"], dtype=torch.float32),
                torch.tensor(data[f"ar_{split}"], dtype=torch.float32),
            )
            for split in ("train", "val", "test")
        }
        return prepared_types.PreparedDataset(
            splits=splits,
            standardization=features.Standardization(
                alpha=float(data["spectrum_scale_alpha"]),
                kept=data["feature_mask"].copy(),
                mean=data["x_mean"].copy(),
                std=data["x_std"].copy(),
            ),
            scaler=inference.ConditionScaler(
                data["y_mean"].copy(), data["y_std"].copy()
            ),
            condition_names=tuple(data["condition_names"].tolist()),
            template_boundary_json=str(data["template_boundary_json"].item()),
        )


def load_model(
    dataset: prepared_types.PreparedDataset, checkpoint: str, config: str
) -> training.Trained:
    """Restore a decoder using its saved architecture and dataset transforms.

    Args:
        dataset: The dataset paired with the checkpoint in the release manifest.
        checkpoint: Tensor-only state dictionary relative to the artifact base.
        config: Archived training configuration relative to the artifact base.

    Returns:
        A CPU model in evaluation mode, with the original conditioning order.
    """
    with artifacts.open_artifact(config) as handle:
        settings = json.load(handle)
    model = models.CVAE(
        x_dim=dataset.standardization.n_features,
        y_dim=len(dataset.condition_names),
        latent_dim=settings["latent_dim"],
        hidden_dims=settings["hidden_dims"],
        use_layer_norm=settings["use_layer_norm"],
        use_residual=settings["use_residual"],
    )
    with artifacts.open_artifact(checkpoint) as handle:
        model.load_state_dict(torch.load(handle, map_location="cpu", weights_only=True))
    return training.Trained(
        model.eval(),
        dataset.standardization,
        dataset.scaler,
        dataset.condition_names,
        [],
    )


def load_surrogate(checkpoint: str, dataset: str) -> surrogate.StandardizedMetrics:
    """Restore the archived metric predictor with its own feature statistics.

    Args:
        checkpoint: Original tensor-only surrogate weights, relative to the base.
        dataset: Paired surrogate NPZ, containing metric and feature normalization.

    Returns:
        Frozen predictor in benchmark units. The unused convergence head is omitted.
    """
    with (
        artifacts.open_artifact(dataset) as handle,
        np.load(handle, allow_pickle=False) as data,
    ):
        names = tuple(data["metric_names"].tolist())
        transform = features.Standardization(
            alpha=float(data["spectrum_scale_alpha"]),
            kept=data["feature_mask"].copy(),
            mean=data["x_mean"].copy(),
            std=data["x_std"].copy(),
        )
        mean = torch.tensor(data["metrics_mean"], dtype=torch.float32)
        std = torch.tensor(data["metrics_std"], dtype=torch.float32)
    model = surrogate.SurrogateModel(x_dim=transform.n_features, n_metrics=len(names))
    with artifacts.open_artifact(checkpoint) as handle:
        state = torch.load(handle, map_location="cpu", weights_only=True)
    # The paper's interpolation uses only the metric head, not the convergence logit.
    model.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("ok_head.")}
    )
    return surrogate.StandardizedMetrics(model, mean, std, names, transform).freeze()


def decode_sweep(
    trained: training.Trained,
    dataset: prepared_types.PreparedDataset,
    gap: gaps.Gap,
    alphas: Sequence[float],
    counts: Sequence[int],
    seed: int = 0,
    predictor: surrogate.StandardizedMetrics | None = None,
    iterations: int = 0,
    margin: float = 0.02,
    clamp_qi: bool = True,
) -> prepared_types.DecodedSweep:
    """Decode the archived anchor-interpolation protocol, optionally adapting
    conditions.

    Args:
        trained: Restored model with its original feature and condition transforms.
        dataset: Paired dataset; endpoints are selected only from its training split.
        gap: Two bracketing anchor aspect ratios.
        alphas: Weights on the low anchor, matching the historical convention.
        counts: Number of endpoint pairs sampled independently at each weight.
        seed: NumPy endpoint-sampling seed.
        predictor: Per-gap metric predictor, required for conditioning updates.
        iterations: Number of updates; zero gives the archived plain blend.
        margin: Absolute inward clamp margin in each metric's units.
        clamp_qi: Whether updates clamp QI as well as the other constrained metrics.

    Returns:
        Decoded raw coefficients with blend weights and training-row indices.

    Raises:
        ValueError: The sweep is empty, lacks an anchor, or omits a required predictor.
    """
    if len(alphas) != len(counts) or not len(alphas) or any(n < 1 for n in counts):
        raise ValueError("provide equally sized, nonempty weights and positive counts")
    if iterations < 0 or (iterations and predictor is None):
        raise ValueError(
            "nonnegative iterations and a predictor are required for updates"
        )
    if predictor is not None and iterations:
        source = trained.standardization
        target = predictor.standardization
        if source.alpha != target.alpha or not np.array_equal(source.kept, target.kept):
            raise ValueError(
                "decoder and surrogate must use the same Fourier whitening"
            )
    split = dataset.splits["train"]
    ar = np.round(split.aspect_ratio.numpy(), 2)
    low = np.flatnonzero(np.isclose(ar, gap.low, atol=gaps.MATCH_TOLERANCE))
    high = np.flatnonzero(np.isclose(ar, gap.high, atol=gaps.MATCH_TOLERANCE))
    if not len(low) or not len(high):
        raise ValueError(f"{gap.name}: prepared training split lacks an anchor")
    rng = np.random.default_rng(seed)
    boundaries, weights, lows, highs = [], [], [], []
    with torch.no_grad():
        for alpha, count in zip(alphas, counts):
            li = low[rng.integers(0, len(low), size=count)]
            hi = high[rng.integers(0, len(high), size=count)]
            z = alpha * trained.model.encode(split.x[li], split.y[li]) + (
                1.0 - alpha
            ) * trained.model.encode(split.x[hi], split.y[hi])
            y = alpha * split.y[li] + (1.0 - alpha) * split.y[hi]
            if predictor is not None and iterations:
                y = _update_conditions(
                    trained, predictor, z, y, iterations, margin, clamp_qi
                )
            raw = trained.standardization.decode(trained.model.decode(z, y).numpy())
            boundaries.append(torch.from_numpy(raw))
            weights.append(torch.full((count,), alpha, dtype=torch.float64))
            lows.append(torch.from_numpy(li))
            highs.append(torch.from_numpy(hi))
    return prepared_types.DecodedSweep(
        torch.cat(boundaries), torch.cat(weights), torch.cat(lows), torch.cat(highs)
    )


def _update_conditions(
    trained: training.Trained,
    predictor: surrogate.StandardizedMetrics,
    z: torch.Tensor,
    y: torch.Tensor,
    iterations: int,
    margin: float,
    clamp_qi: bool,
) -> torch.Tensor:
    """Apply the archived metric clamps while preserving each pair's blended AR.

    Args:
        trained: Model and decoder normalization.
        predictor: Frozen per-gap surrogate with its own normalization.
        z: Fixed interpolated latent coordinates.
        y: Initial standardized conditioning.
        iterations: Number of decoder/surrogate updates.
        margin: Absolute inward constraint margin.
        clamp_qi: Whether to clamp the QI condition.

    Returns:
        Updated standardized conditions with the original AR entries retained.
    """
    ar_index = trained.condition_columns.index("aspect_ratio")
    ar_pinned = y[:, ar_index].clone()
    columns = [predictor.metric_names.index(name) for name in trained.condition_columns]
    source = trained.standardization
    target = predictor.standardization
    source_mean, source_std = torch.as_tensor(source.mean), torch.as_tensor(source.std)
    target_mean, target_std = torch.as_tensor(target.mean), torch.as_tensor(target.std)
    bounds = {
        "edge_rotational_transform_over_n_field_periods": (0.25 + margin, None),
        "edge_magnetic_mirror_ratio": (None, 0.25 - margin),
        "flux_compression_in_regions_of_bad_curvature": (None, 0.9 - margin),
        "vacuum_well": (margin, None),
    }
    if clamp_qi:
        bounds["log10_qi"] = (None, -3.5 - margin)
    for _ in range(iterations):
        white = trained.model.decode(z, y) * source_std + source_mean
        raw = predictor((white - target_mean) / target_std)[:, columns].clone()
        for name, (lower, upper) in bounds.items():
            index = trained.condition_columns.index(name)
            raw[:, index] = raw[:, index].clamp(min=lower, max=upper)
        y = trained.scaler.encode(raw)
        y[:, ar_index] = ar_pinned
    return y
