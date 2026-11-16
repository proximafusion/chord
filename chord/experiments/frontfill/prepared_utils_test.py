"""Check checkpoint pairing and that held-out rows never seed archived inference."""

import json
import pathlib

import numpy as np
import pytest
import torch
from chord import features, models
from chord.experiments.frontfill import gaps, prepared_utils


def test_loaded_splits_preserve_conditions_and_only_training_seeds_sweep(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap = gaps.all_gaps()[0]
    kept = ~features.structurally_zero_mask()
    width = int(kept.sum())
    x = np.arange(2 * width, dtype=np.float32).reshape(2, width) / 256
    y = np.array([[0, -1], [0, 1]], dtype=np.float32)
    mean = np.full(width, 2.0, dtype=np.float32)
    std = np.full(width, 0.5, dtype=np.float32)
    data = {
        "x_train": x,
        "y_train": y,
        "ar_train": np.array([gap.low, gap.high], dtype=np.float32),
        "x_mean": mean,
        "x_std": std,
        "y_mean": np.array([-3.5, 8.625], dtype=np.float32),
        "y_std": np.array([0.25, 0.625], dtype=np.float32),
        "feature_mask": kept,
        "spectrum_scale_alpha": np.array(1.0),
        "condition_names": np.array(["log10_qi", "aspect_ratio"]),
        "template_boundary_json": np.array("{}"),
    }
    for split in ("val", "test"):
        data[f"x_{split}"] = x + 1000
        data[f"y_{split}"] = y + 1000
        data[f"ar_{split}"] = data["ar_train"].copy()
    np.savez(tmp_path / "dataset.npz", **data)
    config = {
        "latent_dim": 2,
        "hidden_dims": [8, 8],
        "use_layer_norm": True,
        "use_residual": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = models.CVAE(x_dim=width, y_dim=2, **config).eval()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    monkeypatch.setenv("CHORD_ARTIFACT_BASE", str(tmp_path))

    dataset = prepared_utils.load_dataset("dataset.npz")
    loaded = prepared_utils.load_model(dataset, "model.pt", "config.json")
    sweep = prepared_utils.decode_sweep(loaded, dataset, gap, [0.0, 1.0], [1, 1])

    with torch.no_grad():
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        reconstruction = model.decode(model.encode(tx, ty), ty).numpy()
    white = reconstruction * std + mean
    expected = np.zeros((2, features.FEATURE_DIM))
    expected[:, kept] = white
    expected *= features.spectrum_scale()
    np.testing.assert_allclose(
        sweep.boundaries.numpy(), expected[::-1], rtol=1e-6, atol=1e-7
    )
    assert sweep.low_train_index.tolist() == [0, 0]
    assert sweep.high_train_index.tolist() == [1, 1]
    np.testing.assert_allclose(
        dataset.splits["test"].x.numpy(), x + 1000, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        dataset.scaler.decode(dataset.splits["train"].y).numpy()[:, 1],
        [gap.low, gap.high],
        rtol=0,
        atol=0,
    )
