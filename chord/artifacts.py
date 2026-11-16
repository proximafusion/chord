"""Reading published artifacts, and adapting a released surrogate's conventions.

Artifacts resolve against a single base, so a location change is one
environment variable rather than an edit in every module that reads data. The
base can point to a Hugging Face repository with `hf://` or to a local directory.

The released surrogate was trained before this package adopted constellaration's
metric conventions, so it predicts under the project's original names and units.
`ConstellarationUnits` translates, in one place and with the conversions stated
explicitly, rather than leaving a rename to be inferred from a name that happens
to look similar.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import IO, Any

import fsspec  # pants: no-infer-dep
import numpy as np
import torch
from numpy.typing import NDArray

from . import feasibility, features
from . import surrogate as surrogate_mod

ARTIFACT_BASE_ENV = "CHORD_ARTIFACT_BASE"
"""Environment variable overriding where published artifacts are read from."""

DEFAULT_ARTIFACT_BASE = (
    "hf://datasets/proxima-fusion/chord-paper@28f84539fddd171d950d5f7d4a8878dcc649ad15"
)
"""Where artifacts are read from when nothing overrides it.

The published location, so a fresh checkout works without configuration. Anywhere with
a different copy sets `CHORD_ARTIFACT_BASE` instead of editing this.

This is also the single line that identifies the authors, which matters for a
double-blind submission: an anonymized mirror is one environment variable at runtime, or
one line here if the default has to be neutral in the submitted archive.
"""

LOG10_QI = feasibility.LOG10_QI
EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO = (
    feasibility.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO
)


@dataclasses.dataclass(frozen=True)
class MetricConversion:
    """How one of the surrogate's outputs maps onto this package's convention."""

    released_name: str
    name: str
    scale_by_field_periods: int = 0
    """Power of the field-period count to multiply by.

    Rotational transforms are reported per field period and so divide by it; the
    magnetic gradient scale length is normalized by it and so multiplies.
    """


RELEASED_SURROGATE_CONVERSIONS: tuple[MetricConversion, ...] = (
    MetricConversion("iota_edge", feasibility.IOTA_EDGE, -1),
    MetricConversion("iota_axis", "axis_rotational_transform_over_n_field_periods", -1),
    MetricConversion("magnetic_mirror_ratio_edge", feasibility.MIRROR_RATIO_EDGE),
    MetricConversion("magnetic_mirror_ratio_axis", "axis_magnetic_mirror_ratio"),
    MetricConversion("vacuum_well", feasibility.VACUUM_WELL),
    MetricConversion("max_elongation", "max_elongation"),
    MetricConversion("aspect_ratio", "aspect_ratio"),
    MetricConversion("log10_qi_2023_edge", LOG10_QI),
    MetricConversion("aeff_minus_ar", EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO),
    MetricConversion(
        "min_normalized_magnetic_gradient_scale_length",
        "minimum_normalized_magnetic_gradient_scale_length",
        1,
    ),
    MetricConversion(
        "landreman_2025_bad_curvature_rho2_049",
        "flux_compression_in_regions_of_bad_curvature",
    ),
)
"""Outputs of the released surrogate, under this package's names and units.

Only `landreman_2025_bad_curvature_rho2_049` is the benchmark's flux
compression. The released model also emits a smoothed variant whose name is
nearly identical and which measures something else, so it is absent here
deliberately.
"""


class ConstellarationUnits:
    """Presents a released surrogate under this package's metric convention."""

    def __init__(
        self,
        inner: surrogate_mod.StandardizedMetrics,
        conversions: Sequence[MetricConversion] = RELEASED_SURROGATE_CONVERSIONS,
        n_field_periods: int = feasibility.N_FIELD_PERIODS,
    ) -> None:
        missing = [
            c.released_name
            for c in conversions
            if c.released_name not in inner.metric_names
        ]
        if missing:
            raise KeyError(
                f"the surrogate does not emit {missing}; it emits "
                f"{list(inner.metric_names)}"
            )
        self.inner = inner
        self.standardization = inner.standardization
        self.metric_names = tuple(c.name for c in conversions)
        self._columns = [inner.metric_names.index(c.released_name) for c in conversions]
        self._scales = [
            float(n_field_periods) ** c.scale_by_field_periods for c in conversions
        ]

    def __call__(self, designs: torch.Tensor) -> torch.Tensor:
        released = self.inner(designs)
        return released[:, self._columns] * released.new_tensor(self._scales)

    def predict_boundaries(self, raw: NDArray) -> torch.Tensor:
        """Metrics for raw Fourier coefficients, encoded the way training was."""
        return self(
            torch.as_tensor(self.standardization.encode(raw), dtype=torch.float32)
        )


def artifact_uri(name: str) -> str:
    """Resolve a published artifact name against the configured base."""
    base = os.environ.get(ARTIFACT_BASE_ENV, DEFAULT_ARTIFACT_BASE)
    return f"{base.rstrip('/')}/{name}"


def open_artifact(name: str, mode: str = "rb") -> IO[Any]:
    """Open a published artifact, whichever backend its base names."""
    filesystem, path = fsspec.core.url_to_fs(artifact_uri(name))
    return filesystem.open(path, mode)


def load_state_dict(name: str, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint from the configured artifact store."""
    with open_artifact(name) as handle:
        return torch.load(handle, map_location=map_location, weights_only=False)


def load_surrogate(
    name: str, map_location: str = "cpu"
) -> surrogate_mod.StandardizedMetrics:
    """Rebuild the released surrogate from a checkpoint, frozen and ready to use.

    The checkpoint carries its own metric normalization and its own feature
    standardization, so neither can be paired with the wrong weights. The feature
    statistics matter as much as the weights: this surrogate was fitted on a superset of
    the published corpus whose per-column statistics differ from it substantially, so
    encoding boundaries against the corpus instead would quietly produce wrong metrics.
    """
    checkpoint = load_state_dict(name, map_location=map_location)
    required = (
        "state_dict",
        "metric_names",
        "metrics_mean",
        "metrics_std",
        "feature_mean",
        "feature_std",
        "feature_kept",
        "feature_alpha",
    )
    for key in required:
        if key not in checkpoint:
            raise KeyError(
                f"surrogate checkpoint is missing {key!r}; it carries "
                f"{sorted(checkpoint)}"
            )

    metric_names = tuple(checkpoint["metric_names"])
    mean = torch.as_tensor(checkpoint["metrics_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["metrics_std"], dtype=torch.float32)
    standardization = features.Standardization(
        alpha=float(checkpoint["feature_alpha"]),
        kept=np.asarray(checkpoint["feature_kept"], dtype=bool),
        mean=np.asarray(checkpoint["feature_mean"], dtype=np.float64),
        std=np.asarray(checkpoint["feature_std"], dtype=np.float64),
    )
    model = surrogate_mod.SurrogateModel(
        x_dim=int(checkpoint.get("x_dim", int(standardization.kept.sum()))),
        n_metrics=len(metric_names),
        hidden_dims=list(checkpoint.get("hidden_dims", [512, 512, 512])),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return surrogate_mod.StandardizedMetrics(
        model, mean, std, metric_names, standardization
    ).freeze()


def released_surrogate(
    name: str = "surrogate.pt", map_location: str = "cpu"
) -> ConstellarationUnits:
    """The released surrogate, translated into this package's metric convention."""
    return ConstellarationUnits(load_surrogate(name, map_location=map_location))
