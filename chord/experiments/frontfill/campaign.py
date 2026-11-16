"""The optimization-campaign front that front-filling is measured against.

The study comparison contains only anchor and held-out campaigns. The broader benchmark
comparison also includes excluded campaigns and can span a different aspect-ratio range.
Both are reported explicitly alongside the anchors and any generated designs.

The published table carries every campaign with the role it plays, so the anchor front
is a filter over one artifact rather than a second artifact that could disagree with it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ... import artifacts
from . import pareto

ROLE = "role"
ANCHOR = "anchor"
HELD_OUT = "held_out"
EXCLUDED = "excluded"
OUTSIDE = "outside"

ASPECT_RATIO = "aspect_ratio"
FEASIBLE = "soft_feasible"


def load(name: str = "campaign_front.parquet") -> pd.DataFrame:
    """Read the campaign table from the configured artifact store, or a direct path."""
    path = (
        name if "://" in name or os.path.exists(name) else artifacts.artifact_uri(name)
    )
    frame = pd.read_parquet(path)
    missing = [
        column
        for column in (ROLE, ASPECT_RATIO, FEASIBLE, pareto.GRADIENT_SCALE_LENGTH)
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"the campaign table is missing {missing}; it carries "
            f"{sorted(frame.columns)}"
        )
    return frame


class Front(NamedTuple):
    """The two axes a front is scored on, named so they cannot be swapped.

    Still a tuple, so existing unpacking at call sites reads the same, but the fields
    carry their meaning rather than relying on position.
    """

    aspect_ratio: NDArray
    gradient_scale_length: NDArray


def points(frame: pd.DataFrame, roles: Sequence[str] | None = None) -> Front:
    """Feasible designs of the given roles, as `(aspect ratio, scale length)`.

    Infeasible designs are dropped rather than scored, since a front made of designs
    that violate the constraints measures nothing.
    """
    selected = frame[frame[FEASIBLE]]
    if roles is not None:
        selected = selected[selected[ROLE].isin(list(roles))]
    return Front(
        aspect_ratio=selected[ASPECT_RATIO].to_numpy(dtype=float),
        gradient_scale_length=selected[pareto.GRADIENT_SCALE_LENGTH].to_numpy(
            dtype=float
        ),
    )


def score(
    frame: pd.DataFrame,
    generated_aspect_ratio: NDArray | None = None,
    generated_scale_length: NDArray | None = None,
) -> dict[str, dict[str, float]]:
    """Score the anchors, study campaigns, and broader benchmark separately.

    Args:
        frame: Campaign rows with provenance roles and benchmark feasibility flags.
        generated_aspect_ratio: Aspect ratios of already feasibility-filtered designs.
        generated_scale_length: Corresponding field-period-scaled gradient lengths.

    Returns:
        Front counts, hypervolumes, and largest adjacent aspect-ratio gaps.
        `study_campaign` contains anchors and held-out campaigns; `full_campaign`
        contains every feasible campaign row, including excluded campaigns.
        `recovered` uses the broader benchmark denominator and `recovered_study`
        uses the study denominator. Generated-front counts use provenance, not
        proximity to rounded campaign targets.
    """
    anchors = points(frame, [ANCHOR])
    fronts = {"anchors_only": anchors}

    if generated_aspect_ratio is not None and generated_scale_length is not None:
        fronts["anchors_plus_generated"] = Front(
            aspect_ratio=np.concatenate(
                [anchors.aspect_ratio, np.asarray(generated_aspect_ratio)]
            ),
            gradient_scale_length=np.concatenate(
                [anchors.gradient_scale_length, np.asarray(generated_scale_length)]
            ),
        )
    fronts["study_campaign"] = points(frame, [ANCHOR, HELD_OUT])
    fronts["full_campaign"] = points(frame)

    report: dict[str, dict[str, float]] = {}
    for name, front in fronts.items():
        mask = pareto.non_dominated(front.aspect_ratio, front.gradient_scale_length)
        ratios = np.unique(front.aspect_ratio[mask])
        report[name] = {
            "n_points": float(len(front.aspect_ratio)),
            "n_non_dominated": float(mask.sum()),
            "largest_adjacent_aspect_ratio_gap": (
                float(np.diff(ratios).max()) if len(ratios) > 1 else 0.0
            ),
            "hypervolume": pareto.hypervolume(
                front.aspect_ratio, front.gradient_scale_length
            ),
        }
        if name == "anchors_plus_generated":
            report[name]["n_generated_non_dominated"] = float(
                mask[len(anchors.aspect_ratio) :].sum()
            )
    if "anchors_plus_generated" in report:
        for key, reference in (
            ("recovered", "full_campaign"),
            ("recovered_study", "study_campaign"),
        ):
            recovered = pareto.recovered_fraction(
                report["anchors_only"]["hypervolume"],
                report["anchors_plus_generated"]["hypervolume"],
                report[reference]["hypervolume"],
            )
            report[key] = {
                "fraction_of_available_hypervolume": (
                    float("nan") if recovered is None else recovered
                ),
                "added_frontier_designs": (
                    report["anchors_plus_generated"]["n_non_dominated"]
                    - report["anchors_only"]["n_non_dominated"]
                ),
            }
    return report
