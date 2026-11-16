"""Choose what a per-gap model is allowed to learn from.

Each gap is filled by a model that has only ever seen its own two anchor campaigns.
That is what makes the separation between gaps structural: a model cannot leak across a
gap it was never shown.

Enforcing it has one trap, and it is the whole reason this module exists. Augmented rows
are rescaled campaign boundaries, so a row's aspect ratio says nothing about where it
came from. A boundary from a held-out campaign rescaled onto an anchor's aspect ratio is
still held-out information, and no window on aspect ratio catches it. Selection is
therefore on provenance: which design a row was rescaled *from*, which
`chord.experiments.frontfill.augment` records for exactly this purpose.

The converse is deliberate. An anchor boundary rescaled onto a held-out aspect ratio is
kept. It needed no new campaign, and it is precisely the rescale-and-polish baseline the
generative model has to beat.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ... import corpus as corpus_mod
from . import augment, pareto
from . import gaps as gaps_mod

_LOG = logging.getLogger(__name__)

ASPECT_RATIO = augment.ASPECT_RATIO

CONDITION_COLUMNS: tuple[str, ...] = (*corpus_mod.CONDITION_COLUMNS, ASPECT_RATIO)
"""Conditioning for a front-filling model.

Aspect ratio joins the usual vector because filling a gap means asking for a design at a
particular aspect ratio. Without it the decoder has no way to be told which part of the
gap to produce.
"""

SURROGATE_METRICS: tuple[str, ...] = (
    *CONDITION_COLUMNS,
    pareto.GRADIENT_SCALE_LENGTH,
)
"""What a per-gap surrogate has to predict.

Everything the decoder is conditioned on, plus the magnetic gradient scale length. That
last one is never conditioned on but it is the other axis of the front, so a surrogate
without it can propose designs and not say whether they improve anything.
"""


@dataclasses.dataclass(frozen=True)
class GapDataset:
    """Rows one gap's model may train on, and what they cover."""

    gap: gaps_mod.Gap
    frame: pd.DataFrame
    in_gap: NDArray
    """Rows whose achieved aspect ratio landed strictly inside the interval.

    These are what give the model signal where no campaign ran. They come from anchor
    boundaries rescaled inward, so they carry no held-out information.
    """

    def __len__(self) -> int:
        return len(self.frame)

    def boundaries(self) -> NDArray:
        return np.stack(self.frame[corpus_mod.BOUNDARY_COLUMN].tolist())

    def conditions(self, columns: tuple[str, ...] = CONDITION_COLUMNS) -> NDArray:
        missing = [c for c in columns if c not in self.frame.columns]
        if missing:
            raise KeyError(
                f"the gap dataset is missing conditioning columns {missing}; it "
                f"carries {sorted(self.frame.columns)}"
            )
        return self.frame[list(columns)].to_numpy(dtype=np.float64)

    def describe(self) -> str:
        return (
            f"{self.gap.name}: {len(self)} rows, {int(self.in_gap.sum())} inside "
            f"[{self.gap.low:.2f}, {self.gap.high:.2f}]"
        )


def from_anchors(
    source_aspect_ratio: NDArray,
    gap: gaps_mod.Gap,
    tolerance: float = gaps_mod.MATCH_TOLERANCE,
) -> NDArray:
    """Mask over rows whose source design sits at one of the gap's anchors.

    Takes the aspect ratio of the design each row was rescaled *from*, not the one it
    achieved. Passing the achieved value would admit held-out campaigns that happen to
    have been rescaled nearby, which is the leak this guards against.
    """
    source = np.asarray(source_aspect_ratio, dtype=float)
    return np.asarray(
        np.minimum(np.abs(source - gap.low), np.abs(source - gap.high)) <= tolerance
    )


def select(
    labelled: pd.DataFrame,
    source_aspect_ratio: NDArray,
    gap: gaps_mod.Gap,
    tolerance: float = gaps_mod.MATCH_TOLERANCE,
) -> GapDataset:
    """Rows a model for this gap may train on.

    `labelled` is augmentation output, carrying a source row per design and the metrics
    the simulator returned for it. `source_aspect_ratio` is indexed by the source corpus
    row, so a row's provenance resolves through it.
    """
    for column in (augment.SOURCE_ROW, augment.CONVERGED, ASPECT_RATIO):
        if column not in labelled.columns:
            raise KeyError(
                f"the labelled table is missing {column!r}; it carries "
                f"{sorted(labelled.columns)}"
            )

    sources = labelled[augment.SOURCE_ROW].to_numpy()
    source = np.asarray(source_aspect_ratio, dtype=float)
    if sources.size and sources.max() >= len(source):
        raise IndexError(
            f"a row names source {sources.max()} but only {len(source)} source "
            f"aspect ratios were given; they must index the corpus augmented from"
        )

    keep = from_anchors(source[sources], gap, tolerance)
    # A design the simulator could not solve has no metrics to learn from.
    keep &= labelled[augment.CONVERGED].to_numpy(dtype=bool)
    frame = labelled[keep].reset_index(drop=True)

    achieved = frame[ASPECT_RATIO].to_numpy(dtype=float)
    in_gap = np.asarray(
        (achieved > gap.low + tolerance) & (achieved < gap.high - tolerance)
    )
    dataset = GapDataset(gap=gap, frame=frame, in_gap=in_gap)
    _LOG.info("%s", dataset.describe())
    return dataset


def as_corpus(
    dataset: GapDataset,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> corpus_mod.Corpus:
    """Shape a gap's rows as a corpus, so the usual training code accepts them.

    The basin label records whether a row landed inside the gap. Nothing in training
    reads it, but it keeps visible which rows carry signal where no campaign ran.
    """
    order = np.random.default_rng(seed).permutation(len(dataset))
    n_validation = int(round(validation_fraction * len(dataset)))
    split = np.empty(len(dataset), dtype=object)
    split[order[:n_validation]] = "val"
    split[order[n_validation:]] = "train"

    frame = dataset.frame.copy()
    frame[corpus_mod.SPLIT_COLUMN] = split
    frame[corpus_mod.BASIN_COLUMN] = np.where(dataset.in_gap, "in_gap", "at_anchor")
    keep = [
        corpus_mod.BOUNDARY_COLUMN,
        corpus_mod.SPLIT_COLUMN,
        corpus_mod.BASIN_COLUMN,
        *CONDITION_COLUMNS,
    ]
    return corpus_mod.from_frame(frame[keep])


def leaked(
    labelled: pd.DataFrame,
    source_aspect_ratio: NDArray,
    gap: gaps_mod.Gap,
    tolerance: float = gaps_mod.MATCH_TOLERANCE,
) -> NDArray:
    """Rows an aspect-ratio filter would have admitted but provenance rejects.

    Not used for selection. It exists so the size of the leak a naive filter would let
    through can be reported rather than assumed.
    """
    sources = labelled[augment.SOURCE_ROW].to_numpy()
    source = np.asarray(source_aspect_ratio, dtype=float)
    achieved = labelled[ASPECT_RATIO].to_numpy(dtype=float)
    looks_like_anchor = np.asarray(
        np.minimum(np.abs(achieved - gap.low), np.abs(achieved - gap.high)) <= tolerance
    )
    return looks_like_anchor & ~from_anchors(source[sources], gap, tolerance)
