"""Which aspect ratios a model may learn from, and which it is tested on.

Building a Pareto front over aspect ratio by optimization costs one campaign per target
value. The claim this experiment tests is that a few campaigns plus a cheap aspect-
ratio-conditioned model per interval recovers the rest.

So the available campaigns are split. A handful are retained as anchors and are the only
data any model may see; the campaigns between them are held out and keep their fronts as
ground truth. Each interval is filled independently, by a model that has only ever seen
its own two endpoints.

Independence is the point rather than an implementation detail. Three or four intervals
working separately is a far stronger claim than one wide interval working, and a model
confined to its own two anchors cannot leak across a gap even by accident, so the
guarantee is structural rather than something a filter has to enforce.
"""

from __future__ import annotations

import dataclasses

ANCHORS: tuple[float, ...] = (8.00, 9.25, 10.04, 10.82, 12.00)
"""Retained campaigns, the only data a model is allowed to learn from."""

HELD_OUT: tuple[float, ...] = (8.46, 8.86, 9.64, 10.43, 11.21, 11.61)
"""Campaigns withheld from training.

Each keeps its front as ground truth.
"""

EXCLUDED: tuple[float, ...] = (7.29, 7.68, 10.00)
"""Campaigns left out of the study, for two different reasons.

The two low values sit below the feasible band and carry almost no feasible designs, so
there is no front at them to recover. The third is excluded for a more interesting
reason: it sits 0.04 away from an anchor, far closer than any real gap. That makes it
useless as a test point, since generating at 10.00 when 10.04 is a training anchor
measures nothing, and actively harmful as training data, since it is effectively a
second copy of that anchor and would let a model interpolate to within 0.04 of its own
endpoint and call it a success.
"""

MATCH_TOLERANCE = 0.02
"""How close a design's aspect ratio must be to count as belonging to an anchor.

Deliberately smaller than the 0.04 separating the two closest campaign values, so those
two can never be merged into one group.
"""


@dataclasses.dataclass(frozen=True)
class Gap:
    """One interval: two training anchors and the held-out points between them."""

    name: str
    low: float
    high: float
    test_points: tuple[float, ...]

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def anchors(self) -> tuple[float, float]:
        return (self.low, self.high)

    def contains(self, aspect_ratio: float) -> bool:
        """Whether a value falls strictly inside the interval, clear of both anchors."""
        return self.low + MATCH_TOLERANCE < aspect_ratio < self.high - MATCH_TOLERANCE

    def describe(self) -> str:
        return (
            f"{self.name}: [{self.low:.2f}, {self.high:.2f}] width {self.width:.2f}, "
            f"testing {list(self.test_points)}"
        )


def all_gaps() -> tuple[Gap, ...]:
    """Every interval between consecutive anchors, left to right."""
    return tuple(
        Gap(
            name=f"g{index}",
            low=low,
            high=high,
            test_points=tuple(
                point
                for point in HELD_OUT
                if low + MATCH_TOLERANCE < point < high - MATCH_TOLERANCE
            ),
        )
        for index, (low, high) in enumerate(zip(ANCHORS[:-1], ANCHORS[1:]), start=1)
    )


def gap_by_name(name: str) -> Gap:
    gaps = all_gaps()
    for gap in gaps:
        if gap.name == name:
            return gap
    raise ValueError(f"unknown gap {name!r}; expected one of {[g.name for g in gaps]}")
