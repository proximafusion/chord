"""The four variants the paper compares.

A variant is two independent switches, not four implementations: whether the decoder is
conditioned on a metric vector, and whether the chord regularizer is applied during
training. Both the constructed benchmarks and the stellarator experiments compare the
same four, so the table lives here rather than once per experiment.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Variant:
    """One cell of the comparison."""

    conditional: bool
    chord_lambda: float


VARIANTS: dict[str, Variant] = {
    "vae": Variant(conditional=False, chord_lambda=0.0),
    "chord_unc": Variant(conditional=False, chord_lambda=1.0),
    "cvae": Variant(conditional=True, chord_lambda=0.0),
    "chord": Variant(conditional=True, chord_lambda=1.0),
}
"""Named as the paper's tables name them."""
