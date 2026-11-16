"""Fill the gaps between anchor campaigns with generated designs.

Two commands, because the expensive step should be taken deliberately and once.

`build` expands an explicitly supplied anchor corpus across one gap and sends every
rescaled boundary through the simulator. This costs roughly a minute of CPU at screening
fidelity, and it writes a table the other command reads. It is separate so that
re-training or re-generating never silently re-runs it.

`fill` trains a surrogate and a decoder on that table, generates designs at aspect
ratios no campaign ran, and scores what they add to the front.

    python -m chord.experiments.frontfill.run build --source-corpus anchors.parquet
        --gap g3 --out g3.parquet
    python -m chord.experiments.frontfill.run fill --gap g3 --dataset g3.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
from chord import corpus as corpus_mod
from chord import surrogate as surrogate_mod
from chord.experiments import variants as variants_mod
from chord.experiments.frontfill import (
    augment,
    campaign,
    gaps,
    generate,
    pareto,
    selection,
)
from chord.experiments.stellarator import training as training_mod

_LOG = logging.getLogger(__name__)

_SOURCE_ASPECT_RATIO = "source_campaign_aspect_ratio"
"""Target aspect ratio of the source campaign, stamped at build time.

Carried in the table so provenance survives the trip through a file, which is the whole
basis of per-gap separation.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("build", help="expand and label; the expensive step")
    builder.add_argument("--gap", default=gaps.all_gaps()[2].name)
    builder.add_argument(
        "--source-corpus",
        required=True,
        help="anchor corpus with source_campaign_aspect_ratio provenance",
    )
    builder.add_argument("--out", required=True)
    builder.add_argument("--per-source", type=int, default=5)
    builder.add_argument("--max-sources", type=int, default=2000)
    builder.add_argument("--n-workers", type=int, default=None)
    builder.add_argument("--seed", type=int, default=0)

    filler = commands.add_parser("fill", help="train, generate and score")
    filler.add_argument("--gap", default=gaps.all_gaps()[2].name)
    filler.add_argument("--dataset", required=True)
    filler.add_argument(
        "--variant", default="chord", choices=list(variants_mod.VARIANTS)
    )
    filler.add_argument("--n-epochs", type=int, default=300)
    filler.add_argument("--surrogate-epochs", type=int, default=100)
    filler.add_argument("--n-bins", type=int, default=10)
    filler.add_argument("--per-bin", type=int, default=30)
    filler.add_argument("--results-path", default=None)
    filler.add_argument("--seed", type=int, default=0)

    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    if arguments.command == "build":
        _build(arguments)
    else:
        _fill(arguments)


def _build(arguments: argparse.Namespace) -> None:
    gap = gaps.gap_by_name(arguments.gap)
    source = corpus_mod.load_corpus(arguments.source_corpus)
    if _SOURCE_ASPECT_RATIO not in source.metrics:
        raise ValueError(
            f"source corpus must carry {_SOURCE_ASPECT_RATIO}; achieved aspect ratio "
            "cannot identify a source campaign. The fixed-AR corpus is not an "
            "anchor-campaign training pool."
        )
    targets = source.metrics[_SOURCE_ASPECT_RATIO].to_numpy(dtype=float)
    for anchor in (gap.low, gap.high):
        if not np.any(np.abs(targets - anchor) <= gaps.MATCH_TOLERANCE):
            raise ValueError(
                f"source corpus has no campaign at anchor {anchor} for {gap.name}; "
                "no simulations have been started"
            )
    source_rows = np.flatnonzero(selection.from_anchors(targets, gap))
    source = corpus_mod.Corpus(
        boundaries=source.boundaries[source_rows],
        metrics=source.metrics.iloc[source_rows].reset_index(drop=True),
        split=source.split[source_rows],
        basin=source.basin[source_rows],
    )
    _LOG.info("expanding %d corpus rows across %s", len(source), gap.describe())

    augmented = augment.augment_for_gap(
        source,
        gap,
        per_source=arguments.per_source,
        max_sources=arguments.max_sources,
        rng=np.random.default_rng(arguments.seed),
    )
    _LOG.info("labelling %d boundaries; this is the expensive step", len(augmented))
    labelled = augment.label(augmented, n_workers=arguments.n_workers)
    labelled[augment.SOURCE_ROW] = source_rows[augmented.source_row]
    labelled[_SOURCE_ASPECT_RATIO] = targets[source_rows[augmented.source_row]]
    labelled.to_parquet(arguments.out, index=False)
    _LOG.info("wrote %d rows to %s", len(labelled), arguments.out)


def _fill(arguments: argparse.Namespace) -> None:
    gap = gaps.gap_by_name(arguments.gap)
    labelled = pd.read_parquet(arguments.dataset)
    if _SOURCE_ASPECT_RATIO not in labelled.columns:
        raise KeyError(
            f"the dataset has no {_SOURCE_ASPECT_RATIO!r} column, so a row's "
            f"provenance cannot be resolved; rebuild it with the build command"
        )

    # Provenance resolves through the source corpus row, so the lookup is indexed
    # by it rather than positionally.
    source_aspect_ratio = np.zeros(int(labelled[augment.SOURCE_ROW].max()) + 1)
    source_aspect_ratio[labelled[augment.SOURCE_ROW].to_numpy()] = labelled[
        _SOURCE_ASPECT_RATIO
    ].to_numpy()

    dataset = selection.select(labelled, source_aspect_ratio, gap)
    leaks = selection.leaked(labelled, source_aspect_ratio, gap)
    _LOG.info(
        "an aspect-ratio filter would have admitted %d rows this one rejects",
        int(leaks.sum()),
    )
    if not len(dataset):
        raise ValueError(
            f"no rows survive selection for {gap.name}; the dataset may have been "
            f"built for a different gap"
        )

    _LOG.info("fitting a surrogate on %d rows", len(dataset))
    predictor, _ = surrogate_mod.fit(
        dataset.boundaries(),
        dataset.conditions(selection.SURROGATE_METRICS),
        selection.SURROGATE_METRICS,
        surrogate_mod.SurrogateConfig(n_epochs=arguments.surrogate_epochs),
        seed=arguments.seed,
    )

    gap_corpus = selection.as_corpus(dataset, seed=arguments.seed)
    _LOG.info("training %s on %d rows", arguments.variant, len(gap_corpus))
    trained = training_mod.train(
        gap_corpus,
        variants_mod.VARIANTS[arguments.variant],
        predictor,
        config=training_mod.training_config(
            variants_mod.VARIANTS[arguments.variant], arguments.n_epochs
        ),
        condition_columns=selection.CONDITION_COLUMNS,
        seed=arguments.seed,
    )

    generated = generate.fill(
        trained,
        predictor,
        gap,
        generate.GenerationConfig(n_bins=arguments.n_bins, per_bin=arguments.per_bin),
        seed=arguments.seed,
    )
    _report(gap, dataset, generated, arguments)


def _report(
    gap: gaps.Gap,
    dataset: selection.GapDataset,
    generated: generate.Generated,
    arguments: argparse.Namespace,
) -> None:
    feasible = generated.feasible
    ratio = generated.achieved_aspect_ratio()[feasible]
    length = generated.metrics[feasible][
        :, generated.metric_names.index(pareto.GRADIENT_SCALE_LENGTH)
    ]
    front = campaign.load()
    scored = campaign.score(front, ratio, length)

    row = {
        "gap": gap.name,
        "variant": arguments.variant,
        "training_rows": len(dataset),
        "training_rows_in_gap": int(dataset.in_gap.sum()),
        "generated": len(generated),
        "generated_feasible": int(feasible.sum()),
        **{
            f"{name}_hypervolume": values["hypervolume"]
            for name, values in scored.items()
            if "hypervolume" in values
        },
        "added_frontier_designs": scored.get("recovered", {}).get(
            "added_frontier_designs", float("nan")
        ),
    }

    print()
    for key, value in row.items():
        print(f"  {key:<32} {value}")
    print(
        "\nHypervolume by the surrogate's account. Validate with chord.evaluate "
        "before reporting."
    )
    if arguments.results_path:
        with open(arguments.results_path, "w") as handle:
            handle.write(json.dumps(row, default=float) + "\n")
        _LOG.info("wrote %s", arguments.results_path)


if __name__ == "__main__":
    main()
