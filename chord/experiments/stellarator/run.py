"""Reproduce the cross-basin interpolation result.

Trains the requested variants on the published corpus, walks latent chords between held-
out designs from different basins under both conditioning strategies, and reports
feasibility along the path. Needs the stellarator extra for the simulator check.

python -m chord.experiments.stellarator.run --variants cvae chord python -m
chord.experiments.stellarator.run --no-validate --n-epochs 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from ... import artifacts
from ... import corpus as corpus_mod
from .. import variants as variants_mod
from . import interpolation, training

_LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    arguments = _parse(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    torch.manual_seed(arguments.seed)

    predictor = artifacts.released_surrogate()
    corpus = corpus_mod.load_corpus()
    rows_a, rows_b = interpolation.cross_basin_pairs(
        corpus, n_pairs=arguments.n_pairs, seed=arguments.seed
    )
    _LOG.info("corpus %d rows, %d cross-basin pairs", len(corpus), len(rows_a))

    rows = []
    for name in arguments.variants:
        _LOG.info("training %s", name)
        trained = training.train(
            corpus,
            variants_mod.VARIANTS[name],
            predictor,
            config=training.training_config(
                variants_mod.VARIANTS[name], arguments.n_epochs
            ),
            seed=arguments.seed,
        )
        for conditioning in arguments.conditioning:
            result = interpolation.sweep(
                trained,
                predictor,
                corpus,
                rows_a,
                rows_b,
                conditioning=conditioning,
                n_alpha=arguments.n_alpha,
            )
            row = {
                "variant": name,
                "conditioning": conditioning,
                "seed": arguments.seed,
                "surrogate_interior": result.interior(),
                "surrogate_full_path": result.full_path(),
                "surrogate_fully_feasible": result.fully_feasible(),
            }
            if arguments.validate:
                checked = interpolation.validate(
                    result, stride=arguments.stride, max_pairs=arguments.validate_pairs
                )
                row.update(
                    simulator_interior=checked.interior(),
                    simulator_convergence=checked.convergence_rate(),
                    benchmark_feasible=float(checked.benchmark_feasible.mean()),
                )
            rows.append(row)
            _LOG.info("%s / %s: %s", name, conditioning, _brief(row))

    _print_table(rows, validated=arguments.validate)
    if arguments.results_path:
        with open(arguments.results_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=float) + "\n")
        _LOG.info("wrote %s", arguments.results_path)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["cvae", "chord"],
        choices=list(variants_mod.VARIANTS),
    )
    parser.add_argument(
        "--conditioning",
        nargs="+",
        default=list(interpolation.CONDITIONING),
        choices=list(interpolation.CONDITIONING),
    )
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--n-alpha", type=int, default=21)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--validate-pairs", type=int, default=8)
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="skip the simulator check, leaving only the surrogate's verdict",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", default=None)
    return parser.parse_args(argv)


def _brief(row: dict) -> str:
    parts = [f"surrogate interior {row['surrogate_interior']:.3f}"]
    if "simulator_interior" in row:
        parts.append(f"simulator interior {row['simulator_interior']:.3f}")
    return ", ".join(parts)


def _print_table(rows: list[dict], validated: bool) -> None:
    header = f"{'variant':<10}{'conditioning':<17}{'interior':>10}{'full':>8}{'all':>8}"
    if validated:
        header += f"{'sim interior':>14}{'converged':>11}{'benchmark':>11}"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['variant']:<10}{row['conditioning']:<17}"
            f"{row['surrogate_interior']:>10.3f}{row['surrogate_full_path']:>8.3f}"
            f"{row['surrogate_fully_feasible']:>8.3f}"
        )
        if validated:
            line += (
                f"{row['simulator_interior']:>14.3f}"
                f"{row['simulator_convergence']:>11.3f}"
                f"{row['benchmark_feasible']:>11.3f}"
            )
        print(line)
    print()
    print(
        "Feasibility along latent chords between held-out designs from "
        "different basins. Higher is better."
    )
    print(
        f"'interior' covers alpha in {interpolation.INTERIOR}, away from the endpoints."
    )


if __name__ == "__main__":
    main()
