"""Reproduce the constructed-benchmark comparison.

Generates one corpus, trains every variant on it, and reports feasibility along latent
chords between held-out designs from different basins.

python -m chord.experiments.toy.run --problem cshape python -m chord.experiments.toy.run
--problem shells --variants chord cvae
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from . import cshape, datasets, interpolation, shells, training
from . import problem as problem_mod

_LOG = logging.getLogger(__name__)

LATENT_DIM = {"cshape": 2, "shells": 4}


def main(argv: list[str] | None = None) -> None:
    arguments = _parse(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    torch.manual_seed(arguments.seed)

    problem = _problem(arguments)
    dataset = datasets.generate(problem, seed=arguments.seed)

    rows = []
    for name in arguments.variants:
        _LOG.info("training %s on %s", name, problem.name)
        trained = training.train(
            problem,
            dataset,
            training.VARIANTS[name],
            n_epochs=arguments.n_epochs,
            latent_dim=arguments.latent_dim or LATENT_DIM[problem.name],
            seed=arguments.seed,
        )
        results = interpolation.across_basins(
            problem, trained, dataset, n_pairs=arguments.n_pairs, seed=arguments.seed
        )
        rows.append(
            {
                "problem": problem.name,
                "variant": name,
                "seed": arguments.seed,
                **interpolation.summarize(results),
                "by_pair": {
                    f"{a}-{b}": chord.middle() for (a, b), chord in results.items()
                },
            }
        )
        _LOG.info("%s: middle %.3f", name, rows[-1]["middle"])

    _print_table(rows)
    if arguments.results_path:
        with open(arguments.results_path, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=float) + "\n")
        _LOG.info("wrote %s", arguments.results_path)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="cshape", choices=sorted(LATENT_DIM))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(training.VARIANTS),
        choices=list(training.VARIANTS),
    )
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument(
        "--n-dim", type=int, default=2, help="C-shape only: added nuisance dimensions"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-path", default=None)
    return parser.parse_args(argv)


def _problem(arguments: argparse.Namespace) -> problem_mod.Problem:
    if arguments.problem == "cshape":
        return cshape.CShape(n_dim=arguments.n_dim)
    return shells.Shells()


def _print_table(rows: list[dict]) -> None:
    pairs = sorted({key for row in rows for key in row["by_pair"]})
    header = f"{'variant':<12}{'middle':>9}{'full':>9}{'all-feasible':>14}"
    print()
    print(header + "".join(f"{pair:>9}" for pair in pairs))
    print("-" * (len(header) + 9 * len(pairs)))
    for row in rows:
        line = (
            f"{row['variant']:<12}{row['middle']:>9.3f}{row['full_path']:>9.3f}"
            f"{row['fully_feasible']:>14.3f}"
        )
        print(line + "".join(f"{row['by_pair'][p]:>9.3f}" for p in pairs))
    pairs_evaluated = rows[0]["n_pairs"]
    print()
    print(
        f"Feasibility along latent chords between held-out designs, "
        f"{pairs_evaluated} pairs."
    )
    print(f"'middle' covers alpha in {interpolation.MIDDLE}, away from the endpoints.")


if __name__ == "__main__":
    main()
