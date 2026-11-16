"""Decode the paper's archived front models without running the simulator.

Set `CHORD_ARTIFACT_BASE` to the directory containing `frontfill/g1`, etc.
Outputs are candidate boundaries, not feasibility evaluations. Validate with the
pinned ConStellaration version before reporting new physical results.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from chord.experiments.frontfill import gaps, prepared_utils


def main(argv: list[str] | None = None) -> None:
    """Decode a saved model and write boundaries with endpoint provenance.

    Args:
        argv: Command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gap", choices=[g.name for g in gaps.all_gaps()], required=True
    )
    parser.add_argument("--model", choices=["chord", "cvae"], default="chord")
    parser.add_argument("--n-alpha", type=int, default=19)
    parser.add_argument("--per-alpha", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--no-clamp-qi", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    base = f"frontfill/{args.gap}"
    dataset = prepared_utils.load_dataset(f"{base}/dataset.npz")
    trained = prepared_utils.load_model(
        dataset, f"{base}/{args.model}.pt", f"{base}/config.json"
    )
    predictor = (
        prepared_utils.load_surrogate(
            f"{base}/surrogate.pt", f"{base}/surrogate_metadata.npz"
        )
        if args.iterations
        else None
    )
    result = prepared_utils.decode_sweep(
        trained,
        dataset,
        gaps.gap_by_name(args.gap),
        tuple(float(a) for a in np.linspace(0.05, 0.95, args.n_alpha)),
        [args.per_alpha] * args.n_alpha,
        seed=args.seed,
        predictor=predictor,
        iterations=args.iterations,
        margin=args.margin,
        clamp_qi=not args.no_clamp_qi,
    )
    metadata = vars(args) | {"out": str(args.out)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        np.savez_compressed(
            handle,
            boundary_coefficients=result.boundaries.numpy(),
            alpha=result.alpha.numpy(),
            low_train_index=result.low_train_index.numpy(),
            high_train_index=result.high_train_index.numpy(),
            template_boundary_json=dataset.template_boundary_json,
            config_json=json.dumps(metadata),
        )
    print(f"Wrote {len(result.boundaries)} unevaluated boundaries to {args.out}")


if __name__ == "__main__":
    main()
