"""Sampling-based search through the latent.

Rather than optimize a latent code, draw many from the prior, let the
conditioning settle on each, and keep the best of what comes out. The surrogate
ranks candidates for free, so the simulator is called only on a shortlist.

Nothing in the loop touches the simulator. Ten thousand codes through thirty
iterations is three hundred thousand decoder and surrogate passes, which costs
less than one simulator call; the expense is entirely in validating the
shortlist. This is the search the method enables, and it only works because the
latent interior is feasible: a model whose interior is not will spend the whole
budget decoding designs that fail.

What comes back is a set of distinct designs rather than a single point, which
is the property separating this from an optimizer that has converged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from . import artifacts, models
from . import feasibility as feasibility_mod
from . import inference as inference_mod


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Budget and initialization for one sampling run."""

    n_samples: int = 10_000
    n_iterations: int = 30
    top_k: int = 50

    objective: str = artifacts.LOG10_QI
    """Metric the candidates are ranked by, ascending, so lower is better."""

    objective_low: float = -3.5
    objective_high: float = -1.5
    """Range the objective's starting request is drawn from.

    Asking for values across the good tail of the distribution rather than for the
    corpus mean is what points the search somewhere worth looking. The request is only a
    starting point; the self-consistent iteration is free to settle elsewhere.
    """

    min_separation: float | None = None
    """Deduplication radius in design space, or `None` to keep every candidate.

    An optimizer that has converged proposes the same design repeatedly. Setting this
    makes the number of designs returned reflect how many are distinct.
    """


@dataclasses.dataclass(frozen=True)
class Proposal:
    """Everything a run produced, not only what it selected."""

    designs: torch.Tensor
    metrics: torch.Tensor
    feasible: torch.Tensor
    """Mask over the batch, by the surrogate's account rather than the simulator's."""

    ranking: torch.Tensor
    """Indices into `designs`, best first, after filtering and deduplication."""

    def selected(self) -> torch.Tensor:
        """The shortlist, in the order it should be sent to the simulator."""
        return self.designs[self.ranking]

    def selected_metrics(self) -> torch.Tensor:
        return self.metrics[self.ranking]


def propose(
    model: models.CVAE,
    predict_metrics: inference_mod.MetricPredictorWithNames,
    scaler: inference_mod.ConditionScaler,
    condition_columns: Sequence[str],
    constraints: feasibility_mod.ConstraintSet,
    config: SamplingConfig | None = None,
    generator: torch.Generator | None = None,
) -> Proposal:
    """Draw from the prior, settle the conditioning, rank what comes out."""
    config = config or SamplingConfig()
    _check_objective(predict_metrics, condition_columns, config)

    z = torch.randn(config.n_samples, model.latent_dim, generator=generator)
    y = initial_conditioning(scaler, condition_columns, config, generator)
    y = inference_mod.self_consistent_conditioning(
        model,
        z,
        y,
        predict_metrics,
        scaler,
        condition_columns,
        constraints=constraints,
        n_iterations=config.n_iterations,
    )

    with torch.no_grad():
        designs = model.decode(z, y)
        metrics = predict_metrics(designs)

    objective = metrics[:, predict_metrics.metric_names.index(config.objective)]
    feasible = constraints.is_feasible(metrics)
    return Proposal(
        designs=designs,
        metrics=metrics,
        feasible=feasible,
        ranking=_rank(designs, objective, feasible, config),
    )


def initial_conditioning(
    scaler: inference_mod.ConditionScaler,
    condition_columns: Sequence[str],
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Standardized conditioning to start the iteration from.

    Every component starts at the corpus mean except the objective, which is drawn
    across the range the config asks for.
    """
    if len(scaler.mean) != len(condition_columns):
        raise ValueError(
            f"the scaler covers {len(scaler.mean)} components but there are "
            f"{len(condition_columns)} conditioning columns; it has to be fitted "
            f"on the same conditioning matrix the decoder consumes"
        )
    mean = torch.as_tensor(scaler.mean, dtype=torch.get_default_dtype())
    raw = mean.reshape(1, -1).repeat(config.n_samples, 1)
    position = list(condition_columns).index(config.objective)
    spread = torch.rand(config.n_samples, generator=generator, dtype=raw.dtype)
    raw[:, position] = config.objective_low + spread * (
        config.objective_high - config.objective_low
    )
    return scaler.encode(raw)


def distinct_count(
    designs: torch.Tensor, indices: torch.Tensor, min_separation: float
) -> int:
    """How many of the indexed designs are distinct at the given radius.

    Reporting a shortlist of fifty means little if the entries coincide, so this is the
    honest count to compare search methods on.
    """
    return int(_deduplicate(designs, indices, min_separation, indices.numel()).numel())


def _check_objective(
    predict_metrics: inference_mod.MetricPredictorWithNames,
    condition_columns: Sequence[str],
    config: SamplingConfig,
) -> None:
    if config.objective not in predict_metrics.metric_names:
        raise KeyError(
            f"objective {config.objective!r} is not predicted; the predictor "
            f"reports {list(predict_metrics.metric_names)}"
        )
    # The objective is requested through the conditioning vector, so a decoder
    # not conditioned on it would ignore the initialization entirely.
    if config.objective not in condition_columns:
        raise KeyError(
            f"objective {config.objective!r} is not among the conditioning "
            f"columns {list(condition_columns)}, so it cannot be asked for"
        )


def _rank(
    designs: torch.Tensor,
    objective: torch.Tensor,
    feasible: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    # Falling back to the full batch when nothing passes keeps a failed run
    # inspectable, rather than returning an empty shortlist with no account of
    # what was rejected.
    pool = torch.nonzero(feasible, as_tuple=False).flatten()
    if pool.numel() == 0:
        pool = torch.arange(designs.shape[0])
    order = pool[torch.argsort(objective[pool])]
    if config.min_separation is None:
        return order[: config.top_k]
    return _deduplicate(designs, order, config.min_separation, config.top_k)


def _deduplicate(
    designs: torch.Tensor, order: torch.Tensor, min_separation: float, limit: int
) -> torch.Tensor:
    kept: list[int] = []
    for index in order.tolist():
        if kept:
            distance = torch.linalg.vector_norm(
                designs[kept] - designs[index], dim=-1
            ).min()
            if float(distance) < min_separation:
                continue
        kept.append(index)
        if len(kept) == limit:
            break
    return torch.tensor(kept, dtype=torch.long, device=designs.device)
