"""Simulator evaluation, delegated to constellaration.

Everything here is a thin adapter. Metric definitions, VMEC settings and the
benchmark's own feasibility test come from constellaration rather than being
reimplemented, so our numbers and the published benchmark's cannot drift apart.
That was a real hazard: our pipeline once carried its own copy of these
definitions and the two had to be checked for parity by hand.

The simulator raises on a boundary it cannot solve. Scoring a batch of
generated designs needs failures reported rather than propagated, so `evaluate`
converts them into an `Evaluation` with `converged` false.

This module needs the `stellarator` extra. The rest of the package does not.
"""

# constellaration is an optional runtime dependency and is deliberately absent
# from this repository's resolve, so neither the type checker nor the build
# system can see it here. It resolves normally in the published package.
# pyright: reportMissingImports=false

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib
import logging
import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from . import feasibility, features

if TYPE_CHECKING:
    from constellaration import (
        forward_model as forward_model_module,  # pants: no-infer-dep
    )
    from constellaration.geometry import surface_rz_fourier  # pants: no-infer-dep

_LOG = logging.getLogger(__name__)

_WORKER_DIED = "the solver killed its worker process"

_EXTRA_REQUIRED = (
    "chord.evaluate requires the constellaration package. Install the optional "
    "extra with: pip install 'chord[stellarator]'"
)


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """Outcome of running the simulator on one boundary."""

    converged: bool
    metrics: Any | None = None
    effective_aspect_ratio: float | None = None
    """Derived from the equilibrium while it is still in hand.

    The equilibrium itself is not kept. It is a whole solved wout, and a batch runs in
    worker processes, so retaining it would mean pickling one back per design.
    """

    failure: str | None = None


def very_low_fidelity_settings() -> forward_model_module.ConstellarationSettings:
    """Screening settings, matching the fidelity the design corpus was labelled at.

    Roughly an order of magnitude cheaper than the benchmark's scoring fidelity,
    at the cost of a small shift in metrics near constraint boundaries. Use
    `high_fidelity_settings` for anything being reported.
    """
    return _forward_model().ConstellarationSettings(
        vmec_preset_settings=_vmec_settings().VmecPresetSettings(
            fidelity="very_low_fidelity"
        ),
    )


def high_fidelity_settings() -> forward_model_module.ConstellarationSettings:
    """Scoring settings used by the published benchmark."""
    return _forward_model().ConstellarationSettings.default_high_fidelity()


def boundary_from_vector(
    x: NDArray, n_field_periods: int = feasibility.N_FIELD_PERIODS
) -> surface_rz_fourier.SurfaceRZFourier:
    """Build a boundary the simulator accepts from a raw coefficient vector.

    This is the only way a design the model produced reaches the simulator, so the
    layout it assumes has to be the one `chord.features` defines rather than a
    separately maintained copy of it.
    """
    vector = np.asarray(x, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(
            f"expected a single coefficient vector, got shape {tuple(vector.shape)}"
        )
    # Stellarator symmetry forces some coefficients to zero, and the simulator
    # rejects a surface that puts anything there. Checking against the package's
    # own mask says which coefficients rather than leaving it to a validator
    # several layers down.
    forced_zero = features.structurally_zero_mask()
    if np.any(vector[forced_zero] != 0.0):
        offending = np.flatnonzero(forced_zero & (vector != 0.0))
        raise ValueError(
            f"coefficients {offending.tolist()} are forced to zero by stellarator "
            f"symmetry but are not; see chord.features.structurally_zero_mask"
        )
    r_cos, z_sin = features.vector_to_coefficients(vector)
    return _surface_rz_fourier().SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=True,
    )


def vector_from_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> NDArray:
    """Inverse of `boundary_from_vector`, for boundaries that came from elsewhere."""
    return features.coefficients_to_vector(
        np.asarray(boundary.r_cos, dtype=np.float64),
        np.asarray(boundary.z_sin, dtype=np.float64),
    )


def evaluate(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    settings: forward_model_module.ConstellarationSettings | None = None,
) -> Evaluation:
    """Run the simulator on one boundary, reporting failure rather than raising."""
    if settings is None:
        settings = very_low_fidelity_settings()
    try:
        metrics, equilibrium = _forward_model().forward_model(
            boundary, settings=settings
        )
    except Exception as exc:
        return Evaluation(converged=False, failure=f"{type(exc).__name__}: {exc}")
    return Evaluation(
        converged=True,
        metrics=metrics,
        effective_aspect_ratio=effective_aspect_ratio(equilibrium),
    )


def evaluate_batch(
    boundaries: Sequence[surface_rz_fourier.SurfaceRZFourier],
    settings: forward_model_module.ConstellarationSettings | None = None,
    n_workers: int | None = None,
) -> list[Evaluation]:
    """Evaluate boundaries in parallel, preserving input order.

    Each call is CPU-bound inside the solver, so this uses processes rather than
    threads.

    The solver can fail hard enough to kill its worker rather than raise, which a plain
    pool turns into a broken pool and a lost batch. Designs decoded from latent
    interpolation are exactly the ones likely to do that, since nothing constrains them
    to be geometrically sensible. So work is retried on a fresh pool, narrowing to one
    worker, until every boundary has either an outcome or is recorded as having killed
    its worker.
    """
    if not boundaries:
        return []
    if settings is None:
        settings = very_low_fidelity_settings()
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    _LOG.info("Evaluating %d boundaries on %d workers", len(boundaries), n_workers)
    results: list[Evaluation | None] = [None] * len(boundaries)
    outstanding = list(range(len(boundaries)))
    workers = n_workers

    while outstanding:
        for index, evaluation in _evaluate_some(
            boundaries, settings, outstanding, workers
        ).items():
            results[index] = evaluation
        remaining = [index for index in outstanding if results[index] is None]
        if not remaining:
            break
        if workers > 1:
            _LOG.warning(
                "a worker died with %d boundaries unresolved; retrying serially",
                len(remaining),
            )
            workers = 1
        else:
            # Serial submission is processed in order, so everything before the
            # first unresolved boundary came back and this one is the culprit.
            # Blaming it costs at most one design if that reasoning is wrong,
            # and guarantees the loop shrinks.
            results[remaining[0]] = Evaluation(converged=False, failure=_WORKER_DIED)
            remaining = remaining[1:]
        outstanding = remaining
    return [_or_failure(result) for result in results]


def _evaluate_some(
    boundaries: Sequence[surface_rz_fourier.SurfaceRZFourier],
    settings: forward_model_module.ConstellarationSettings,
    indices: Sequence[int],
    workers: int,
) -> dict[int, Evaluation]:
    """Evaluate what it can, returning whatever resolved even if the pool breaks."""
    resolved: dict[int, Evaluation] = {}
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate, boundaries[index], settings): index
                for index in indices
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    resolved[futures[future]] = future.result()
                except concurrent.futures.process.BrokenProcessPool:
                    # Keep draining rather than stopping here. Once the pool
                    # breaks every pending future raises, but ones that already
                    # finished still carry a result, and `as_completed` can yield
                    # them after a broken one. Stopping would discard a good
                    # result and leave it unresolved, which the serial retry
                    # would then blame for killing the worker.
                    continue
    except concurrent.futures.process.BrokenProcessPool:
        pass
    return resolved


def _or_failure(result: Evaluation | None) -> Evaluation:
    return result or Evaluation(converged=False, failure=_WORKER_DIED)


def benchmark_feasible(metrics: Any) -> bool:
    """Whether a design satisfies the published MHD-stable benchmark's constraints.

    Delegates to the benchmark's own feasibility test, including its relative tolerance,
    rather than restating the bounds here.
    """
    return _problems().MHDStableQIStellarator().is_feasible(metrics)


def effective_aspect_ratio(equilibrium: Any) -> float:
    """Aspect ratio of the axisymmetric flat-axis configuration matching this one.

    Matching means carrying the same external poloidal current at the same field
    strength and minor radius, so the gap between this and the true aspect ratio grows
    with torsion of the magnetic axis. That gap is what separates the two design basins,
    which is why it appears in the conditioning vector.

    Not a constellaration metric. It is derived here from the equilibrium the forward
    model returns, rather than recomputed from the boundary, so no physics is
    duplicated.
    """
    poloidal_current = _extrapolate_to_axis(np.asarray(equilibrium.bvco)[1:])
    field_strength = _extrapolate_to_axis(np.asarray(equilibrium.bmnc)[0, 1:])
    return float(poloidal_current / equilibrium.Aminor_p / field_strength)


class StackedMetrics(NamedTuple):
    """Metrics stacked across evaluations, with the mask that says which are real.

    Named because both are tensors of the same shape, and reading them in the wrong
    order would silently treat a convergence mask as a metric.
    """

    values: torch.Tensor
    converged: torch.Tensor


def corpus_metrics(evaluations: Sequence[Evaluation]) -> dict[str, NDArray]:
    """Simulator output under this package's metric convention.

    Three columns are not read straight off the simulator. Quasi-isodynamic quality is
    reported as its base-10 logarithm, which is how it spans a usable range; the
    effective aspect ratio is derived from the equilibrium rather than returned by the
    forward model; and the difference between it and the true aspect ratio is the axis
    non-planarity the conditioning vector uses.

    Rows for designs the simulator could not solve are NaN throughout.
    """
    names = [
        field
        for field in _forward_model().ConstellarationMetrics.model_fields
        if field != "qi"
    ]
    columns: dict[str, list[float]] = {name: [] for name in names}
    columns[feasibility.LOG10_QI] = []
    columns[feasibility.EFFECTIVE_ASPECT_RATIO] = []
    columns[feasibility.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO] = []

    for evaluation in evaluations:
        metrics = evaluation.metrics if evaluation.converged else None
        for name in names:
            columns[name].append(_scalar(metrics, name))
        qi = _scalar(metrics, "qi")
        columns[feasibility.LOG10_QI].append(
            float(np.log10(qi)) if qi == qi and qi > 0 else float("nan")
        )
        effective = (
            evaluation.effective_aspect_ratio
            if evaluation.effective_aspect_ratio is not None
            else float("nan")
        )
        columns[feasibility.EFFECTIVE_ASPECT_RATIO].append(effective)
        columns[feasibility.EFFECTIVE_ASPECT_RATIO_MINUS_ASPECT_RATIO].append(
            effective - _scalar(metrics, "aspect_ratio")
        )
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _scalar(metrics: Any | None, name: str) -> float:
    if metrics is None:
        return float("nan")
    value = getattr(metrics, name, None)
    return float("nan") if value is None else float(value)


def metrics_to_tensor(
    evaluations: Iterable[Evaluation], metric_names: Sequence[str]
) -> StackedMetrics:
    """Stack selected metrics into a tensor alongside a convergence mask.

    Rows for designs the simulator could not solve are filled with NaN, so a caller that
    forgets the mask gets NaN rather than a plausible number.
    """
    rows: list[list[float]] = []
    converged: list[bool] = []
    for evaluation in evaluations:
        if evaluation.converged and evaluation.metrics is not None:
            rows.append([float(getattr(evaluation.metrics, n)) for n in metric_names])
            converged.append(True)
        else:
            rows.append([float("nan")] * len(metric_names))
            converged.append(False)
    return StackedMetrics(
        values=torch.tensor(rows, dtype=torch.float32),
        converged=torch.tensor(converged),
    )


def constellaration_available() -> bool:
    """Whether the optional `stellarator` extra is installed."""
    try:
        import constellaration  # noqa: F401  # pants: no-infer-dep
    except ImportError:
        return False
    return True


def _extrapolate_to_axis(values: NDArray) -> float:
    """First-order extrapolation to the axis from a quantity on the half grid.

    VMEC stores these quantities at half-grid points, none of which sits on the magnetic
    axis, so the axis value is extrapolated from the first two.
    """
    if values.shape[0] < 2:
        raise ValueError(
            f"need at least two half-grid points to extrapolate, got {values.shape[0]}"
        )
    return float(values[0] * 1.5 - values[1] * 0.5)


def _surface_rz_fourier() -> Any:
    return _import_or_explain("constellaration.geometry.surface_rz_fourier")


def _forward_model() -> Any:
    return _import_or_explain("constellaration.forward_model")


def _problems() -> Any:
    return _import_or_explain("constellaration.problems")


def _vmec_settings() -> Any:
    return _import_or_explain("constellaration.mhd.vmec_settings")


def _import_or_explain(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(_EXTRA_REQUIRED) from exc
