"""Deterministic parameter optimization for pilot-scale QAOA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .simulator import FeasibleSubspaceQAOA, QAOAResult


@dataclass(frozen=True)
class OptimizationResult:
    """Selected parameters and the provenance of their objective value.

    ``gamma`` is always physical gamma. ``parameter_bounds`` describe the
    search coordinates, named by ``gamma_coordinate`` (``u`` or ``gamma``)
    and beta. Evaluation counts include the grid and solver objective calls,
    but exclude the final evaluation used to report metrics.
    """

    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    result: QAOAResult
    objective_evaluations: int
    finite_objective: bool
    local_refinement_converged: bool
    selected_source: str
    selected_source_success: bool | None
    lbfgsb_success: bool | None
    powell_success: bool | None
    improvement_over_grid: float | None
    gamma_boundary_hit: bool | None
    beta_boundary_hit: bool | None
    gamma_coordinate: str
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]]


def optimize_p1(
    simulator: FeasibleSubspaceQAOA,
    grid_size: int = 17,
    *,
    gamma_scale: str = "feasible_span",
) -> OptimizationResult:
    """Coarse deterministic grid followed by two bounded local refinements.

    Both solvers start from the identical grid point. ``feasible_span`` uses
    ``u = gamma * (C_max - C_min)`` and centered normalized costs throughout
    the search. Its box is a scale convention, not a fundamental period.
    ``legacy`` retains the original raw-gamma/raw-expectation search.

    ``improvement_over_grid`` is measured in normalized-gap units, or is None
    for a constant/unresolved objective. Boundary flags use a relative box
    tolerance of 1e-8. ``local_refinement_converged`` refers only to a selected
    local candidate; individual solver statuses are reported separately.
    """
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    if gamma_scale not in {"feasible_span", "legacy"}:
        raise ValueError("gamma_scale must be 'feasible_span' or 'legacy'")
    normalized = gamma_scale == "feasible_span"
    span = simulator.feasible_span
    bounds = [(0.0, 2.0 * np.pi), (0.0, np.pi)]
    gamma_coordinate = "u" if normalized else "gamma"
    if normalized and simulator.cost_status != "nonconstant":
        result = simulator.evaluate_dimensionless([0.0], [0.0])
        return OptimizationResult(
            gamma=(0.0,),
            beta=(0.0,),
            result=result,
            objective_evaluations=0,
            finite_objective=bool(np.isfinite(result.expectation)),
            local_refinement_converged=False,
            selected_source="not_run",
            selected_source_success=None,
            lbfgsb_success=None,
            powell_success=None,
            improvement_over_grid=None,
            gamma_boundary_hit=None,
            beta_boundary_hit=None,
            gamma_coordinate=gamma_coordinate,
            parameter_bounds=tuple(bounds),
        )

    def evaluate(theta: np.ndarray | tuple[float, float]) -> QAOAResult:
        if normalized:
            return simulator.evaluate_dimensionless([theta[0]], [theta[1]])
        return simulator.evaluate([theta[0]], [theta[1]])

    def objective_value(theta: np.ndarray | tuple[float, float]) -> float:
        result = evaluate(theta)
        return float(result.normalized_gap) if normalized else result.expectation

    gammas = np.linspace(0.0, bounds[0][1], grid_size, endpoint=False)
    betas = np.linspace(0.0, np.pi, grid_size, endpoint=False)
    best_value = float("inf")
    best = (0.0, 0.0)
    evaluations = 0
    for gamma in gammas:
        for beta in betas:
            value = objective_value((gamma, beta))
            evaluations += 1
            if value < best_value:
                best_value = value
                best = (float(gamma), float(beta))

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return objective_value(theta)

    refined = minimize(
        objective,
        np.asarray(best),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-13, "gtol": 1e-9},
    )
    powell = minimize(
        objective,
        np.asarray(best),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 180, "ftol": 1e-12, "xtol": 1e-10},
    )
    candidates = [(best_value, np.asarray(best), "grid", None)]
    if np.isfinite(refined.fun):
        candidates.append((float(refined.fun), refined.x, "lbfgsb", bool(refined.success)))
    if np.isfinite(powell.fun):
        candidates.append((float(powell.fun), powell.x, "powell", bool(powell.success)))
    selected_value, selected, source, success = min(candidates, key=lambda item: item[0])
    coordinate, beta = (float(selected[0]), float(selected[1]))
    gamma = coordinate / span if normalized else coordinate
    if normalized and (not np.isfinite(gamma) or (coordinate != 0.0 and gamma == 0.0)):
        raise ValueError("selected physical gamma is outside the supported floating-point range")
    result = evaluate(selected)
    improvement = best_value - selected_value
    if simulator.cost_status != "nonconstant":
        improvement = None
    elif not normalized:
        improvement /= span

    def boundary_hit(value: float, bound: tuple[float, float]) -> bool:
        tolerance = 1e-8 * (bound[1] - bound[0])
        return min(abs(value - bound[0]), abs(value - bound[1])) <= tolerance

    return OptimizationResult(
        gamma=(gamma,),
        beta=(beta,),
        result=result,
        objective_evaluations=evaluations,
        finite_objective=bool(np.isfinite(result.expectation)),
        local_refinement_converged=source != "grid" and bool(success),
        selected_source=source,
        selected_source_success=success,
        lbfgsb_success=bool(refined.success),
        powell_success=bool(powell.success),
        improvement_over_grid=improvement,
        gamma_boundary_hit=boundary_hit(coordinate, bounds[0]),
        beta_boundary_hit=boundary_hit(beta, bounds[1]),
        gamma_coordinate=gamma_coordinate,
        parameter_bounds=tuple(bounds),
    )
