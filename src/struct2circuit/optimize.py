"""Deterministic parameter optimization for pilot-scale QAOA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .simulator import FeasibleSubspaceQAOA, QAOAResult


@dataclass(frozen=True)
class OptimizationResult:
    gamma: tuple[float, ...]
    beta: tuple[float, ...]
    result: QAOAResult
    objective_evaluations: int
    finite_objective: bool
    local_refinement_converged: bool


def optimize_p1(
    simulator: FeasibleSubspaceQAOA,
    grid_size: int = 17,
    *,
    gamma_scale: str = "feasible_span",
) -> OptimizationResult:
    """Coarse deterministic grid followed by two bounded local refinements.

    L-BFGS-B is efficient in smooth interior regions, while Powell is more
    reliable when the optimum lies on a periodic box boundary. Both start from
    the identical grid point; the best finite result is retained.
    """
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    if gamma_scale not in {"feasible_span", "legacy"}:
        raise ValueError("gamma_scale must be 'feasible_span' or 'legacy'")
    span = simulator.cost_max - simulator.cost_min
    if gamma_scale == "feasible_span" and span > 1e-14:
        gamma_max = 2.0 * np.pi / span
    else:
        gamma_max = 2.0 * np.pi
    gammas = np.linspace(0.0, gamma_max, grid_size, endpoint=False)
    betas = np.linspace(0.0, np.pi, grid_size, endpoint=False)
    best_value = float("inf")
    best = (0.0, 0.0)
    evaluations = 0
    for gamma in gammas:
        for beta in betas:
            value = simulator.evaluate([gamma], [beta]).expectation
            evaluations += 1
            if value < best_value:
                best_value = value
                best = (float(gamma), float(beta))

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return simulator.evaluate([theta[0]], [theta[1]]).expectation

    refined = minimize(
        objective,
        np.asarray(best),
        method="L-BFGS-B",
        bounds=[(0.0, gamma_max), (0.0, np.pi)],
        options={"maxiter": 120, "ftol": 1e-13, "gtol": 1e-9},
    )
    powell = minimize(
        objective,
        np.asarray(best),
        method="Powell",
        bounds=[(0.0, 2.0 * np.pi), (0.0, np.pi)],
        options={"maxiter": 180, "ftol": 1e-12, "xtol": 1e-10},
    )
    candidates = [(best_value, np.asarray(best), True)]
    if np.isfinite(refined.fun):
        candidates.append((float(refined.fun), refined.x, bool(refined.success)))
    if np.isfinite(powell.fun):
        candidates.append((float(powell.fun), powell.x, bool(powell.success)))
    _, selected, _ = min(candidates, key=lambda item: item[0])
    gamma, beta = (float(selected[0]), float(selected[1]))
    result = simulator.evaluate([gamma], [beta])
    return OptimizationResult(
        gamma=(gamma,),
        beta=(beta,),
        result=result,
        objective_evaluations=evaluations,
        finite_objective=bool(np.isfinite(result.expectation)),
        local_refinement_converged=bool(refined.success or powell.success),
    )
