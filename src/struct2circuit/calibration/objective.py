"""Native normalized objective, exact-coordinate cache, and analytic derivatives."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from struct import pack

import numpy as np

from ..optimize import optimize_p1
from .config import BY_NAME


def array_hash(*arrays):
    digest = sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Evaluation:
    u: float
    beta: float
    gamma: float | None
    gap: float | None
    expectation: float
    span: float
    cost_status: str
    state: np.ndarray
    mixer_width: float

    def record(self):
        return dict(u=self.u, beta=self.beta, physical_gamma=self.gamma, normalized_gap=self.gap,
                    expectation=self.expectation, feasible_span=self.span, cost_status=self.cost_status,
                    objective_provenance="native compensated normalized_gap; dimensionless state",
                    v=self.beta*self.mixer_width)


class Objective:
    def __init__(self, simulator, budget, landscape, *, cache=None, account="essential", optional=False):
        self.simulator, self.budget, self.landscape = simulator, budget, landscape
        self.cache = {} if cache is None else cache
        self.account, self.optional = account, optional
        self.identity = array_hash(simulator.problem.Q, simulator.problem.c, simulator.basis,
                                   simulator.mixer_hamiltonian, simulator.initial_state,
                                   np.asarray(simulator.normalized_costs if simulator.normalized_costs is not None else [], dtype=float))
        self.identity += ':'+str(landscape)+':compensated-normalized-p1-v1'
        self.last = None

    def key(self, u, beta):
        if not np.isfinite([u, beta]).all():
            raise ValueError("Finite dimensionless coordinates required")
        return self.identity, pack('>dd', float(u), float(beta))

    def evaluate(self, u, beta, domain="B00", *, fresh=False):
        key = self.key(u, beta)
        self.budget.request(self.landscape, domain)
        if not fresh and key in self.cache:
            self.budget.counts['cache_hits'] += 1
            self.last = self.cache[key]
            return self.last
        self.budget.charge(key, self.landscape, domain, self.account, optional=self.optional)
        # This is exactly evaluate_dimensionless's pipeline, with the state retained
        # for derivatives. Reuse the existing state and result evaluators unchanged.
        state = self.simulator.state_dimensionless([u], [beta])
        result = self.simulator._evaluate_state(state)
        span = self.simulator.feasible_span
        gamma = float(u)/span if span > 0 else None
        if gamma is not None and (not np.isfinite(gamma) or (u != 0 and gamma == 0)):
            raise ArithmeticError("Physical gamma is outside the supported float64 range")
        if not np.isfinite(state).all() or result.normalized_gap is not None and not np.isfinite(result.normalized_gap):
            raise ArithmeticError("Nonfinite normalized state/objective")
        state.setflags(write=False)
        value = Evaluation(float(u), float(beta), gamma, result.normalized_gap,
                           result.expectation, span, result.cost_status, state, float(np.ptp(self.simulator._mixer_eigenvalues)))
        self.cache[key] = self.last = value
        self.budget.check_time()
        return value

    def value(self, point, domain):
        value = self.evaluate(*point, domain)
        if value.gap is None:
            raise ValueError("Constant or numerically unresolved costs have no normalized objective")
        return value.gap

    def gradient(self, point, domain):
        value = self.evaluate(*point, domain)
        if value.gap is None:
            raise ValueError("No normalized derivative for unresolved costs")
        self.budget.counts['gradient_computations'] += 1
        sim, u, beta = self.simulator, *map(float, point)
        costs = sim.normalized_costs
        phase = np.exp(-1j*u*costs)*sim.initial_state
        du = sim._apply_mixer(-1j*costs*phase, beta)
        db = -1j*sim.mixer_hamiltonian@value.state
        target = costs*value.state
        gradient = np.asarray([2*np.real(np.vdot(d, target)) for d in (du, db)])
        self.budget.check_time()
        return gradient


def production_search(objective, grid_size):
    """Call the unchanged optimizer through a recording, budgeted adapter."""
    if grid_size not in (9, 17, 33):
        raise ValueError("Only approved B00 production policies are eligible")
    sim = objective.simulator

    class Adapter:
        feasible_span = sim.feasible_span
        cost_status = sim.cost_status

        def evaluate_dimensionless(self, u, beta):
            # The final call is fresh, as is every production request; preserve the
            # unchanged solver's function values and selected exact coordinates.
            evaluation = objective.evaluate(float(u[0]), float(beta[0]), 'B00', fresh=True)
            return sim._evaluate_state(evaluation.state)

    result = optimize_p1(Adapter(), grid_size=grid_size, gamma_scale='feasible_span')
    selected = objective.last
    if selected is None or not BY_NAME['B00'].contains((selected.u, selected.beta)):
        raise ArithmeticError("Production selected an out-of-box candidate")
    return result, selected
