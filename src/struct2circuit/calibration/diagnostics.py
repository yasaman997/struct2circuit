"""Independent numerical oracles and native-H spectral/invariance controls."""
from __future__ import annotations

from copy import copy
from dataclasses import replace

import numpy as np
from scipy.linalg import expm

from ..mixers import MixerSpec
from ..problems import CardinalityQUBO
from ..simulator import FeasibleSubspaceQAOA
from .budget import BudgetExhausted
from .decisions import GAIN, band
from .objective import Objective

POINT_TOL = 1e-10
GRADIENT_TOL = 1e-6


def charge_oracle(obj, point, domain, tag):
    obj.budget.request(obj.landscape, domain)
    obj.budget.charge(obj.key(*point), obj.landscape, domain, obj.account, oracle=True, optional=obj.optional)


def direct_oracle(obj, point, domain='B00', *, record_callback=None):
    charge_oracle(obj, point, domain, 'direct_expm')
    sim = obj.simulator
    if sim.normalized_costs is None:
        return dict(status='unresolved', reason='constant_or_unresolved')
    obj.budget.counts['matrix_exponentials'] += 2
    u, beta = point
    state = expm(-1j*beta*sim.mixer_hamiltonian)@expm(-1j*u*np.diag(sim.normalized_costs))@sim.initial_state
    native = obj.evaluate(u, beta, domain)
    probabilities = abs(state)**2
    errors = dict(probability=float(np.max(abs(probabilities-abs(native.state)**2))),
                  gap=abs(float(probabilities@sim.normalized_costs)-native.gap),
                  normalization=abs(float(np.sum(probabilities))-1.),
                  native_normalization=abs(float(np.sum(abs(native.state)**2))-1.),
                  native_state_norm=abs(float(np.linalg.norm(native.state))-1.))
    check = dict(kind='direct_expm', point=tuple(point), domain=domain, native=native.record(),
                 status='pass' if max(errors.values()) <= POINT_TOL else 'fail', errors=errors)
    if record_callback:
        record_callback(check)
    obj.budget.check_time()
    return check


def gradient_check(obj, point, domain='B11'):
    width = float(np.ptp(obj.simulator._mixer_eigenvalues))
    steps = (1e-5, 1e-5/max(1., width))
    analytic = obj.gradient(point, domain)
    numerical = []
    for axis, step in enumerate(steps):
        plus, minus = np.array(point, dtype=float), np.array(point, dtype=float)
        plus[axis] += step
        minus[axis] -= step
        numerical.append((obj.value(plus, domain)-obj.value(minus, domain))/(2*step))
    error = float(np.max(abs(analytic-numerical)))
    return dict(kind='gradient', point=tuple(point), domain=domain, steps=steps,
                status='pass' if error <= GRADIENT_TOL else 'fail', error=error,
                analytic=analytic.tolist(), numerical=numerical)


def spectral_coordinates(obj, point, domain='B00', *, record_callback=None):
    sim = obj.simulator
    values = sim._mixer_eigenvalues
    width, center = float(np.ptp(values)), float((values[0]+values[-1])/2)
    if width <= 0 or sim.normalized_costs is None:
        return dict(status='unresolved')
    charge_oracle(obj, point, domain, 'spectral_coordinates')
    obj.budget.counts['matrix_exponentials'] += 1
    u, beta = point
    transformed = (sim.mixer_hamiltonian-center*np.eye(len(values)))/width
    state = expm(-1j*(beta*width)*transformed)@(np.exp(-1j*u*sim.normalized_costs)*sim.initial_state)
    native = obj.evaluate(u, beta, domain)
    probability_error = float(np.max(abs(abs(state)**2-abs(native.state)**2)))
    gap_error = abs(float(abs(state)**2@sim.normalized_costs)-native.gap)
    check = dict(kind='mapped_spectral_coordinates', point=tuple(point), domain=domain,
                 status='pass' if max(probability_error, gap_error) <= POINT_TOL else 'fail',
                 probability_error=probability_error, gap_error=gap_error, v=beta*width)
    if record_callback:
        record_callback(check)
    obj.budget.check_time()
    return check


def _degrees(values):
    return dict(min=int(np.min(values)), max=int(np.max(values)), mean=float(np.mean(values)), variance=float(np.var(values)))


def spectral_diagnostics(sim):
    eigenvalues = sim._mixer_eigenvalues
    physical = np.zeros(sim.problem.n, dtype=int)
    for i, j in sim.mixer.edges:
        physical[i] += 1
        physical[j] += 1
    extrema = {}
    for which, endpoint in (('low', eigenvalues[0]), ('high', eigenvalues[-1])):
        mask = sim._extremal_mask(which)
        state = getattr(sim, f'mixer_{which}_state')
        other = eigenvalues[~mask]
        fidelity = sim.uniform_mixer_extremal_fidelity(which)
        extrema[which] = dict(multiplicity=int(mask.sum()), uniform_fidelity=fidelity,
                             projection_branch='projection' if np.sqrt(fidelity) > 1e-12 else 'lexicographic_fallback',
                             residual=float(np.linalg.norm(sim.mixer_hamiltonian@state-endpoint*state)),
                             next_distinct_gap=float(np.min(abs(other-endpoint))) if len(other) else None)
    return dict(lambda_min=float(eigenvalues[0]), lambda_max=float(eigenvalues[-1]), width=float(np.ptp(eigenvalues)),
                rho=float(np.max(abs(eigenvalues))), physical_degree=_degrees(physical),
                feasible_degree=_degrees(np.sum(sim.mixer_hamiltonian != 0, axis=1)),
                neighboring_separations=np.diff(eigenvalues).tolist(), extrema=extrema)


def reuse_initialization(base, initialization, budget):
    if initialization not in ('uniform', 'mixer_low', 'mixer_high'):
        raise ValueError('Unknown initialization')
    sim = copy(base)
    sim.initialization = initialization
    sim.initial_state = getattr(base, initialization+'_state') if initialization != 'uniform' else base.uniform_state
    budget.counts['eigensolver_reuse'] += 1
    return sim


def invariance_checks(obj, points, *, record_callback=None):
    """Freeze graph/state for scaling; explicitly distinguish transported/rebuilt states."""
    sim, budget = obj.simulator, obj.budget
    records = []
    def record(check):
        records.append(check)
        if record_callback:
            record_callback(check)
    for multiplier in (2.**-20, 2.**20):
        problem = CardinalityQUBO(sim.problem.Q*multiplier, sim.problem.c*multiplier, sim.problem.k)
        scaled = FeasibleSubspaceQAOA(problem, sim.mixer, initialization=sim.initialization)
        budget.counts['eigensolver_calls'] += 1
        scaled.initial_state = sim.initial_state
        other = Objective(scaled, budget, obj.landscape, account=obj.account)
        cost_error = float(np.max(abs(scaled.normalized_costs-sim.normalized_costs)))
        for point in points:
            a, b = obj.evaluate(*point), other.evaluate(*point)
            error = max(cost_error, abs(a.gap-b.gap), float(np.max(abs(abs(a.state)**2-abs(b.state)**2))))
            gamma_ok = np.isclose(a.gamma/multiplier, b.gamma, rtol=1e-10, atol=0)
            record(dict(kind='cost_units', multiplier=multiplier, point=point,
                                status='pass' if error <= POINT_TOL and gamma_ok else 'fail', error=error))
    permutation = np.roll(np.arange(sim.problem.n), 1)  # new label j stores old label permutation[j]
    inverse = np.argsort(permutation)
    problem = CardinalityQUBO(sim.problem.Q[np.ix_(permutation, permutation)], sim.problem.c[permutation], sim.problem.k)
    graph = MixerSpec('cyclic_transport', sim.mixer.n,
                     tuple((int(inverse[i]), int(inverse[j])) for i, j in sim.mixer.edges), 'transport')
    rebuilt = FeasibleSubspaceQAOA(problem, graph, initialization=sim.initialization)
    budget.counts['eigensolver_calls'] += 1
    index = {tuple(row): i for i, row in enumerate(sim.basis)}
    mapping = []
    for row in rebuilt.basis:
        old = np.empty_like(row)
        old[permutation] = row
        mapping.append(index[tuple(old)])
    transported = copy(rebuilt)
    transported.initial_state = sim.initial_state[mapping].copy()
    transported_obj = Objective(transported, budget, obj.landscape, account=obj.account)
    rebuilt_obj = Objective(rebuilt, budget, obj.landscape, account=obj.account)
    for point in points:
        a = obj.evaluate(*point)
        b = transported_obj.evaluate(*point)
        error = max(abs(a.gap-b.gap), float(np.max(abs(abs(a.state[mapping])**2-abs(b.state)**2))))
        record(dict(kind='cyclic_transport', point=point, status='pass' if error <= POINT_TOL else 'fail',
                    error=error, permutation=permutation.tolist()))
        try:
            c = rebuilt_obj.evaluate(*point)
        except BudgetExhausted as interruption:
            record(dict(kind='cyclic_reconstruction', point=point, status='unresolved',
                        reconstructed_gap_sensitivity=None, selection_status='unresolved',
                        reason=str(interruption), permutation=permutation.tolist()))
            raise
        sensitivity = abs(a.gap-c.gap)
        record(dict(kind='cyclic_reconstruction', point=point, status='pass',
                    reconstructed_gap_sensitivity=sensitivity,
                    selection_status=band(sensitivity, GAIN), permutation=permutation.tolist()))
    return records


def common_spectral_time_reference(objective, ring_width, n_beta, identity, *, baseline_reference=None):
    """Optional Bv reference only, never expanded production qualification.

    Caller must supply the preselected development-audit case, after all mandatory
    work. This opt-in helper does not generate QUBOs or enlarge the optional panel.
    """
    from .config import spectral_domain
    from .reference import ReferenceSearch
    from .decisions import reference_status
    if not objective.optional or objective.account != 'shared' or not objective.budget.mandatory_complete:
        raise ValueError('Bv requires the optional account after mandatory completion')
    width = float(np.ptp(objective.simulator._mixer_eigenvalues))
    domain = spectral_domain(ring_width, width)
    search = ReferenceSearch(objective, n_beta, identity)
    contained = tuple(baseline_reference.candidates) if baseline_reference else ()
    result = search.search(domain, contained=contained)
    if (reference_status(result.r) != 'pass' or not result.faces_complete) and not objective.budget.stop_reason:
        search.refine_domain(result, domain, contained=contained)
    return dict(domain=domain, reference=result, physical_convention='common_spectral_time',
                production_qualification='NOT_TESTED', completed=result.complete)
