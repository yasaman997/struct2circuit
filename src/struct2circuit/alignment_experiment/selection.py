"""Exact coefficient-only C6 orbit and alignment selection, before simulation."""
from fractions import Fraction
from itertools import combinations, permutations
from math import fsum, sqrt

import numpy as np

from ..calibration.objective import array_hash

PERMUTATIONS = tuple(permutations(range(6)))
BASE_EDGES = ((0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5))


def relabel(edges, p):
    return tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in edges))


def cycles():
    representatives = {}
    for p in PERMUTATIONS:
        representatives.setdefault(relabel(BASE_EDGES, p), p)
    assert len(PERMUTATIONS) == 720 and len(representatives) == 60
    return dict(sorted(representatives.items()))


def exact_costs(problem):
    if (problem.n, problem.k) != (6, 3) or not np.array_equal(problem.Q, problem.Q.T):
        raise ValueError('This frozen experiment requires n=6,k=3 and exactly symmetric Q')
    occupied = tuple(combinations(range(6), 3))
    q = [[Fraction(float(v)) for v in row] for row in problem.Q]
    c = [Fraction(float(v)) for v in problem.c]
    costs = [sum((q[i][j] for i in state for j in state), Fraction()) + sum((c[i] for i in state), Fraction())
             for state in occupied]
    return occupied, costs, q, c


def select(problem):
    occupied, costs, q, c = exact_costs(problem)
    lookup = dict(zip(occupied, costs))
    symmetries = [p for p in PERMUTATIONS if all(lookup[tuple(sorted(p[i] for i in state))] == value
                                               for state, value in lookup.items())]
    span = max(map(float, costs)) - min(map(float, costs))
    status = 'nonconstant' if span > 0 else 'constant_or_unresolved'
    rho = {}
    for i, j in combinations(range(6), 2):
        b = [2*(q[j][l]-q[i][l]) for l in range(6) if l not in (i, j)]
        mean_b = sum(b)/4
        mu = q[j][j]-q[i][i]+c[j]-c[i]+sum(b)/2
        variance = sum((v-mean_b)**2 for v in b)/3  # m(N-m)/(N(N-1)) = 1/3
        rho[i, j] = sqrt(float(mu*mu+variance))
    enumeration = []
    for edges, p in cycles().items():
        representative = min(relabel(edges, s) for s in symmetries)
        score = fsum(rho[e] for e in edges)/(6*span) if span > 0 else None
        enumeration.append(dict(edges=edges, permutation=p, representative=representative,
                                score=score, selection_score=round(score, 12) if score is not None else None))
    eligible = [r for r in enumeration if r['edges'] == r['representative']]
    chosen = {}
    if span > 0 and len(eligible) >= 2:
        low = min(r['selection_score'] for r in eligible)
        high = max(r['selection_score'] for r in eligible)
        if low != high:
            chosen = {arm: next(r for r in eligible if r['selection_score'] == score)
                      for arm, score in [('low', low), ('high', high)]}
    return dict(coefficients_sha256=array_hash(problem.Q, problem.c), n=6, k=3, span=span,
                cost_status=status, status='selected' if chosen else 'NO_ALIGNMENT_CONTRAST',
                raw_permutations=720, distinct_cycles=60, cost_symmetries=symmetries,
                enumeration=enumeration, selected=chosen)
