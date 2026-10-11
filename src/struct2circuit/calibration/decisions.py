"""Pure numerical rules. Missing data never passes and known failures take priority."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

REFERENCE = 1e-5
DEFICIT = 1e-4
GAIN = 1e-3
RANK = 2e-3
DISTORTION = 2e-4
GUARD = 1e-5
THRESHOLDS = dict(reference=REFERENCE, deficit=DEFICIT, gain=GAIN, rank=RANK, distortion=DISTORTION, guard=GUARD)


def finite(*values):
    return all(v is not None and isfinite(v) for v in values)


def difference(a, b):
    return a-b if finite(a, b) else None


def agreement(a, b, pooled):
    return max(abs(a-b), abs(a-pooled), abs(b-pooled)) if finite(a, b, pooled) else None


def refinement(coarse, fine):
    return difference(coarse, fine)


def deficit(production, reference):
    return difference(production, reference)


def domain_gain(narrow, wide):
    return difference(narrow, wide)


def challenge_gains(r00, r11, r22, rsign):
    return dict(direct=domain_gain(r00, r22), step=domain_gain(r11, r22),
                signed=domain_gain(r11, rsign), signed_total=domain_gain(r00, rsign))


def contrast(i, j):
    return difference(j, i)


def distortion(pi, pj, ri, rj):
    return abs((pj-pi)-(rj-ri)) if finite(pi, pj, ri, rj) else None


def band(value, threshold):
    if not finite(value):
        return "unresolved"
    return "pass" if value < threshold-GUARD else "fail" if value > threshold+GUARD else "ambiguous"


def reference_status(value):
    if not finite(value) or value < 0:
        return "unresolved"
    return "pass" if value <= REFERENCE else "unresolved"


def ordering(value):
    if not finite(value):
        return "unresolved"
    if abs(value) > RANK+GUARD:
        return "i" if value > 0 else "j"
    return "near_tie" if abs(value) < RANK-GUARD else "ambiguous"


def reversal(first, second):
    a, b = ordering(first), ordering(second)
    if {a, b} == {"i", "j"}:
        return "reversal"
    if "unresolved" in (a, b):
        return "unresolved"
    if "ambiguous" in (a, b) and finite(first, second) and first*second < 0:
        return "ambiguous"
    return "none"


def challenge_required(*, sentinel=False, near_upper=False, gain=None, rank_change="none"):
    return sentinel or near_upper or band(gain, GAIN) in ("fail", "ambiguous") or rank_change in ("reversal", "ambiguous")


def escalation_reasons(*, r=None, faces_complete=True, e=None, k=None, gains=(), near_nonflat=False, rank_unresolved=False, audit=False):
    reasons = []
    if reference_status(r) != 'pass': reasons.append('reference_agreement')
    if not faces_complete: reasons.append('incomplete_faces')
    if e is not None and band(e, DEFICIT) != 'pass': reasons.append('production_deficit')
    if k is not None and band(k, DISTORTION) != 'pass': reasons.append('paired_distortion')
    if any(band(g, GAIN) != 'pass' for g in gains): reasons.append('domain_gain')
    if near_nonflat: reasons.append('nonflat_boundary')
    if rank_unresolved: reasons.append('ranking')
    if audit: reasons.append('scheduled_audit')
    return tuple(reasons)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    essential: bool = True

    def __post_init__(self):
        if self.status not in ('pass', 'fail', 'ambiguous', 'unresolved', 'missing', 'omitted'):
            raise ValueError("Unknown check status")


def classify(checks, expected, *, optional_complete=False, limitations=()):
    """Expected names are declared independently of observed rows."""
    checks = tuple(checks)
    names = [c.name for c in checks]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate acceptance check")
    if not expected:
        return 'INCONCLUSIVE'
    if any(c.status == 'fail' and c.essential for c in checks):
        return 'FAILED'
    observed = {c.name: c for c in checks}
    if not set(expected) <= set(observed):
        return 'INCONCLUSIVE'
    if any(observed[n].status in ('missing', 'unresolved', 'ambiguous', 'omitted') for n in expected):
        return 'INCONCLUSIVE'
    if any(c.status == 'fail' for c in checks) or limitations or not optional_complete:
        return 'ACCEPTED WITH EXPLICIT LIMITATIONS'
    return 'ACCEPTED'
