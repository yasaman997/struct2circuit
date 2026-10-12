"""The unchanged reference algorithm, with explicit Powell start evidence."""
from ..calibration.reference import ReferenceSearch as V1ReferenceSearch


class ReferenceSearch(V1ReferenceSearch):
    def _attempt(self, result, domain, route, source, runner, cap, seed=None):
        before = len(result.candidates)
        attempts_before = len(result.attempts)
        try:
            return super()._attempt(result, domain, route, source, runner, cap, seed)
        finally:
            # Powell's first objective call is its declared initial point. Retain
            # it even when a later call interrupts the attempt; no solver changes.
            if (source == 'powell' and len(result.attempts) > attempts_before
                and len(result.candidates) > before):
                result.attempts[-1]['start'] = result.candidates[before].point
