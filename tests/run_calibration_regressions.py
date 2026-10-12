"""Run existing regressions while enforcing the protected-manifest restriction.

Two tests explicitly opening the protected repository manifest are reported as
skipped; they are never silently counted as passing. No existing test is edited.
"""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
BLOCKED={
    'test_benchmark_manifest.BenchmarkManifestTests.test_checked_in_manifest_is_current',
    'test_benchmark_manifest.BenchmarkManifestTests.test_blind_records_require_explicit_access',
}


def main():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    def flatten(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from flatten(test)
            else:yield test
    tests=list(flatten(suite))
    for test in tests:
        if test.id() in BLOCKED:
            method=getattr(type(test),test._testMethodName)
            setattr(type(test),test._testMethodName,unittest.skip('Protected benchmark manifest access is not authorized')(method))
    result=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(tests))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':main()
