from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from struct2circuit.calibration.config import (BY_NAME, PRIMARY, beta_intervals, build_manifest,
    construct_graphs, de_seed, grid_axes, instance_specs, spectral_domain, verify_seed_assignments)
from struct2circuit.calibration.decisions import (Check, DEFICIT, DISTORTION, GAIN, GUARD,
    agreement, band, challenge_gains, challenge_required, classify, contrast, deficit,
    distortion, escalation_reasons, ordering, reference_status, refinement, reversal)
from struct2circuit.problems import CardinalityQUBO


class ConfigurationTests(unittest.TestCase):
    def test_manifest_is_metadata_only_and_complete(self):
        with patch('struct2circuit.problems.block_correlated_qubo', side_effect=AssertionError('No official generation')):
            manifest = build_manifest()
        self.assertEqual(len(manifest['instances']), 8)
        self.assertEqual(len(manifest['landscapes']), 192)
        self.assertEqual(len(manifest['audits']), 24)
        self.assertEqual(len(manifest['mandatory_challenges']), 16)
        self.assertEqual({r['instance_id'] for r in manifest['instances']}, {0, 2, 4, 6, 8, 9, 10, 11})
        self.assertEqual(manifest['status'], 'NOT_EXECUTED')
        self.assertEqual(len(build_manifest(True)['landscapes']), 288)
        self.assertEqual(len(build_manifest(True)['audits']), 36)

    def test_exact_seed_namespaces(self):
        self.assertEqual(verify_seed_assignments(), 4032)
        self.assertEqual(de_seed(2, 3, 1, 0, 1), 720020311)
        self.assertEqual(de_seed(2, 3, 1, 5, 1), 730020313)
        seeds = {spec.seed for spec in instance_specs(True)}
        self.assertEqual(len(seeds), 12)
        self.assertIn(420103000, seeds)

    def test_nested_grids_retain_exact_old_nodes(self):
        coarse = grid_axes(BY_NAME['B11'], 64)
        fine = grid_axes(BY_NAME['B11'], 64, 1)
        self.assertEqual(tuple(len(a) for a in coarse), (65, 65))
        for old, new in zip(coarse, fine):
            np.testing.assert_array_equal(old, new[::2])
        for box in PRIMARY:
            u, b = grid_axes(box, 64)
            self.assertEqual((u[0], u[-1]), box.u)
            self.assertEqual((b[0], b[-1]), box.beta)
            self.assertTrue(set(u) <= set(coarse[0]))
            self.assertTrue(set(b) <= set(coarse[1]))
        self.assertEqual(tuple(len(a) for a in grid_axes(BY_NAME['B22'], 64)), (129, 129))
        self.assertEqual(tuple(len(a) for a in grid_axes(BY_NAME['Bsign'], 64)), (65, 129))

    def test_domains_and_spectral_control(self):
        self.assertTrue(BY_NAME['B22'].includes(BY_NAME['B11']))
        self.assertTrue(BY_NAME['Bsign'].includes(BY_NAME['B11']))
        self.assertFalse(BY_NAME['B00'].contains((0, -1)))
        self.assertEqual(beta_intervals(12), 128)
        self.assertEqual(beta_intervals(18), 256)
        self.assertAlmostEqual(spectral_domain(2, 4).beta[1], np.pi/2)
        with self.assertRaises(ValueError): spectral_domain(0, 2)
        with self.assertRaises(ValueError): grid_axes(BY_NAME['B00'], 65)

    def test_dedup_retains_labels_and_shuffle_determinism(self):
        problem = CardinalityQUBO(np.zeros((4, 4)), np.arange(4.), 2)
        a, canonical, permutation = construct_graphs(problem, 1042)
        b, repeated, second_permutation = construct_graphs(problem, 1042)
        self.assertEqual(len(a), 8)
        self.assertEqual([g.edges for g in a], [g.edges for g in b])
        self.assertEqual(canonical, repeated)
        self.assertEqual(permutation, second_permutation)
        for i, c in enumerate(canonical):
            self.assertLessEqual(c, i)
            self.assertEqual(a[c].edges, a[i].edges)
            self.assertEqual(c, min(j for j, graph in enumerate(a) if graph.edges == a[i].edges))

    def test_public_seed_collision_blocks_without_accessing_protected_records(self):
        import tempfile
        from struct2circuit.calibration.config import verify_public_seed_metadata
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'configs').mkdir()
            (root/'results').mkdir()
            (root/'configs/benchmark_v1_draft.json').write_text(json.dumps(dict(master_seed=12345, splits={}, random_baseline_replicates=1)))
            for name in ('pilot_results.csv', 'independent_dks_benchmark_v1_instances.csv'):
                (root/'results'/name).write_text('problem_seed\n17\n')
            with patch('struct2circuit.benchmark.build_manifest', side_effect=AssertionError('Do not inspect blind records')):
                self.assertFalse(verify_public_seed_metadata(root)['protected_records_accessed'])
                (root/'results/pilot_results.csv').write_text('problem_seed\n420000000\n')
                with self.assertRaisesRegex(ValueError, 'collision'):
                    verify_public_seed_metadata(root)

    def test_manifest_records_exact_settings_and_all_requested_seeds(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest['search_seeds']), 192)
        self.assertEqual(manifest['settings']['local']['objective_call_cap'], 250)
        self.assertEqual(manifest['settings']['budgets']['total'], 8_000_000)
        self.assertEqual(manifest['settings']['de']['routine_maxiter'], 63)
        self.assertFalse(manifest['settings']['broader_adequacy_essential'])


class DecisionTests(unittest.TestCase):
    def test_minimization_signs_and_complete_reference_formula(self):
        self.assertAlmostEqual(deficit(.21, .2), .01)
        self.assertAlmostEqual(refinement(.21, .2), .01)
        self.assertAlmostEqual(agreement(.2, .200002, .2), .000002)
        gains = challenge_gains(.3, .2, .1, .15)
        for name, expected in [('direct', .2), ('step', .1), ('signed', .05), ('signed_total', .15)]:
            self.assertAlmostEqual(gains[name], expected)
        self.assertGreater(contrast(.1, .2), 0)
        self.assertAlmostEqual(distortion(.10005, .2, .1, .2), .00005)

    def test_every_guard_boundary_and_neighbor(self):
        for threshold in (DEFICIT, GAIN, DISTORTION):
            with self.subTest(threshold=threshold):
                lo, hi = threshold-GUARD, threshold+GUARD
                self.assertEqual(band(lo, threshold), 'ambiguous')
                self.assertEqual(band(threshold, threshold), 'ambiguous')
                self.assertEqual(band(hi, threshold), 'ambiguous')
                self.assertEqual(band(np.nextafter(lo, -np.inf), threshold), 'pass')
                self.assertEqual(band(np.nextafter(hi, np.inf), threshold), 'fail')
                self.assertEqual(band(None, threshold), 'unresolved')
                self.assertEqual(band(float('nan'), threshold), 'unresolved')

    def test_reference_exception_and_ranking(self):
        self.assertEqual(reference_status(0), 'pass')
        self.assertEqual(reference_status(1e-5), 'pass')
        self.assertEqual(reference_status(np.nextafter(1e-5, np.inf)), 'unresolved')
        self.assertEqual(ordering(.003), 'i')
        self.assertEqual(ordering(-.003), 'j')
        self.assertEqual(ordering(.001), 'near_tie')
        self.assertEqual(ordering(.002), 'ambiguous')
        self.assertEqual(reversal(.003, -.003), 'reversal')
        self.assertEqual(reversal(.001, -.003), 'none')
        self.assertEqual(reversal(.002, -.003), 'ambiguous')

    def test_escalation_and_challenge_triggers(self):
        self.assertTrue(challenge_required(sentinel=True))
        self.assertTrue(challenge_required(near_upper=True))
        self.assertTrue(challenge_required(gain=GAIN))
        self.assertTrue(challenge_required(rank_change='ambiguous'))
        self.assertFalse(challenge_required(gain=0))
        self.assertEqual(escalation_reasons(r=0), ())
        self.assertEqual(escalation_reasons(r=0, audit=True), ('scheduled_audit',))
        self.assertIn('reference_agreement', escalation_reasons(r=2e-5))

    def test_classification_completeness_and_failure_precedence(self):
        passed = Check('required', 'pass')
        self.assertEqual(classify([passed], ['required'], optional_complete=True), 'ACCEPTED')
        self.assertEqual(classify([passed], ['required']), 'ACCEPTED WITH EXPLICIT LIMITATIONS')
        self.assertEqual(classify([passed], ['required', 'missing']), 'INCONCLUSIVE')
        self.assertEqual(classify([], []), 'INCONCLUSIVE')
        self.assertEqual(classify([Check('bad', 'fail')], ['missing']), 'FAILED')
        self.assertEqual(classify([passed, Check('broader', 'fail', False)], ['required', 'broader']), 'ACCEPTED WITH EXPLICIT LIMITATIONS')
        self.assertEqual(classify([passed, Check('broader', 'fail', True)], ['required', 'broader']), 'FAILED')
        self.assertEqual(classify([passed, Check('broader', 'unresolved', False)], ['required', 'broader']), 'INCONCLUSIVE')
        with self.assertRaises(ValueError): classify([passed, passed], ['required'])


if __name__ == '__main__': unittest.main()
