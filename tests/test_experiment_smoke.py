from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExperimentSmokeTests(unittest.TestCase):
    def run_script(self, script: str, output: Path, *arguments: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(output.parent / "matplotlib"),
            "XDG_CACHE_HOME": str(output.parent / "cache"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        })
        return subprocess.run(
            [sys.executable, str(ROOT / "experiments" / script), *arguments, "--output", str(output)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )

    def read_rows(self, output: Path) -> list[dict]:
        with output.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def assert_historical_semantics(self, rows: list[dict]) -> None:
        for row in rows:
            self.assertEqual(row["gamma_scale"], "legacy")
            self.assertEqual(row["finite_objective"], "True")
            self.assertEqual(row["cost_status"], "nonconstant")
            self.assertIn(row["selected_source"], ("grid", "lbfgsb", "powell"))
            self.assertIn("selected_source_success", row)
            self.assertIn("lbfgsb_success", row)
            self.assertIn("powell_success", row)
            self.assertGreater(int(row["objective_evaluations"]), 0)
            self.assertNotIn("optimizer_success", row)

    def test_historical_pilot_completes(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            result = self.run_script(
                "run_pilot.py", output,
                "--instances", "4", "--n", "4", "--k", "2", "--grid-size", "5",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.read_rows(output / "pilot_results.csv")
            self.assertEqual(len(rows), 12)
            self.assert_historical_semantics(rows)
            summary = json.loads((output / "pilot_summary.json").read_text())
            self.assertEqual(summary["configuration"]["instances"], 4)
            self.assertTrue((output / "pilot_report.md").is_file())
            self.assertTrue((output / "pilot_quality_resource.png").is_file())

    def test_historical_dks_completes(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dks"
            result = self.run_script(
                "run_independent_dks_benchmark.py", output,
                "--instances", "4", "--n", "4", "--k", "2", "--edge-budget", "4",
                "--random-replicates", "1", "--grid-size", "5",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.read_rows(output / "independent_dks_benchmark_v1_raw.csv")
            self.assertEqual(len(rows), 16)
            self.assert_historical_semantics(rows)
            self.assertEqual(len(self.read_rows(output / "independent_dks_benchmark_v1_instances.csv")), 4)

    def test_constant_dks_retains_raw_rows_and_refuses_summary(self) -> None:
        # Every k=1 DKS cost is zero, regardless of sampled graph weights.
        with TemporaryDirectory() as directory:
            output = Path(directory) / "constant"
            result = self.run_script(
                "run_independent_dks_benchmark.py", output,
                "--instances", "4", "--n", "4", "--k", "1", "--edge-budget", "4",
                "--random-replicates", "1", "--grid-size", "5",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("4 of 4 instances", result.stderr)
            rows = self.read_rows(output / "independent_dks_benchmark_v1_raw.csv")
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["normalized_gap"] == "" for row in rows))
            self.assertTrue(all(row["cost_status"] == "constant_or_unresolved" for row in rows))
            self.assertFalse((output / "independent_dks_benchmark_v1_instances.csv").exists())

    def test_alignment_rejects_unmatched_ring_budget(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "alignment.csv"
            result = self.run_script(
                "run_alignment_control.py", output,
                "--instances", "0", "--n", "4", "--k", "2", "--edge-budget", "5",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("actual ring edge count (4)", result.stderr)
            self.assertFalse(output.exists())

    def test_dks_requires_a_random_comparator(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dks"
            result = self.run_script(
                "run_independent_dks_benchmark.py", output,
                "--instances", "4", "--random-replicates", "0",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--random-replicates must be at least 1", result.stderr)
            self.assertFalse(output.exists())

    def test_alignment_records_actual_edges_and_three_conditions(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "alignment.csv"
            result = self.run_script(
                "run_alignment_control.py", output,
                "--instances", "1", "--n", "4", "--k", "2", "--edge-budget", "4",
                "--random-replicates", "1", "--grid-size", "5",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = self.read_rows(output)
            self.assertEqual(len(rows), 9)
            self.assertEqual({row["initialization"] for row in rows}, {"uniform", "mixer_low", "mixer_high"})
            self.assertEqual({row["mixer_edges"] for row in rows}, {"4"})
            self.assertEqual({row["gamma_coordinate"] for row in rows}, {"u"})
            self.assertEqual({row["gamma_scale"] for row in rows}, {"feasible_span"})


if __name__ == "__main__":
    unittest.main()
