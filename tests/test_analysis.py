from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.analysis import (  # noqa: E402
    plot_pilot,
    require_defined_gaps,
    save_summary,
    summarize_pilot,
)


class UndefinedGapAggregationTests(unittest.TestCase):
    def records(self, instances: int = 2) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "instance": instance, "mixer": mixer, "normalized_gap": 0.2,
                "probability_optimum": 0.5, "mixer_edges": 4 if mixer != "complete" else 6,
            }
            for instance in range(instances)
            for mixer in ("ring", "structure", "complete")
        ])

    def test_defined_gaps_are_accepted(self) -> None:
        require_defined_gaps(self.records(), "mixer", expected_instances=range(2))

    def test_complete_pilot_summary_is_accepted(self) -> None:
        summary = summarize_pilot(self.records(), {"instances": 2})
        self.assertEqual(summary["paired_structure_vs_ring"]["ties"], 2)
        for mixer in ("ring", "structure", "complete"):
            self.assertEqual(summary["by_mixer"][mixer]["mean_probability_optimum"], 0.5)

    def test_pilot_requires_declared_instance_count(self) -> None:
        for config in ({}, {"instances": 0}, {"instances": True}):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "declare a positive integer"):
                    summarize_pilot(self.records(), config)

    def test_entire_expected_instance_missing_refuses_summary(self) -> None:
        records = self.records(instances=24)
        records = records[records["instance"] != 23]
        unchanged = records.copy(deep=True)
        with self.assertRaisesRegex(
            ValueError, "1 of 24 instances.*1 expected instances are completely missing"
        ):
            summarize_pilot(records, {"instances": 24})
        pd.testing.assert_frame_equal(records, unchanged)

    def test_gap_guard_checks_declared_ids_not_observed_count(self) -> None:
        records = self.records()
        records.loc[records["instance"] == 1, "instance"] = 2
        with self.assertRaisesRegex(
            ValueError, "1 of 2 expected instances are completely missing; 1 unexpected"
        ):
            require_defined_gaps(records, "mixer", expected_instances=(0, 1))

    def test_gap_guard_supports_nonconsecutive_expected_ids(self) -> None:
        records = self.records()
        records["instance"] = records["instance"].map({0: "seed-a", 1: "seed-z"})
        require_defined_gaps(records, "mixer", expected_instances=("seed-a", "seed-z"))

    def test_empty_observed_study_reports_all_expected_instances_missing(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "2 of 2 instances.*2 expected instances are completely missing"
        ):
            summarize_pilot(self.records().iloc[:0], {"instances": 2})

    def test_undefined_gap_refuses_whole_summary_with_instance_count(self) -> None:
        for missing in (None, np.nan, np.inf, -np.inf):
            with self.subTest(missing=missing):
                records = self.records()
                records.loc[0, "normalized_gap"] = missing
                with self.assertRaisesRegex(ValueError, "1 of 2 instances"):
                    summarize_pilot(records, {"instances": 2})
                self.assertEqual(len(records), 6)

    def test_undefined_probability_for_each_summarized_method_refuses_summary(self) -> None:
        for mixer in ("structure", "ring", "complete"):
            for missing in (None, np.nan, np.inf, -np.inf):
                with self.subTest(mixer=mixer, missing=missing):
                    records = self.records(instances=24)
                    mask = (records["instance"] == 23) & (records["mixer"] == mixer)
                    records.loc[mask, "probability_optimum"] = missing
                    unchanged = records.copy(deep=True)
                    with self.assertRaisesRegex(
                        ValueError, "probability_optimum: 1 of 24 instances"
                    ):
                        summarize_pilot(records, {"instances": 24})
                    pd.testing.assert_frame_equal(records, unchanged)

    def test_absent_probability_column_refuses_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability_optimum: 2 of 2 instances"):
            summarize_pilot(
                self.records().drop(columns="probability_optimum"), {"instances": 2}
            )

    def test_constant_or_unresolved_rows_remain_unchanged_on_refusal(self) -> None:
        records = self.records()
        records["normalized_gap"] = None
        records["probability_optimum"] = None
        records["cost_status"] = "constant_or_unresolved"
        unchanged = records.copy(deep=True)
        with self.assertRaisesRegex(
            ValueError, "normalized_gap, probability_optimum: 2 of 2 instances"
        ):
            summarize_pilot(records, {"instances": 2})
        pd.testing.assert_frame_equal(records, unchanged)

    def test_explicit_unresolved_status_rejects_old_finite_metric_sentinels(self) -> None:
        records = self.records()
        records["cost_status"] = "nonconstant"
        affected = records["instance"] == 1
        records.loc[affected, "cost_status"] = "constant_or_unresolved"
        records.loc[affected, "normalized_gap"] = 0.0
        records.loc[affected, "probability_optimum"] = 1.0
        unchanged = records.copy(deep=True)
        message = "1 of 2 instances.*1 expected instances are explicitly marked constant_or_unresolved"
        with self.assertRaisesRegex(ValueError, message):
            summarize_pilot(records, {"instances": 2})
        with self.assertRaisesRegex(ValueError, message):
            require_defined_gaps(records, "mixer", expected_instances=range(2))
        pd.testing.assert_frame_equal(records, unchanged)

    def test_missing_comparator_row_refuses_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 of 2 instances"):
            require_defined_gaps(self.records().iloc[1:], "mixer", expected_instances=range(2))

    def test_entirely_absent_required_comparator_refuses_summary(self) -> None:
        records = self.records()
        records = records[records["mixer"] != "complete"]
        with self.assertRaisesRegex(ValueError, "2 of 2 instances"):
            summarize_pilot(records, {"instances": 2})

    def test_plot_refuses_undefined_gap_before_creating_output(self) -> None:
        records = self.records()
        records["normalized_gap"] = None
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            with self.assertRaisesRegex(ValueError, "2 of 2 instances"):
                plot_pilot(records, output, expected_instances=range(2))
            self.assertFalse(output.exists())

    def test_plot_refuses_entire_missing_instance_before_creating_output(self) -> None:
        records = self.records()
        records = records[records["instance"] == 0]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            with self.assertRaisesRegex(
                ValueError, "1 of 2 instances.*1 expected instances are completely missing"
            ):
                plot_pilot(records, output, expected_instances=range(2))
            self.assertFalse(output.exists())

    def test_json_uses_null_and_rejects_nonstandard_nan(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            save_summary({"normalized_gap": None}, output)
            self.assertIn("null", output.read_text())
            self.assertEqual(json.loads(output.read_text()), {"normalized_gap": None})
            with self.assertRaises(ValueError):
                save_summary({"normalized_gap": float("nan")}, output)


if __name__ == "__main__":
    unittest.main()
