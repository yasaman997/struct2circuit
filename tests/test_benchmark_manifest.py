from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.benchmark import (  # noqa: E402
    build_manifest,
    load_manifest,
    require_new_version_for_frozen_change,
    validate_config,
    write_manifest,
)


class BenchmarkManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs" / "benchmark_v1_draft.json").read_text())

    def test_records_and_seeds_are_unique_and_splits_disjoint(self) -> None:
        records = build_manifest(self.config)["records"]
        self.assertEqual(len(records), 368)
        self.assertEqual(len({record["instance_id"] for record in records}), len(records))
        self.assertEqual(len({record["generator_seed"] for record in records}), len(records))
        split_ids = {
            split: {record["instance_id"] for record in records if record["split"] == split}
            for split in self.config["splits"]
        }
        for left, left_ids in split_ids.items():
            for right, right_ids in split_ids.items():
                if left != right:
                    self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_regeneration_is_byte_identical(self) -> None:
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_manifest(self.config, first)
            write_manifest(self.config, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_checked_in_manifest_is_current(self) -> None:
        expected = build_manifest(self.config)
        actual = json.loads((ROOT / "manifests" / "benchmark_v1_draft.json").read_text())
        self.assertEqual(actual, expected)

    def test_blind_records_require_explicit_access(self) -> None:
        path = ROOT / "manifests" / "benchmark_v1_draft.json"
        default = load_manifest(path)
        authorized = load_manifest(path, allow_blind=True)
        self.assertNotIn("blind_test", {record["split"] for record in default["records"]})
        self.assertEqual(sum(record["split"] == "blind_test" for record in authorized["records"]), 128)

    def test_record_and_manifest_checksum_guards(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(self.config, path)
            changed = json.loads(path.read_text())
            changed["records"][0]["n"] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                load_manifest(path)

    def test_schema_validation(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["splits"]["train"][0]["k"] = invalid["splits"]["train"][0]["n"]
        with self.assertRaisesRegex(ValueError, "0 < k < n"):
            validate_config(invalid)

    def test_frozen_change_requires_new_version(self) -> None:
        old = copy.deepcopy(self.config)
        old["status"] = "FROZEN"
        changed = copy.deepcopy(old)
        changed["master_seed"] += 1
        with self.assertRaisesRegex(ValueError, "new benchmark_version"):
            require_new_version_for_frozen_change(old, changed)
        changed["benchmark_version"] = "benchmark_v2"
        require_new_version_for_frozen_change(old, changed)


if __name__ == "__main__":
    unittest.main()
