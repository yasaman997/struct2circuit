from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.benchmark import (  # noqa: E402
    build_manifest,
    canonical_json,
    load_manifest,
    record_to_qubo,
    require_new_version_for_frozen_change,
    validate_config,
    write_manifest,
)
from struct2circuit import benchmark as benchmark_module  # noqa: E402


class BenchmarkManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs" / "benchmark_v1_draft.json").read_text())

    def test_records_and_seeds_are_unique_and_splits_disjoint(self) -> None:
        records = build_manifest(self.config)["records"]
        self.assertEqual(len(records), 368)
        self.assertEqual(len({record["instance_id"] for record in records}), len(records))
        self.assertEqual(len({record["generator_seed"] for record in records}), len(records))
        generator_seeds = {record["generator_seed"] for record in records}
        baseline_seeds = [seed for record in records for seed in record["random_baseline_seeds"]]
        self.assertEqual(len(baseline_seeds), 368 * self.config["random_baseline_replicates"])
        self.assertEqual(len(set(baseline_seeds)), len(baseline_seeds))
        self.assertTrue(generator_seeds.isdisjoint(baseline_seeds))
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

    def test_stale_record_checksum_with_recomputed_manifest_checksum(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(self.config, path)
            changed = json.loads(path.read_text())
            changed["records"][0]["n"] += 1
            changed.pop("manifest_checksum")
            changed["manifest_checksum"] = hashlib.sha256(
                canonical_json(changed).encode("utf-8")
            ).hexdigest()
            path.write_text(canonical_json(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "record checksum mismatch"):
                load_manifest(path)

    def test_instance_id_covers_complete_generator_definition(self) -> None:
        original = build_manifest(self.config)["records"]
        mutations = []
        for field, value in (("n", 9), ("k", 2)):
            changed = copy.deepcopy(self.config)
            changed["splits"]["train"][0][field] = value
            mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["splits"]["train"][0]["parameters"]["block_strength"] = 0.66
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["master_seed"] += 1
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed["splits"]["train"][0]):
                updated = build_manifest(changed)["records"]
                original_ids = {record["instance_id"] for record in original}
                updated_ids = {record["instance_id"] for record in updated}
                self.assertNotEqual(original_ids, updated_ids)
        with patch.dict(
            benchmark_module.GENERATOR_VERSIONS,
            {"block_correlated": "block_correlated_v2"},
        ):
            version_updated_ids = {
                record["instance_id"] for record in build_manifest(self.config)["records"]
            }
        self.assertNotEqual({record["instance_id"] for record in original}, version_updated_ids)

    def test_non_blind_records_reconstruct_to_qubos(self) -> None:
        records = build_manifest(self.config)["records"]
        representatives = {}
        for record in records:
            if record["split"] != "blind_test":
                representatives.setdefault(record["family"], record)
        self.assertEqual(set(representatives), set(self.config["splits"]["train"][i]["family"] for i in range(0, 8, 2)))
        for family, record in representatives.items():
            with self.subTest(family=family):
                problem = record_to_qubo(record)
                self.assertEqual((problem.n, problem.k), (record["n"], record["k"]))

    def test_blind_reconstruction_is_guarded_without_instantiation(self) -> None:
        blind_record = next(
            record for record in build_manifest(self.config)["records"] if record["split"] == "blind_test"
        )
        with self.assertRaisesRegex(PermissionError, "allow_blind"):
            record_to_qubo(blind_record)

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

    def test_write_path_enforces_frozen_version_change(self) -> None:
        frozen = copy.deepcopy(self.config)
        frozen["status"] = "FROZEN"
        changed = copy.deepcopy(frozen)
        changed["master_seed"] += 1
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(frozen, path)
            with self.assertRaisesRegex(ValueError, "new benchmark_version"):
                write_manifest(changed, path)
            changed["benchmark_version"] = "benchmark_v2"
            write_manifest(changed, path)


if __name__ == "__main__":
    unittest.main()
