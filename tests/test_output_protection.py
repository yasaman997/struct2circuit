from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.outputs import preflight_outputs


PILOT = ("pilot_results.csv", "pilot_summary.json", "pilot_report.md",
         "pilot_quality_resource.png")
DKS = ("independent_dks_benchmark_v1_raw.csv",
       "independent_dks_benchmark_v1_instances.csv")


def load_script(filename):
    spec = importlib.util.spec_from_file_location("output_test_" + Path(filename).stem,
                                                 ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(directory):
    return {str(p.relative_to(directory)): (p.is_symlink(), p.is_dir(),
            p.read_bytes() if p.is_file() else None) for p in directory.rglob("*")}


class OutputProtectionTests(unittest.TestCase):
    def assert_early_collision(self, filename, output, destinations, *, use_default=False,
                               refusal="Refusing to overwrite"):
        module = load_script(filename)
        generator = ("block_correlated_qubo" if filename == "run_pilot.py"
                     else "weighted_densest_k_subgraph_qubo")
        argv = [str(module.__file__)]
        if not use_default:
            argv += ["--output", str(output)]
        # Abort even the old implementation before it can simulate or write.
        with patch.object(sys, "argv", argv), patch.object(
            module, generator, side_effect=SystemExit("SIMULATION_STARTED")
        ) as create_problem, patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(SystemExit) as caught:
                module.main()
            self.assertIn(refusal, str(caught.exception))
            self.assertIn("--output", str(caught.exception))
            for destination in destinations:
                self.assertIn(str(destination), str(caught.exception))
            create_problem.assert_not_called()
            mkdir.assert_not_called()

    def test_fresh_destinations_allow_existing_unrelated_files_without_creating_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            note = root / "notes.txt"
            note.write_text("keep me")
            preflight_outputs(root / name for name in PILOT)
            preflight_outputs((root / "new-run" / "result.csv",))
            self.assertEqual(note.read_text(), "keep me")
            self.assertEqual(list(root.iterdir()), [note])

    def test_preflight_reports_all_files_directories_and_dangling_links(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "old.csv"
            file.write_bytes(b"preserve")
            folder = root / "old.json"
            folder.mkdir()
            link = root / "old.png"
            link.symlink_to(root / "missing-target")
            file_link = root / "file-link"
            file_link.symlink_to(file)
            hardlink = root / "hardlink"
            hardlink.hardlink_to(file)
            before = snapshot(root)
            occupied = (file, folder, link, file_link, hardlink)
            with self.assertRaises(FileExistsError) as caught:
                preflight_outputs((root / "fresh.csv", *occupied))
            for destination in occupied:
                self.assertIn(str(destination), str(caught.exception))
            self.assertEqual(snapshot(root), before)

    def test_scripts_reject_missing_parent_traversal_before_mkdir_or_simulation(self):
        scripts = (("run_pilot.py", PILOT),
                   ("run_independent_dks_benchmark.py", DKS),
                   ("run_alignment_control.py", ("alignment.csv",)))
        for filename, names in scripts:
            for traversal in ("not-created/..", "outer/inner/../..",
                              "not-created/../missing-again/.."):
                with self.subTest(script=filename, path=traversal), TemporaryDirectory() as directory:
                    root = Path(directory)
                    for name in names:
                        (root / name).write_bytes(b"historical sentinel\x00\xff\n")
                    alias = root / traversal
                    destinations = tuple(alias / name for name in names)
                    output = destinations[0] if filename == "run_alignment_control.py" else alias
                    before = snapshot(root)
                    self.assert_early_collision(
                        filename, output, destinations,
                        refusal="Refusing output paths containing '..' components",
                    )
                    self.assertEqual(snapshot(root), before)

    def test_traversal_is_rejected_even_without_existing_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (root / "missing" / ".." / "fresh.csv", Path("..") / "fresh.csv"):
                with self.subTest(path=path), self.assertRaises(FileExistsError) as caught:
                    preflight_outputs(iter((root / "safe.csv", path)))
                self.assertIn(str(path), str(caught.exception))
                self.assertIn("without parent traversal", str(caught.exception))
            self.assertEqual(snapshot(root), {})

    def test_scripts_preserve_dangling_links_and_symlinked_parent_collisions(self):
        for filename, name in (("run_pilot.py", PILOT[0]),
                               ("run_independent_dks_benchmark.py", DKS[0]),
                               ("run_alignment_control.py", "alignment.csv")):
            for kind in ("dangling", "parent-alias"):
                with self.subTest(script=filename, kind=kind), TemporaryDirectory() as directory:
                    root = Path(directory)
                    real = root / "results"
                    real.mkdir()
                    if kind == "dangling":
                        (real / name).symlink_to(root / "missing-target")
                        output_dir = real
                    else:
                        (real / name).write_bytes(b"preserve through parent alias\x00\xff")
                        output_dir = root / "alias"
                        output_dir.symlink_to(real, target_is_directory=True)
                    destination = output_dir / name
                    output = destination if filename == "run_alignment_control.py" else output_dir
                    before = snapshot(root)
                    self.assert_early_collision(filename, output, (destination,))
                    self.assertEqual(snapshot(root), before)

    def test_fresh_nested_and_symlinked_outputs_reach_generation(self):
        for filename in ("run_pilot.py", "run_independent_dks_benchmark.py",
                         "run_alignment_control.py"):
            for parent_alias in (False, True):
                with self.subTest(script=filename, alias=parent_alias), TemporaryDirectory() as directory:
                    root = Path(directory)
                    note = root / "sentinel.txt"
                    note.write_bytes(b"unrelated sentinel\x00\xff")
                    parent = root
                    if parent_alias:
                        parent = root / "alias"
                        parent.symlink_to(root, target_is_directory=True)
                    output_dir = parent / "fresh" / "nested"
                    output = (output_dir / "alignment.csv"
                              if filename == "run_alignment_control.py" else output_dir)
                    module = load_script(filename)
                    generator = ("block_correlated_qubo" if filename == "run_pilot.py"
                                 else "weighted_densest_k_subgraph_qubo")
                    with patch.object(sys, "argv", [module.__file__, "--output", str(output)]), patch.object(
                        module, generator, side_effect=SystemExit("SIMULATION_STARTED")
                    ) as create_problem:
                        with self.assertRaisesRegex(SystemExit, "^SIMULATION_STARTED$"):
                            module.main()
                        create_problem.assert_called_once()
                    self.assertTrue(output_dir.is_dir())
                    self.assertEqual(list(output_dir.iterdir()), [])
                    self.assertEqual(note.read_bytes(), b"unrelated sentinel\x00\xff")

    def test_each_pilot_destination_rejects_before_simulation_and_partial_writes(self):
        for name in PILOT:
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "notes.txt").write_bytes(b"unrelated")
                destination = root / name
                destination.write_bytes(b"historical")
                before = snapshot(root)
                self.assert_early_collision("run_pilot.py", root, (destination,))
                self.assertEqual(snapshot(root), before)

    def test_each_dks_destination_rejects_before_simulation_and_partial_writes(self):
        for name in DKS:
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "notes.txt").write_bytes(b"unrelated")
                destination = root / name
                destination.write_bytes(b"historical")
                before = snapshot(root)
                self.assert_early_collision("run_independent_dks_benchmark.py", root, (destination,))
                self.assertEqual(snapshot(root), before)

    def test_alignment_rejects_existing_output_before_simulation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "alignment.csv"
            destination.write_bytes(b"previous diagnostic")
            before = snapshot(root)
            self.assert_early_collision("run_alignment_control.py", destination, (destination,))
            self.assertEqual(snapshot(root), before)

    def test_scripts_report_all_expected_collisions_together(self):
        for filename, names in (("run_pilot.py", PILOT),
                                ("run_independent_dks_benchmark.py", DKS)):
            with self.subTest(filename=filename), TemporaryDirectory() as directory:
                root = Path(directory)
                destinations = tuple(root / name for name in names)
                for destination in destinations:
                    destination.write_bytes(b"original")
                before = snapshot(root)
                self.assert_early_collision(filename, root, destinations)
                self.assertEqual(snapshot(root), before)

    def test_canonical_artifacts_and_directory_aliases_are_protected(self):
        results = ROOT / "results"
        before = snapshot(results)
        for filename, names in (("run_pilot.py", PILOT),
                                ("run_independent_dks_benchmark.py", DKS)):
            existing = tuple(results / name for name in names if (results / name).exists())
            self.assertTrue(existing)
            self.assert_early_collision(filename, results, existing, use_default=True)
            with TemporaryDirectory() as directory:
                alias = Path(directory) / "historical-alias"
                alias.symlink_to(results, target_is_directory=True)
                self.assert_early_collision(filename, alias,
                                            tuple(alias / p.name for p in existing))
        self.assertEqual(snapshot(results), before)


if __name__ == "__main__":
    unittest.main()
