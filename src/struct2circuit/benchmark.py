"""Deterministic benchmark manifests with guarded blind-split access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .problems import (
    CardinalityQUBO,
    block_correlated_qubo,
    weak_structure_null_qubo,
    weighted_densest_k_subgraph_qubo,
    weighted_max_k_vertex_cover_qubo,
)


SPLITS = ("train", "validation", "blind_test", "transfer")
FAMILIES = ("block_correlated", "densest_k_subgraph", "max_k_vertex_cover", "weak_structure_null")
STATUSES = ("DRAFT_UNFROZEN", "FROZEN")
GENERATOR_VERSIONS = {
    "block_correlated": "block_correlated_v1",
    "densest_k_subgraph": "weighted_densest_k_subgraph_v1",
    "max_k_vertex_cover": "weighted_max_k_vertex_cover_v1",
    "weak_structure_null": "weak_structure_null_v1",
}


def canonical_json(value: Any) -> str:
    """Return the canonical, byte-stable JSON representation used for hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "benchmark_version", "status", "master_seed", "random_baseline_replicates",
        "normalization", "exact_solver_limit_n", "splits",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing configuration fields: {sorted(missing)}")
    if not isinstance(config["benchmark_version"], str) or not config["benchmark_version"]:
        raise ValueError("benchmark_version must be a nonempty string")
    if config["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    for field in ("master_seed", "random_baseline_replicates", "exact_solver_limit_n"):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if config["random_baseline_replicates"] < 1:
        raise ValueError("random_baseline_replicates must be positive")
    if not isinstance(config["normalization"], str) or not config["normalization"]:
        raise ValueError("normalization must be a nonempty string")
    if set(config["splits"]) != set(SPLITS):
        raise ValueError(f"splits must be exactly {SPLITS}")
    for split, entries in config["splits"].items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"split {split} must contain entries")
        for entry in entries:
            if set(entry) != {"family", "n", "k", "count", "parameters"}:
                raise ValueError(f"invalid entry schema in split {split}")
            if entry["family"] not in FAMILIES:
                raise ValueError(f"unknown family {entry['family']}")
            n, k, count = entry["n"], entry["k"], entry["count"]
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (n, k, count)):
                raise ValueError("n, k, and count must be integers")
            if not 0 < k < n or count < 1:
                raise ValueError("entries require 0 < k < n and count >= 1")
            if not isinstance(entry["parameters"], dict):
                raise ValueError("parameters must be an object")


def _derived_seed(master_seed: int, split: str, family: str, entry: int, replicate: int) -> int:
    digest = hashlib.sha256(
        f"struct2circuit|{master_seed}|{split}|{family}|{entry}|{replicate}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _baseline_seed(generator_seed: int, baseline_replicate: int) -> int:
    digest = hashlib.sha256(
        f"struct2circuit|random_baseline|{generator_seed}|{baseline_replicate}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Build a canonically ordered manifest without generating or solving instances."""
    validate_config(config)
    records: list[dict[str, Any]] = []
    all_seeds: set[int] = set()
    for split in SPLITS:
        for entry_index, entry in enumerate(config["splits"][split]):
            for replicate in range(entry["count"]):
                seed = _derived_seed(
                    config["master_seed"], split, entry["family"], entry_index, replicate
                )
                if seed in all_seeds:
                    raise RuntimeError("derived generator seed collision")
                all_seeds.add(seed)
                baseline_seeds = [
                    _baseline_seed(seed, baseline_replicate)
                    for baseline_replicate in range(config["random_baseline_replicates"])
                ]
                if len(set(baseline_seeds)) != len(baseline_seeds):
                    raise RuntimeError("derived random-baseline seed collision")
                if all_seeds.intersection(baseline_seeds):
                    raise RuntimeError("generator and random-baseline seeds must be disjoint")
                all_seeds.update(baseline_seeds)
                generator_version = GENERATOR_VERSIONS[entry["family"]]
                identity = {
                    "benchmark_version": config["benchmark_version"],
                    "split": split,
                    "family": entry["family"],
                    "n": entry["n"],
                    "k": entry["k"],
                    "generator_seed": seed,
                    "generator_parameters": entry["parameters"],
                    "generator_version": generator_version,
                }
                record = {
                    "benchmark_version": config["benchmark_version"],
                    "benchmark_status": config["status"],
                    "instance_id": f"s2c-{_checksum(identity)[:20]}",
                    "split": split,
                    "family": entry["family"],
                    "n": entry["n"],
                    "k": entry["k"],
                    "generator_seed": seed,
                    "generator_parameters": entry["parameters"],
                    "generator_version": generator_version,
                    "random_baseline_seeds": baseline_seeds,
                    "random_baseline_seed_namespace": (
                        f"{config['benchmark_version']}:{split}:{entry['family']}:{seed}:"
                        f"replicates={config['random_baseline_replicates']}"
                    ),
                }
                record["record_checksum"] = _checksum(record)
                records.append(record)
    records.sort(key=lambda item: (SPLITS.index(item["split"]), item["family"], item["n"], item["k"], item["instance_id"]))
    if len({record["instance_id"] for record in records}) != len(records):
        raise RuntimeError("duplicate instance ID")
    body = {
        "benchmark_version": config["benchmark_version"],
        "status": config["status"],
        "config_checksum": _checksum(config),
        "records": records,
    }
    body["manifest_checksum"] = _checksum(body)
    return body


def write_manifest(config: dict[str, Any], path: Path) -> dict[str, Any]:
    """Write a manifest, enforcing version changes for an existing frozen target."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "FROZEN"
            and existing.get("config_checksum") != _checksum(config)
            and existing.get("benchmark_version") == config.get("benchmark_version")
        ):
            raise ValueError("changing a frozen configuration requires a new benchmark_version")
    manifest = build_manifest(config)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: Path, *, allow_blind: bool = False) -> dict[str, Any]:
    """Load and verify a manifest, excluding blind records unless authorized."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_checksum", None)
    actual = _checksum(manifest)
    if expected != actual:
        raise ValueError("manifest checksum mismatch")
    manifest["manifest_checksum"] = expected
    for record in manifest["records"]:
        payload = {key: value for key, value in record.items() if key != "record_checksum"}
        if record.get("record_checksum") != _checksum(payload):
            raise ValueError(f"record checksum mismatch: {record.get('instance_id')}")
    if not allow_blind:
        manifest["records"] = [record for record in manifest["records"] if record["split"] != "blind_test"]
    return manifest


def require_new_version_for_frozen_change(old: dict[str, Any], new: dict[str, Any]) -> None:
    """Reject edits to a frozen configuration that reuse its version label."""
    validate_config(old)
    validate_config(new)
    if old["status"] == "FROZEN" and _checksum(old) != _checksum(new):
        if old["benchmark_version"] == new["benchmark_version"]:
            raise ValueError("changing a frozen configuration requires a new benchmark_version")


def record_to_qubo(record: dict[str, Any], *, allow_blind: bool = False) -> CardinalityQUBO:
    """Reconstruct one manifest record, with explicit procedural blind access."""
    if record.get("split") == "blind_test" and not allow_blind:
        raise PermissionError("blind-test reconstruction requires allow_blind=True")
    family = record.get("family")
    if family not in GENERATOR_VERSIONS:
        raise ValueError(f"unknown family {family}")
    if record.get("generator_version") != GENERATOR_VERSIONS[family]:
        raise ValueError("generator version does not match the installed implementation")
    parameters = dict(record["generator_parameters"])
    parameters.pop("regime", None)
    n, k, seed = record["n"], record["k"], record["generator_seed"]
    if family == "block_correlated":
        return block_correlated_qubo(n, k, seed, **parameters)
    if family == "densest_k_subgraph":
        return weighted_densest_k_subgraph_qubo(n, k, seed=seed, **parameters)
    if family == "max_k_vertex_cover":
        return weighted_max_k_vertex_cover_qubo(n, k, seed=seed, **parameters)
    parameters.pop("coefficient_distribution", None)
    parameters.pop("scale_normalization", None)
    return weak_structure_null_qubo(n, k, seed=seed, **parameters)
