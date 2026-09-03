"""Deterministic benchmark manifests with guarded blind-split access."""

from __future__ import annotations

import hashlib
import json
import math
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
GENERATOR_PARAMETERS = {
    "block_correlated": frozenset(
        {"n_blocks", "block_strength", "cross_strength", "linear_scale"}
    ),
    "densest_k_subgraph": frozenset(
        {"density", "weight_distribution", "weight_range", "planted_community_strength"}
    ),
    "max_k_vertex_cover": frozenset(
        {
            "density", "weight_distribution", "weight_range",
            "community_strength", "hub_strength",
        }
    ),
    "weak_structure_null": frozenset({"density", "coefficient_scale"}),
}
AUDIT_PARAMETERS = {
    "block_correlated": frozenset({"regime"}),
    "densest_k_subgraph": frozenset({"regime"}),
    "max_k_vertex_cover": frozenset({"regime"}),
    "weak_structure_null": frozenset(
        {"regime", "coefficient_distribution", "scale_normalization"}
    ),
}
RECORD_FIELDS = frozenset(
    {
        "benchmark_version", "benchmark_status", "instance_id", "split", "family",
        "n", "k", "generator_seed", "generator_parameters", "generator_version",
        "random_baseline_seeds", "random_baseline_seed_namespace", "record_checksum",
    }
)


def canonical_json(value: Any) -> str:
    """Return the canonical, byte-stable JSON representation used for hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_family_parameters(family: str, parameters: Any, *, n: int) -> None:
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    required = GENERATOR_PARAMETERS[family] | AUDIT_PARAMETERS[family]
    missing = required - parameters.keys()
    unexpected = parameters.keys() - required
    if missing:
        raise ValueError(f"missing parameters for {family}: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected parameters for {family}: {sorted(unexpected)}")
    if not isinstance(parameters["regime"], str) or not parameters["regime"]:
        raise ValueError("regime must be a nonempty audit label")
    if family == "block_correlated":
        if not isinstance(parameters["n_blocks"], int) or isinstance(parameters["n_blocks"], bool):
            raise ValueError("n_blocks must be an integer")
        if not 1 <= parameters["n_blocks"] <= n:
            raise ValueError("n_blocks must lie in [1, n]")
        for key in ("block_strength", "cross_strength", "linear_scale"):
            value = parameters[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if parameters["block_strength"] <= 0 or parameters["cross_strength"] < 0 or parameters["linear_scale"] <= 0:
            raise ValueError("block and linear strengths must be positive; cross_strength must be nonnegative")
    elif family in {"densest_k_subgraph", "max_k_vertex_cover"}:
        density = parameters["density"]
        if not isinstance(density, (int, float)) or isinstance(density, bool) or not math.isfinite(density) or not 0 <= density <= 1:
            raise ValueError("density must lie in [0, 1]")
        if parameters["weight_distribution"] not in {"uniform", "log_uniform"}:
            raise ValueError("unsupported weight_distribution")
        weight_range = parameters["weight_range"]
        if not isinstance(weight_range, (list, tuple)) or len(weight_range) != 2:
            raise ValueError("weight_range must contain two values")
        low, high = weight_range
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in weight_range) or low <= 0 or high < low:
            raise ValueError("weight_range must satisfy 0 < low <= high")
        strength_keys = (
            ("planted_community_strength",)
            if family == "densest_k_subgraph"
            else ("community_strength", "hub_strength")
        )
        for key in strength_keys:
            value = parameters[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
    if family == "weak_structure_null":
        density, scale = parameters["density"], parameters["coefficient_scale"]
        if not isinstance(density, (int, float)) or isinstance(density, bool) or not math.isfinite(density) or not 0 <= density <= 1:
            raise ValueError("density must lie in [0, 1]")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("coefficient_scale must be positive and finite")
        if parameters["coefficient_distribution"] != "zero_mean_gaussian":
            raise ValueError("unsupported null coefficient_distribution")
        if parameters["scale_normalization"] != "sqrt(max(1,density*(n-1)))":
            raise ValueError("unsupported null scale_normalization")


def _generator_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_version": record["benchmark_version"],
        "split": record["split"],
        "family": record["family"],
        "n": record["n"],
        "k": record["k"],
        "generator_seed": record["generator_seed"],
        "generator_parameters": record["generator_parameters"],
        "generator_version": record["generator_version"],
    }


def validate_record(record: Any) -> None:
    """Validate a complete record, including checksums and definition identity."""
    if not isinstance(record, dict):
        raise ValueError("manifest record must be an object")
    missing = RECORD_FIELDS - record.keys()
    unexpected = record.keys() - RECORD_FIELDS
    if missing:
        raise ValueError(f"missing manifest record fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected manifest record fields: {sorted(unexpected)}")
    if record["family"] not in FAMILIES:
        raise ValueError(f"unknown family {record['family']}")
    if record["split"] not in SPLITS:
        raise ValueError(f"unknown split {record['split']}")
    if record["benchmark_status"] not in STATUSES:
        raise ValueError("invalid benchmark status")
    if not isinstance(record["benchmark_version"], str) or not record["benchmark_version"]:
        raise ValueError("benchmark_version must be a nonempty string")
    n, k, seed = record["n"], record["k"], record["generator_seed"]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (n, k, seed)):
        raise ValueError("record n, k, and generator_seed must be integers")
    if not 0 < k < n or seed < 0:
        raise ValueError("record requires 0 < k < n and a nonnegative generator_seed")
    _validate_family_parameters(record["family"], record["generator_parameters"], n=n)
    if record["generator_version"] != GENERATOR_VERSIONS[record["family"]]:
        raise ValueError("generator version does not match the installed implementation")
    baseline_seeds = record["random_baseline_seeds"]
    if (
        not isinstance(baseline_seeds, list)
        or not baseline_seeds
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in baseline_seeds)
        or len(set(baseline_seeds)) != len(baseline_seeds)
        or seed in baseline_seeds
    ):
        raise ValueError("invalid random_baseline_seeds")
    if not isinstance(record["random_baseline_seed_namespace"], str):
        raise ValueError("random_baseline_seed_namespace must be a string")
    payload = {key: value for key, value in record.items() if key != "record_checksum"}
    if record["record_checksum"] != _checksum(payload):
        raise ValueError(f"record checksum mismatch: {record['instance_id']}")
    expected_id = f"s2c-{_checksum(_generator_identity(record))[:20]}"
    if record["instance_id"] != expected_id:
        raise ValueError("instance_id does not match the generator definition")


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
            _validate_family_parameters(entry["family"], entry["parameters"], n=n)


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
                record = {
                    "benchmark_version": config["benchmark_version"],
                    "benchmark_status": config["status"],
                    "instance_id": "",
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
                record["instance_id"] = f"s2c-{_checksum(_generator_identity(record))[:20]}"
                record["record_checksum"] = _checksum(record)
                validate_record(record)
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
        validate_record(record)
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
    validate_record(record)
    family = record["family"]
    parameters = {
        key: value
        for key, value in record["generator_parameters"].items()
        if key in GENERATOR_PARAMETERS[family]
    }
    n, k, seed = record["n"], record["k"], record["generator_seed"]
    if family == "block_correlated":
        problem = block_correlated_qubo(n, k, seed, **parameters)
    elif family == "densest_k_subgraph":
        problem = weighted_densest_k_subgraph_qubo(n, k, seed=seed, **parameters)
    elif family == "max_k_vertex_cover":
        problem = weighted_max_k_vertex_cover_qubo(n, k, seed=seed, **parameters)
    else:
        problem = weak_structure_null_qubo(n, k, seed=seed, **parameters)
    if (problem.n, problem.k) != (n, k):
        raise RuntimeError("reconstructed QUBO size or cardinality does not match record")
    metadata = problem.metadata or {}
    if metadata.get("seed") != seed or metadata.get("generator_version") != record["generator_version"]:
        raise RuntimeError("reconstructed QUBO generator identity does not match record")
    for key, expected in parameters.items():
        if canonical_json(metadata.get(key)) != canonical_json(expected):
            raise RuntimeError(f"reconstructed QUBO parameter does not match record: {key}")
    if family == "weak_structure_null":
        audit = record["generator_parameters"]
        if metadata.get("coefficient_distribution") != audit["coefficient_distribution"]:
            raise RuntimeError("reconstructed null coefficient distribution does not match record")
        if metadata.get("scale_normalization") != audit["scale_normalization"]:
            raise RuntimeError("reconstructed null scale normalization does not match record")
    return problem
