"""Synthetic-only sensitivity analysis for the pre-freeze statistical design.

This module never loads benchmark manifests or constructs QUBOs.  Its experimental
unit is a synthetic paired normalized-gap difference for one hypothetical instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np
from scipy.stats import binom, norm, rankdata


FAMILY_TOTALS = (32, 48, 64, 96)
EFFECTS = (0.0, 0.005, 0.010, 0.015)
SCENARIOS = (
    "gaussian", "heavy_tailed", "skewed", "heterogeneous_regime",
    "ties", "optimizer_failure",
)


@dataclass(frozen=True)
class SensitivityConfig:
    """Controls a bounded, reproducible synthetic sensitivity run."""

    seed: int = 20260903
    alpha: float = 0.05
    bootstrap_samples: int = 399
    min_repetitions: int = 200
    max_repetitions: int = 1_000
    batch_size: int = 50
    target_mcse: float = 0.015
    runtime_cap_seconds: float = 60.0
    random_baseline_sd: float = 0.003
    failure_rate: float = 0.10
    failure_value: float = 0.0

    def validate(self) -> None:
        if not (0 < self.alpha < 0.5):
            raise ValueError("alpha must be between 0 and 0.5")
        for name in ("bootstrap_samples", "min_repetitions", "max_repetitions", "batch_size"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_repetitions > self.max_repetitions:
            raise ValueError("min_repetitions cannot exceed max_repetitions")
        if self.runtime_cap_seconds <= 0 or self.target_mcse <= 0:
            raise ValueError("runtime cap and target_mcse must be positive")
        if self.random_baseline_sd < 0 or not (0 <= self.failure_rate <= 1):
            raise ValueError("variance and failure controls are invalid")


def conservative_sign_pvalue(values: np.ndarray, margin: float = 0.0) -> float:
    """Exact one-sided sign p-value, counting ties as failures to exceed margin."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be a non-empty finite vector")
    positives = int(np.count_nonzero(arr > margin))
    return _sign_tail_table(len(arr))[positives]


@lru_cache(maxsize=None)
def _sign_tail_table(total: int) -> tuple[float, ...]:
    return tuple(float(binom.sf(positive - 1, total, 0.5)) for positive in range(total + 1))


@lru_cache(maxsize=None)
def _sign_lower_rank(total: int, alpha: float) -> int:
    admissible = [r for r in range(1, total + 1) if binom.cdf(r - 1, total, 0.5) <= alpha]
    return max(admissible) if admissible else 0


def sign_median_lower_bound(values: np.ndarray, alpha: float) -> float:
    """Distribution-free one-sided ``1-alpha`` lower bound for a median.

    The largest admissible order-statistic rank is used.  This bound and
    :func:`conservative_sign_pvalue` make the same strict-positive decision.
    """

    arr = np.sort(np.asarray(values, dtype=float))
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be a non-empty finite vector")
    if not (0 < alpha < 0.5):
        raise ValueError("alpha must be between 0 and 0.5")
    rank = _sign_lower_rank(len(arr), alpha)
    return float(arr[rank - 1]) if rank else -math.inf


def stratified_bootstrap_median(
    values: np.ndarray,
    cells: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Return stratified bootstrap replicates of the pooled sample median."""

    arr = np.asarray(values, dtype=float)
    strata = np.asarray(cells)
    if arr.ndim != 1 or strata.shape != arr.shape or len(arr) == 0:
        raise ValueError("values and cells must be aligned non-empty vectors")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    pieces = []
    for cell in np.unique(strata):
        group = arr[strata == cell]
        pieces.append(group[rng.integers(0, len(group), size=(samples, len(group)))])
    return np.median(np.concatenate(pieces, axis=1), axis=1)


def holm_rejections(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Holm step-down decisions in original order."""

    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    rejected = np.zeros(len(p), dtype=bool)
    for rank, index in enumerate(order):
        if p[index] <= alpha / (len(p) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def _wilcoxon_normal_pvalue(values: np.ndarray) -> float:
    """Fast secondary signed-rank sensitivity with conservative zero splitting."""

    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum() + 0.5 * ranks[values == 0].sum())
    n = len(values)
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(values), return_counts=True)
    variance = (n * (n + 1) * (2 * n + 1) - np.sum(counts**3 - counts)) / 24
    if variance <= 0:
        return 1.0
    return float(norm.sf((positive - mean - 0.5) / math.sqrt(variance)))


def _draw_differences(
    rng: np.random.Generator,
    total: int,
    effect: float,
    scenario: str,
    config: SensitivityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.repeat(np.arange(4), total // 4)
    if scenario == "gaussian":
        noise = rng.normal(0, 0.015, total)
    elif scenario == "heavy_tailed":
        noise = rng.standard_t(3, total) * (0.015 / math.sqrt(3))
    elif scenario == "skewed":
        noise = (rng.lognormal(-0.5, 0.8, total) - math.exp(-0.5)) * 0.014
    elif scenario == "heterogeneous_regime":
        offsets = np.asarray([-0.006, -0.002, 0.002, 0.006])
        noise = rng.normal(offsets[cells], 0.016)
    elif scenario == "ties":
        noise = rng.normal(0, 0.012, total)
    elif scenario == "optimizer_failure":
        noise = rng.normal(0, 0.015, total)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    values = effect + noise + rng.normal(0, config.random_baseline_sd, total)
    if scenario == "ties":
        values = np.round(values / 0.005) * 0.005
    return values, cells


def _population_median(effect: float, scenario: str, config: SensitivityConfig) -> float:
    """Population median implied by a synthetic location and ITT mechanism."""

    if scenario == "ties":
        return round(effect / 0.005) * 0.005
    if scenario != "optimizer_failure" or config.failure_rate == 0:
        return effect
    # Continuous component includes the two configured baseline-noise additions.
    q_below = (1 - config.failure_rate) * norm.cdf(
        (config.failure_value - effect) / math.sqrt(0.015**2 + 2 * config.random_baseline_sd**2)
    )
    if q_below <= 0.5 <= q_below + config.failure_rate:
        return config.failure_value
    target = 0.5 / (1 - config.failure_rate) if q_below > 0.5 else (0.5 - config.failure_rate) / (1 - config.failure_rate)
    return effect + math.sqrt(0.015**2 + 2 * config.random_baseline_sd**2) * float(norm.ppf(target))


def _comparison_decisions(
    values: list[np.ndarray],
    cells: np.ndarray,
    config: SensitivityConfig,
    rng: np.random.Generator,
    truths: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Return comparison rejections and compatible lower-bound decisions."""

    sign_p = np.asarray([conservative_sign_pvalue(v) for v in values])
    sign_sim = sign_p <= config.alpha / 6
    sign_lower = np.asarray([sign_median_lower_bound(v, config.alpha / 6) for v in values])
    sign_bounds = sign_lower > 0
    family_p = np.asarray([max(sign_p[2 * i:2 * i + 2]) for i in range(3)])
    family_holm = holm_rejections(family_p, config.alpha)
    # Alpha/3 family bounds are conservative simultaneous bounds compatible
    # with (never more permissive than) the Holm-IUT test decision.
    sign_hier_lower = np.asarray([
        min(sign_median_lower_bound(values[2 * i + j], config.alpha / 3) for j in range(2))
        for i in range(3)
    ])
    sign_hier_bounds = sign_hier_lower > 0

    boot_p, boot_sim_lower, boot_hier_lower = [], [], []
    for index, value in enumerate(values):
        draws = stratified_bootstrap_median(
            value, cells, samples=config.bootstrap_samples,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
        )
        boot_p.append((np.count_nonzero(draws <= 0) + 1) / (len(draws) + 1))
        boot_sim_lower.append(float(np.quantile(draws, config.alpha / 6, method="lower")))
        boot_hier_lower.append(float(np.quantile(draws, config.alpha / 3, method="lower")))
    boot_p_arr = np.asarray(boot_p)
    boot_family_p = np.asarray([max(boot_p_arr[2 * i:2 * i + 2]) for i in range(3)])

    wilcoxon_p = [_wilcoxon_normal_pvalue(value) for value in values]
    wilcoxon_family = np.asarray([max(wilcoxon_p[2 * i:2 * i + 2]) for i in range(3)])
    return {
        "sign_simultaneous": (sign_sim.reshape(3, 2).all(axis=1), sign_bounds.reshape(3, 2).all(axis=1), (sign_lower.reshape(3, 2) <= truths[:, None]).all(axis=1)),
        "sign_hierarchical_iut_holm": (family_holm, sign_hier_bounds, sign_hier_lower <= truths),
        "bootstrap_simultaneous": ((boot_p_arr <= config.alpha / 6).reshape(3, 2).all(axis=1), (np.asarray(boot_sim_lower) > 0).reshape(3, 2).all(axis=1), (np.asarray(boot_sim_lower).reshape(3, 2) <= truths[:, None]).all(axis=1)),
        "bootstrap_hierarchical_iut_holm": (holm_rejections(boot_family_p, config.alpha), (np.asarray(boot_hier_lower) > 0).reshape(3, 2).all(axis=1), (np.asarray(boot_hier_lower).reshape(3, 2) <= truths[:, None]).all(axis=1)),
        "wilcoxon_hierarchical_iut_holm_secondary": (holm_rejections(wilcoxon_family, config.alpha), np.ones(3, dtype=bool), None),
    }


def run_sensitivity(
    config: SensitivityConfig,
    *,
    family_totals: tuple[int, ...] = FAMILY_TOTALS,
    effects: tuple[float, ...] = EFFECTS,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> dict:
    """Run the adaptive, wall-clock-bounded synthetic study."""

    config.validate()
    started = time.monotonic()
    root = np.random.SeedSequence(config.seed)
    if any(total <= 0 or total % 4 for total in family_totals):
        raise ValueError("family totals must be positive and balanced across four cells")
    if any(scenario not in SCENARIOS for scenario in scenarios):
        raise ValueError("unsupported scenario")
    grid = [(n, effect, scenario) for n in family_totals for effect in effects for scenario in scenarios]
    children = root.spawn(len(grid))
    rows: list[dict] = []
    capped = False
    for (total, effect, scenario), child in zip(grid, children, strict=True):
        rng = np.random.default_rng(child)
        counts: dict[str, list[int]] = {}
        null_margin_positives = 0
        repetitions = 0
        while repetitions < config.max_repetitions:
            if time.monotonic() - started >= config.runtime_cap_seconds:
                capped = True
                break
            todo = min(config.batch_size, config.max_repetitions - repetitions)
            for _ in range(todo):
                # Three structured families, two baselines. Shared latent noise
                # induces realistic within-instance dependence between baselines.
                comparisons: list[np.ndarray] = []
                cells = np.repeat(np.arange(4), total // 4)
                locations = np.asarray([effect, 0.0, effect]) if effect > 0 else np.zeros(3)
                truths = np.asarray([_population_median(value, scenario, config) for value in locations])
                for family_effect in locations:
                    latent, cells = _draw_differences(rng, total, family_effect, scenario, config)
                    pair = [
                        latent + rng.normal(0, config.random_baseline_sd, total),
                        latent + rng.normal(0, config.random_baseline_sd, total),
                    ]
                    if scenario == "optimizer_failure":
                        for value in pair:
                            failed = rng.random(total) < config.failure_rate
                            value[failed] = config.failure_value  # ITT: retain failures.
                    comparisons.extend(pair)
                decisions = _comparison_decisions(comparisons, cells, config, rng, truths)
                for method, (tests, bounds, coverage) in decisions.items():
                    success = bool(np.count_nonzero(tests & bounds) >= 2)
                    null_families = truths == 0
                    any_false = bool(np.any((tests & bounds)[null_families]))
                    slot = counts.setdefault(method, [0, 0, 0, 0])
                    slot[0] += success
                    slot[1] += any_false
                    if coverage is not None:
                        slot[2] += int(np.count_nonzero(coverage))
                        slot[3] += len(coverage)
                # Separate negative-control diagnostic.  The candidate 0.005
                # practical margin is deliberately not part of a success gate.
                null_margin_positives += int(all(
                    conservative_sign_pvalue(value, margin=0.005) <= config.alpha / 2
                    and sign_median_lower_bound(value - 0.005, config.alpha / 2) > 0
                    for value in comparisons[:2]
                ))
            repetitions += todo
            if repetitions >= config.min_repetitions:
                worst = max(math.sqrt(max(rate * (1 - rate), 0.25 / repetitions) / repetitions) for values in counts.values() for rate in (values[0] / repetitions,))
                if worst <= config.target_mcse:
                    break
        for method, (successes, any_false, covered, coverage_total) in counts.items():
            estimate = successes / repetitions
            rows.append({
                "family_total": total, "effect": effect, "scenario": scenario,
                "method": method, "repetitions": repetitions,
                "success_probability": estimate,
                "success_mcse": math.sqrt(estimate * (1 - estimate) / repetitions),
                "null_configuration": "global_null" if effect == 0 else "mixed_two_alternative_one_null",
                "fwer": any_false / repetitions,
                "fwer_mcse": math.sqrt((any_false / repetitions) * (1 - any_false / repetitions) / repetitions),
                "lower_bound_coverage": covered / coverage_total if coverage_total else None,
                "coverage_mcse": math.sqrt((covered / coverage_total) * (1 - covered / coverage_total) / coverage_total) if coverage_total else None,
            })
        null_rate = null_margin_positives / repetitions
        rows.append({
            "family_total": total, "effect": effect, "scenario": scenario,
            "method": "null_sign_candidate_margin_0.005", "repetitions": repetitions,
            "success_probability": null_rate,
            "success_mcse": math.sqrt(null_rate * (1 - null_rate) / repetitions),
            "fwer": None, "fwer_mcse": None, "lower_bound_coverage": None,
            "coverage_mcse": None, "null_configuration": "negative_control_diagnostic",
        })
        if capped:
            break
    return {
        "status": "synthetic_sensitivity_only_not_benchmark_freeze",
        "config": config.__dict__,
        "design": {"family_totals": family_totals, "effects": effects, "scenarios": scenarios, "cells": 4},
        "completed_grid_points": len({(r["family_total"], r["effect"], r["scenario"]) for r in rows}),
        "planned_grid_points": len(grid),
        # Wall time is intentionally not serialized: otherwise identical seeded
        # runs would not be byte reproducible.  The CLI reports observed time.
        "runtime_seconds": None,
        "runtime_cap_reached": capped,
        "rows": rows,
        "limitations": [
            "Synthetic paired differences are not observed mixer performance.",
            "Bootstrap calibration is accepted only where simulated null FWER and median coverage are adequate.",
            "Wilcoxon is secondary because skewed differences need not be symmetric.",
        ],
    }
