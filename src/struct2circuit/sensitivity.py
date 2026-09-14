"""Synthetic-only sensitivity analysis for the pre-freeze statistical design.

No function in this module loads a benchmark, constructs a QUBO, or evaluates a
mixer.  One synthetic vector represents paired normalized-gap improvements for
independent hypothetical instances; positive values favor structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import time

import numpy as np
from scipy.stats import binom, norm, rankdata


FAMILY_TOTALS = (32, 48, 64, 96)
EFFECTS = (0.0, 0.005, 0.010, 0.015)
SCENARIOS = (
    "gaussian", "heavy_tailed", "skewed", "heterogeneous_regime",
    "ties", "optimizer_failure", "structure_failure",
)
DISPERSION_MULTIPLIERS = {"low": 0.67, "medium": 1.0, "high": 1.5}
NULL_MARGIN = 0.005


@dataclass(frozen=True)
class SensitivityConfig:
    """Controls a bounded and reproducible synthetic sensitivity run."""

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
    neutral_failure_value: float = 0.0
    structure_failure_value: float = -0.40

    def validate(self) -> None:
        if not (0 < self.alpha < 0.5):
            raise ValueError("alpha must be between 0 and 0.5")
        for name in ("bootstrap_samples", "min_repetitions", "max_repetitions", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_repetitions > self.max_repetitions:
            raise ValueError("min_repetitions cannot exceed max_repetitions")
        if self.runtime_cap_seconds <= 0 or self.target_mcse <= 0:
            raise ValueError("runtime cap and target_mcse must be positive")
        if self.random_baseline_sd < 0 or not (0 <= self.failure_rate <= 1):
            raise ValueError("variance and failure controls are invalid")
        if self.structure_failure_value > self.neutral_failure_value:
            raise ValueError("structure_failure_value must be no better than neutral")


@dataclass(frozen=True)
class SyntheticComparison:
    """Two observations and their latent population-median estimands."""

    ring: np.ndarray
    expected_random: np.ndarray
    ring_truth: float
    expected_random_truth: float


def conservative_sign_pvalue(values: np.ndarray, margin: float = 0.0) -> float:
    """Exact one-sided sign p-value, counting ties as non-positive."""

    arr = _finite_vector(values)
    positives = int(np.count_nonzero(arr > margin))
    return _sign_tail_table(len(arr))[positives]


def _finite_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("values must be a non-empty finite vector")
    return arr


@lru_cache(maxsize=None)
def _sign_tail_table(total: int) -> tuple[float, ...]:
    return tuple(float(binom.sf(positive - 1, total, 0.5)) for positive in range(total + 1))


@lru_cache(maxsize=None)
def _sign_lower_rank(total: int, alpha: float) -> int:
    ranks = [r for r in range(1, total + 1) if binom.cdf(r - 1, total, 0.5) <= alpha]
    return max(ranks) if ranks else 0


def sign_median_lower_bound(values: np.ndarray, alpha: float) -> float:
    """Distribution-free one-sided ``1-alpha`` median lower bound."""

    arr = np.sort(_finite_vector(values))
    if not (0 < alpha < 0.5):
        raise ValueError("alpha must be between 0 and 0.5")
    rank = _sign_lower_rank(len(arr), alpha)
    return float(arr[rank - 1]) if rank else -math.inf


def stratified_bootstrap_median(
    values: np.ndarray, cells: np.ndarray, *, samples: int, seed: int,
) -> np.ndarray:
    """Vectorized bootstrap medians preserving all four cell counts."""

    arr = _finite_vector(values)
    strata = np.asarray(cells)
    if strata.shape != arr.shape:
        raise ValueError("values and cells must be aligned vectors")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
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


def joint_lower_bound_coverage(lower_bounds: np.ndarray, truths: np.ndarray) -> bool:
    """Return one repetition's simultaneous coverage indicator."""

    lower = np.asarray(lower_bounds, dtype=float)
    target = np.asarray(truths, dtype=float)
    if lower.shape != target.shape:
        raise ValueError("lower bounds and truths must have the same shape")
    return bool(np.all(lower <= target))


def _difference_noise(
    rng: np.random.Generator, total: int, scenario: str, scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.repeat(np.arange(4), total // 4)
    sd = 0.015 * scale
    if scenario in {"gaussian", "optimizer_failure", "structure_failure"}:
        noise = rng.normal(0, sd, total)
    elif scenario == "heavy_tailed":
        noise = rng.standard_t(3, total) * (sd / math.sqrt(3))
    elif scenario == "skewed":
        noise = (rng.lognormal(-0.5, 0.8, total) - math.exp(-0.5)) * 0.014 * scale
    elif scenario == "heterogeneous_regime":
        offsets = np.asarray([-0.006, -0.002, 0.002, 0.006]) * scale
        noise = rng.normal(offsets[cells], 0.016 * scale)
    elif scenario == "ties":
        noise = rng.normal(0, 0.012 * scale, total)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return noise, cells


def generate_synthetic_comparison(
    *, seed: int, total: int, effect: float, scenario: str,
    dispersion: str, config: SensitivityConfig,
    apply_failure_intervention: bool = True,
) -> tuple[SyntheticComparison, np.ndarray]:
    """Generate ring and expected-random comparisons for one family.

    ``latent = effect + instance_noise`` is the true per-instance paired
    improvement for both baselines.  The observed ring comparison is exactly
    ``latent``.  Only the expected-random comparison adds independent normal
    estimator error with SD ``random_baseline_sd``; this models finite replication
    when estimating the expected random-connected baseline.  Both latent median
    estimands equal ``effect`` before a failure intervention.

    ``optimizer_failure`` replaces both comparisons by zero on failed instances;
    this is a neutral-value sensitivity convention, not intention-to-treat.
    ``structure_failure`` replaces both by a prespecified negative value and is the
    conservative favorable-claim stress test.
    """

    config.validate()
    if total <= 0 or total % 4:
        raise ValueError("total must be positive and balanced across four cells")
    if dispersion not in DISPERSION_MULTIPLIERS:
        raise ValueError("unknown dispersion")
    rng = np.random.default_rng(seed)
    noise, cells = _difference_noise(rng, total, scenario, DISPERSION_MULTIPLIERS[dispersion])
    latent = effect + noise
    ring = latent.copy()
    expected_random = latent + rng.normal(0, config.random_baseline_sd, total)
    truth = effect
    if apply_failure_intervention and scenario in {"optimizer_failure", "structure_failure"}:
        failed = rng.random(total) < config.failure_rate
        replacement = (
            config.neutral_failure_value if scenario == "optimizer_failure"
            else config.structure_failure_value
        )
        ring[failed] = replacement
        expected_random[failed] = replacement
        truth = _mixture_median(
            effect, replacement, config.failure_rate,
            0.015 * DISPERSION_MULTIPLIERS[dispersion],
        )
    if scenario == "ties":
        ring = np.round(ring / 0.005) * 0.005
        expected_random = np.round(expected_random / 0.005) * 0.005
        truth = round(effect / 0.005) * 0.005
    return SyntheticComparison(ring, expected_random, truth, truth), cells


def _mixture_median(effect: float, point: float, rate: float, sd: float) -> float:
    """Median of a point-mass/continuous-location mixture used as target proxy."""

    if rate == 0:
        return effect
    # Failure scenarios use Gaussian instance noise. This target deliberately
    # excludes random-baseline estimator error because that error is measurement,
    # not part of the expected-random estimand.
    below = (1 - rate) * norm.cdf((point - effect) / sd)
    if below <= 0.5 <= below + rate:
        return point
    probability = 0.5 / (1 - rate) if below > 0.5 else (0.5 - rate) / (1 - rate)
    return effect + sd * float(norm.ppf(probability))


def null_control_truth(structured_effects: tuple[float, ...] = ()) -> tuple[float, float]:
    """Return the separate null-control estimands; structured effects are irrelevant."""

    del structured_effects
    return (0.0, 0.0)


def generate_synthetic_repetition(
    *, seed: int, total: int, structured_effects: tuple[float, float, float],
    scenario: str, dispersion: str, config: SensitivityConfig,
) -> tuple[list[SyntheticComparison], SyntheticComparison, np.ndarray]:
    """Generate three structured families and an independently seeded null.

    The fourth seed is reserved for the null, so changing any structured effect
    changes neither its draw nor its population definition.
    """

    children = np.random.SeedSequence(seed).spawn(4)
    structured = [
        generate_synthetic_comparison(
            seed=int(child.generate_state(1, dtype=np.uint64)[0]), total=total,
            effect=effect, scenario=scenario, dispersion=dispersion, config=config,
        )[0]
        for child, effect in zip(children[:3], structured_effects, strict=True)
    ]
    null, cells = generate_synthetic_comparison(
        seed=int(children[3].generate_state(1, dtype=np.uint64)[0]), total=total,
        effect=0.0, scenario=scenario, dispersion=dispersion, config=config,
        apply_failure_intervention=False,
    )
    return structured, null, cells


def _wilcoxon_normal_pvalue(values: np.ndarray) -> float:
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum() + 0.5 * ranks[values == 0].sum())
    n = len(values)
    mean = n * (n + 1) / 4
    _, counts = np.unique(np.abs(values), return_counts=True)
    variance = (n * (n + 1) * (2 * n + 1) - np.sum(counts**3 - counts)) / 24
    return 1.0 if variance <= 0 else float(norm.sf((positive - mean - 0.5) / math.sqrt(variance)))


def _method_decisions(
    comparisons: list[SyntheticComparison], cells: np.ndarray,
    config: SensitivityConfig, rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, bool]]:
    values = [value for pair in comparisons for value in (pair.ring, pair.expected_random)]
    truths = np.asarray([truth for pair in comparisons for truth in (pair.ring_truth, pair.expected_random_truth)])
    sign_p = np.asarray([conservative_sign_pvalue(value) for value in values])
    sign_sim_lower = np.asarray([sign_median_lower_bound(value, config.alpha / 6) for value in values])
    family_p = np.asarray([max(sign_p[2 * i:2 * i + 2]) for i in range(3)])
    sign_family_lower = np.asarray([
        min(sign_median_lower_bound(values[2 * i + j], config.alpha / 3) for j in range(2))
        for i in range(3)
    ])

    boot_p, boot_sim_lower, boot_family_lower = [], [], []
    for value in values:
        draws = stratified_bootstrap_median(
            value, cells, samples=config.bootstrap_samples,
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
        )
        boot_p.append((np.count_nonzero(draws <= 0) + 1) / (len(draws) + 1))
        boot_sim_lower.append(float(np.quantile(draws, config.alpha / 6, method="lower")))
        boot_family_lower.append(float(np.quantile(draws, config.alpha / 3, method="lower")))
    boot_p = np.asarray(boot_p)
    boot_family_p = np.asarray([max(boot_p[2 * i:2 * i + 2]) for i in range(3)])
    boot_family_lower_arr = np.asarray(boot_family_lower).reshape(3, 2).min(axis=1)

    wilcoxon = np.asarray([_wilcoxon_normal_pvalue(value) for value in values])
    wilcoxon_family_p = np.asarray([max(wilcoxon[2 * i:2 * i + 2]) for i in range(3)])
    return {
        "sign_simultaneous": (
            ((sign_p <= config.alpha / 6) & (sign_sim_lower > 0)).reshape(3, 2).all(axis=1),
            joint_lower_bound_coverage(sign_sim_lower, truths),
        ),
        "sign_hierarchical_iut_holm": (
            holm_rejections(family_p, config.alpha) & (sign_family_lower > 0),
            joint_lower_bound_coverage(sign_family_lower, truths.reshape(3, 2).min(axis=1)),
        ),
        # Percentile-bootstrap rejection is required to agree with its lower
        # bound. It remains a candidate only if this joint coverage audit passes.
        "bootstrap_simultaneous": (
            ((boot_p <= config.alpha / 6) & (np.asarray(boot_sim_lower) > 0)).reshape(3, 2).all(axis=1),
            joint_lower_bound_coverage(np.asarray(boot_sim_lower), truths),
        ),
        "bootstrap_hierarchical_iut_holm": (
            holm_rejections(boot_family_p, config.alpha) & (boot_family_lower_arr > 0),
            joint_lower_bound_coverage(boot_family_lower_arr, truths.reshape(3, 2).min(axis=1)),
        ),
        "wilcoxon_hierarchical_iut_holm_secondary": (
            holm_rejections(wilcoxon_family_p, config.alpha), False,
        ),
    }


def _mcse(successes: int, repetitions: int) -> float:
    """Conservative MCSE bound; ``successes`` is retained for call-site clarity."""

    del successes
    return conservative_mc_precision(repetitions)


def conservative_mc_precision(repetitions: int) -> float:
    """Worst-case standard error bound for any Bernoulli proportion."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    return 0.5 / math.sqrt(repetitions)


def run_sensitivity(
    config: SensitivityConfig, *, family_totals: tuple[int, ...] = FAMILY_TOTALS,
    effects: tuple[float, ...] = EFFECTS, scenarios: tuple[str, ...] = SCENARIOS,
    dispersions: tuple[str, ...] = tuple(DISPERSION_MULTIPLIERS),
) -> dict:
    """Run the adaptive synthetic study subject to a wall-clock safety cap."""

    config.validate()
    if any(total <= 0 or total % 4 for total in family_totals):
        raise ValueError("family totals must be positive and balanced across four cells")
    if any(scenario not in SCENARIOS for scenario in scenarios):
        raise ValueError("unsupported scenario")
    if any(level not in DISPERSION_MULTIPLIERS for level in dispersions):
        raise ValueError("unsupported dispersion")
    grid = [(n, effect, scenario, dispersion) for n in family_totals for effect in effects for scenario in scenarios for dispersion in dispersions]
    started = time.monotonic()
    grid_seeds = np.random.SeedSequence(config.seed).spawn(len(grid))
    rows: list[dict] = []
    capped = False
    for (total, effect, scenario, dispersion), grid_seed in zip(grid, grid_seeds, strict=True):
        if time.monotonic() - started >= config.runtime_cap_seconds:
            capped = True
            break
        rng = np.random.default_rng(grid_seed)
        counts: dict[str, list[int]] = {}
        null_margin_positives = 0
        repetitions = 0
        stopping_reason = "max_repetitions"
        while repetitions < config.max_repetitions:
            if time.monotonic() - started >= config.runtime_cap_seconds:
                capped = True
                break
            todo = min(config.batch_size, config.max_repetitions - repetitions)
            completed_in_batch = 0
            for _ in range(todo):
                if time.monotonic() - started >= config.runtime_cap_seconds:
                    capped = True
                    break
                locations = (effect, 0.0, effect) if effect > 0 else (0.0, 0.0, 0.0)
                comparisons, null_pair, cells = generate_synthetic_repetition(
                    seed=int(rng.integers(0, np.iinfo(np.int64).max)), total=total,
                    structured_effects=locations, scenario=scenario,
                    dispersion=dispersion, config=config,
                )
                decisions = _method_decisions(comparisons, cells, config, rng)
                null_margin_positives += int(all(
                    conservative_sign_pvalue(value, NULL_MARGIN) <= config.alpha / 2
                    and sign_median_lower_bound(value - NULL_MARGIN, config.alpha / 2) > 0
                    for value in (null_pair.ring, null_pair.expected_random)
                ))
                for method, (family_rejections, covered) in decisions.items():
                    slot = counts.setdefault(method, [0, 0, 0])
                    slot[0] += int(np.count_nonzero(family_rejections) >= 2)
                    slot[1] += int(np.any(family_rejections[np.asarray(locations) == 0]))
                    slot[2] += int(covered)
                completed_in_batch += 1
            repetitions += completed_in_batch
            if capped:
                stopping_reason = "runtime_cap"
                break
            if repetitions >= config.min_repetitions:
                if conservative_mc_precision(repetitions) <= config.target_mcse:
                    stopping_reason = "target_precision"
                    break
        # If the cap expires before this point has any repetitions, omit it. This
        # produces an explicit, valid partial result rather than dividing by zero.
        if repetitions == 0:
            break
        common = {
            "family_total": total, "effect": effect, "scenario": scenario,
            "dispersion": dispersion,
            "dispersion_multiplier": DISPERSION_MULTIPLIERS[dispersion],
            "random_baseline_sd": config.random_baseline_sd,
            "repetitions": repetitions,
            "achieved_mc_precision": conservative_mc_precision(repetitions),
            "precision_method": "worst_case_bernoulli_0.5_over_sqrt_n",
            "stopping_reason": stopping_reason,
        }
        for method, (successes, false_rejections, coverage_successes) in counts.items():
            rows.append({
                **common, "method": method,
                "success_probability": successes / repetitions,
                "success_mcse": _mcse(successes, repetitions),
                "null_configuration": "global_null" if effect == 0 else "mixed_two_alternative_one_null",
                "fwer": false_rejections / repetitions,
                "fwer_mcse": _mcse(false_rejections, repetitions),
                "joint_lower_bound_coverage": coverage_successes / repetitions if "wilcoxon" not in method else None,
                "coverage_mcse": _mcse(coverage_successes, repetitions) if "wilcoxon" not in method else None,
            })
        rows.append({
            **common, "method": "null_sign_candidate_margin_0.005",
            "success_probability": null_margin_positives / repetitions,
            "success_mcse": _mcse(null_margin_positives, repetitions),
            "null_configuration": "independent_negative_control",
            "fwer": None, "fwer_mcse": None,
            "joint_lower_bound_coverage": None, "coverage_mcse": None,
        })
        if capped:
            break
    completed = len({(row["family_total"], row["effect"], row["scenario"], row["dispersion"]) for row in rows})
    return {
        "status": "synthetic_sensitivity_only_not_benchmark_freeze" if completed == len(grid) else "partial_runtime_cap_reached",
        "config": asdict(config),
        "design": {
            "family_totals": family_totals, "effects": effects,
            "scenarios": scenarios, "dispersions": dispersions,
            "dispersion_multipliers": DISPERSION_MULTIPLIERS, "cells": 4,
            "null_margin_candidate": NULL_MARGIN,
            "null_control_population_targets": {"ring": 0.0, "expected_random": 0.0},
            "null_control_failure_interventions": False,
            "precision_method": "worst_case_bernoulli_0.5_over_sqrt_n",
            "precision_monitors": [
                "success_probability", "false_rejection_fwer",
                "joint_lower_bound_coverage", "null_margin_diagnostic",
            ],
        },
        "completed_grid_points": completed,
        "planned_grid_points": len(grid),
        "runtime_seconds": None,
        "runtime_cap_reached": capped,
        "rows": rows,
        "limitations": [
            "Synthetic paired differences are not observed mixer performance.",
            "The wall-clock cap is a safety bound, not a deterministic stopping rule.",
            "Percentile-bootstrap procedures are unapproved pending joint-coverage validation.",
            "Wilcoxon is secondary because skewed differences need not be symmetric.",
        ],
    }
