"""Pilot statistics, figures and automatically generated research notes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str = "median",
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 20260814,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(samples, len(arr)))
    draws = arr[indices]
    stats = np.median(draws, axis=1) if statistic == "median" else np.mean(draws, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def summarize_pilot(results: pd.DataFrame, config: dict) -> dict:
    pivot_gap = results.pivot(index="instance", columns="mixer", values="normalized_gap")
    pivot_prob = results.pivot(index="instance", columns="mixer", values="probability_optimum")
    improvement = pivot_gap["ring"] - pivot_gap["structure"]
    prob_improvement = pivot_prob["structure"] - pivot_prob["ring"]
    ci_low, ci_high = paired_bootstrap_ci(improvement.to_numpy())
    if np.allclose(improvement, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(improvement, alternative="greater").pvalue)

    by_mixer: dict[str, dict] = {}
    for mixer, frame in results.groupby("mixer", sort=True):
        by_mixer[mixer] = {
            "median_normalized_gap": float(frame["normalized_gap"].median()),
            "mean_normalized_gap": float(frame["normalized_gap"].mean()),
            "median_probability_optimum": float(frame["probability_optimum"].median()),
            "mean_probability_optimum": float(frame["probability_optimum"].mean()),
            "mixer_edges": int(frame["mixer_edges"].iloc[0]),
        }

    summary = {
        "status": "exploratory_pilot_not_confirmatory",
        "central_claim": (
            "A structure-conditioned, feasibility-preserving mixer improves the "
            "solution-quality/resource Pareto frontier over fixed mixers for "
            "cardinality-constrained QUBOs."
        ),
        "configuration": config,
        "by_mixer": by_mixer,
        "paired_structure_vs_ring": {
            "median_gap_reduction": float(np.median(improvement)),
            "mean_gap_reduction": float(np.mean(improvement)),
            "median_gap_reduction_bootstrap_95_ci": [ci_low, ci_high],
            "median_probability_optimum_increase": float(np.median(prob_improvement)),
            "wins": int(np.sum(improvement > 1e-10)),
            "ties": int(np.sum(np.abs(improvement) <= 1e-10)),
            "losses": int(np.sum(improvement < -1e-10)),
            "one_sided_wilcoxon_p": wilcoxon_p,
        },
        "interpretation_rule": (
            "Advance to the preregistered confirmatory study only if the paired median "
            "gap reduction is positive and its exploratory bootstrap interval does not "
            "contain zero. This pilot must not be reported as confirmatory evidence."
        ),
    }
    return summary


def write_report(summary: dict, output_path: Path) -> None:
    pair = summary["paired_structure_vs_ring"]
    by = summary["by_mixer"]
    lo, hi = pair["median_gap_reduction_bootstrap_95_ci"]
    advances = pair["median_gap_reduction"] > 0 and lo > 0
    verdict = (
        "The pilot clears the pre-specified advancement rule. A larger blind study is justified."
        if advances else
        "The pilot does not clear the advancement rule. The construction or hypothesis should be revised before scaling."
    )
    text = f"""# Struct2Circuit pilot report

**Status:** exploratory pilot; not confirmatory evidence and not a quantum-advantage claim.

## Central claim under test

{summary['central_claim']}

## Pilot result

- Paired median normalized-gap reduction, structure versus ring: **{pair['median_gap_reduction']:.6f}**.
- Exploratory paired-bootstrap 95% interval: **[{lo:.6f}, {hi:.6f}]**.
- Structure wins / ties / losses: **{pair['wins']} / {pair['ties']} / {pair['losses']}**.
- One-sided paired Wilcoxon p-value: **{pair['one_sided_wilcoxon_p']:.6g}**.
- Median optimal-solution probability increase: **{pair['median_probability_optimum_increase']:.6f}**.

At the same mixer-edge budget, the ring median gap was **{by['ring']['median_normalized_gap']:.6f}** and the structure-conditioned median gap was **{by['structure']['median_normalized_gap']:.6f}**. The complete mixer used **{by['complete']['mixer_edges']}** edges, compared with **{by['ring']['mixer_edges']}** for both equal-budget mixers.

## Decision

{verdict}

## What this does not establish

This small, noiseless, depth-one study does not establish generalization, hardware advantage, scaling advantage or superiority to tuned classical optimization. It is a software/invariant check and an effect-size estimate for the confirmatory design.

## Next confirmatory milestone

Lock generator seeds before evaluation; include multiple structural regimes and at least one non-financial cardinality family; compare against XY-QAOA, warm-start and counterdiabatic baselines; incorporate cost-layer and transpilation resources; evaluate depths 1-3; and report the search cost required to construct each learned mixer.
"""
    output_path.write_text(text, encoding="utf-8")


def plot_pilot(results: pd.DataFrame, output_path: Path) -> None:
    order = ["ring", "structure", "complete"]
    colors = {"ring": "#64748B", "structure": "#0F766E", "complete": "#2563EB"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    pivot = results.pivot(index="instance", columns="mixer", values="normalized_gap")
    for _, row in pivot.iterrows():
        axes[0].plot([0, 1], [row["ring"], row["structure"]], color="#CBD5E1", lw=0.7, alpha=0.8)
    axes[0].scatter(np.zeros(len(pivot)), pivot["ring"], color=colors["ring"], s=22, label="fixed ring")
    axes[0].scatter(np.ones(len(pivot)), pivot["structure"], color=colors["structure"], s=22, label="structure")
    axes[0].set_xticks([0, 1], ["Ring", "Structure"])
    axes[0].set_ylabel("Normalized optimality gap (lower is better)")
    axes[0].set_title("Paired equal-budget comparison")
    axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.7)

    grouped = results.groupby("mixer", sort=False).agg(
        median_gap=("normalized_gap", "median"),
        q1=("normalized_gap", lambda s: s.quantile(0.25)),
        q3=("normalized_gap", lambda s: s.quantile(0.75)),
        edges=("mixer_edges", "first"),
    ).loc[order]
    yerr = np.vstack([grouped["median_gap"] - grouped["q1"], grouped["q3"] - grouped["median_gap"]])
    for idx, mixer in enumerate(order):
        axes[1].errorbar(
            grouped.loc[mixer, "edges"], grouped.loc[mixer, "median_gap"],
            yerr=yerr[:, idx:idx+1], fmt="o", markersize=7,
            capsize=4, color=colors[mixer], label=mixer.title(),
        )
    axes[1].set_xlabel("Mixer edges per QAOA layer")
    axes[1].set_ylabel("Median normalized gap")
    axes[1].set_title("Pilot quality/resource frontier")
    axes[1].grid(color="#E2E8F0", linewidth=0.7)
    axes[1].legend(frameon=False)

    fig.suptitle("Struct2Circuit exploratory pilot | p=1 feasible-subspace QAOA", fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def save_summary(summary: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

