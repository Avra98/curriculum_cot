from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from addition.config import VALID_MODELS, add_config_arguments, apply_preset, build_config_from_args
from addition.plots import plot_method_comparison
from addition.train import run_experiment


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _aggregate_split_metrics(run_summaries: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    lengths = sorted(run_summaries[0]["final_results"][split_name].keys(), key=int)
    metric_names = ["digit_accuracy", "carry_accuracy", "exact_match", "avg_carry_chain", "avg_carry_density"]
    aggregated: dict[str, Any] = {}
    for length in lengths:
        aggregated[length] = {}
        for metric_name in metric_names:
            values = [float(summary["final_results"][split_name][length][metric_name]) for summary in run_summaries]
            aggregated[length][metric_name] = _mean_std(values)
    return aggregated


def _aggregate_stage_progression(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    max_stage = max(int(entry["stage"]) for summary in run_summaries for entry in summary["history"])
    aggregated: dict[str, Any] = {}
    for stage in range(1, max_stage + 1):
        stage_values = []
        stage_exact = []
        for summary in run_summaries:
            stage_entries = [entry for entry in summary["history"] if int(entry["stage"]) == stage]
            if not stage_entries:
                continue
            stage_values.append(max(float(entry["validation_digit_accuracy"]) for entry in stage_entries))
            stage_exact.append(max(float(entry["validation_exact_match"]) for entry in stage_entries))
        aggregated[str(stage)] = {
            "validation_digit_accuracy": _mean_std(stage_values),
            "validation_exact_match": _mean_std(stage_exact),
        }
    return aggregated


def _aggregate_diagnostics(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [summary["final_results"]["diagnostics"] for summary in run_summaries]
    output: dict[str, Any] = {
        "probe_accuracy": _mean_std([float(diag["probe_accuracy"]) for diag in diagnostics]),
    }
    for attention_key in ("attention_uniform", "attention_carry_heavy"):
        attention_values = [diag.get(attention_key, {}) for diag in diagnostics]
        metric_names = sorted({metric for diag in attention_values for metric in diag.keys()})
        output[attention_key] = {
            metric_name: _mean_std([float(diag.get(metric_name, 0.0)) for diag in attention_values]) for metric_name in metric_names
        }
    return output


def aggregate_runs(results_by_method: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method, run_summaries in results_by_method.items():
        aggregate[method] = {
            "test_uniform": _aggregate_split_metrics(run_summaries, "test_uniform"),
            "test_carry_heavy": _aggregate_split_metrics(run_summaries, "test_carry_heavy"),
            "stage_progression": _aggregate_stage_progression(run_summaries),
            "diagnostics": _aggregate_diagnostics(run_summaries),
        }
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full addition comparison across methods and seeds.")
    add_config_arguments(parser)
    parser.add_argument("--methods", nargs="*", default=list(VALID_MODELS), choices=VALID_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--comparison_output_dir", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = apply_preset(build_config_from_args(args))
    seeds = args.seeds or list(range(base_config.comparison_num_seeds))
    comparison_root = Path(args.comparison_output_dir or f"addition_runs/comparison_{base_config.preset}")
    comparison_root.mkdir(parents=True, exist_ok=True)

    results_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in args.methods:
        results_by_method[method] = []
        for seed in seeds:
            args.model = method
            args.seed = seed
            args.output_dir = str(comparison_root / f"{method}_seed{seed}")
            config = apply_preset(build_config_from_args(args))
            config.output_dir = str(comparison_root / f"{method}_seed{seed}")
            print(f"[addition comparison] running method={method} seed={seed}", flush=True)
            summary = run_experiment(config)
            results_by_method[method].append(summary)

    aggregate = aggregate_runs(results_by_method)
    aggregate_payload = {
        "methods": args.methods,
        "seeds": seeds,
        "aggregate": aggregate,
    }
    with (comparison_root / "aggregate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_payload, handle, indent=2, sort_keys=True)
    plot_method_comparison(aggregate, comparison_root / "plots")


if __name__ == "__main__":
    main()
