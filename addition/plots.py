from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def plot_training_history(history: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    if not history:
        return []
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [entry["global_step"] for entry in history]
    digit_acc = [entry["validation_digit_accuracy"] for entry in history]
    carry_acc = [entry["validation_carry_accuracy"] for entry in history]
    exact_match = [entry["validation_exact_match"] for entry in history]
    stages = [entry["stage"] for entry in history]

    saved_paths: list[Path] = []

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, digit_acc, label="Val digit acc")
    plt.plot(steps, carry_acc, label="Val carry acc")
    plt.plot(steps, exact_match, label="Val exact match")
    plt.xlabel("Global step")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.01)
    plt.legend()
    plt.tight_layout()
    metrics_path = output_dir / "training_curves.png"
    plt.savefig(metrics_path, dpi=160)
    plt.close()
    saved_paths.append(metrics_path)

    plt.figure(figsize=(8, 4.5))
    plt.step(steps, stages, where="post")
    plt.xlabel("Global step")
    plt.ylabel("Curriculum stage")
    plt.tight_layout()
    stage_path = output_dir / "stage_progression.png"
    plt.savefig(stage_path, dpi=160)
    plt.close()
    saved_paths.append(stage_path)

    return saved_paths


def _collect_length_metric(aggregate: dict[str, Any], method: str, split: str, metric: str) -> tuple[list[int], list[float], list[float]]:
    lengths = sorted(int(length) for length in aggregate[method][split].keys())
    means = [aggregate[method][split][str(length)][metric]["mean"] for length in lengths]
    stds = [aggregate[method][split][str(length)][metric]["std"] for length in lengths]
    return lengths, means, stds


def plot_method_comparison(aggregate: dict[str, Any], output_dir: Path) -> list[Path]:
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    methods = list(aggregate.keys())
    splits = [
        ("test_uniform", "uniform_exact_match.png", "Uniform exact-match by length"),
        ("test_carry_heavy", "carry_heavy_exact_match.png", "Carry-heavy exact-match by length"),
    ]
    for split, filename, title in splits:
        plt.figure(figsize=(8, 4.5))
        for method in methods:
            lengths, means, stds = _collect_length_metric(aggregate, method, split, "exact_match")
            plt.plot(lengths, means, marker="o", label=method)
            lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
            upper = [min(1.0, mean + std) for mean, std in zip(means, stds)]
            plt.fill_between(lengths, lower, upper, alpha=0.15)
        plt.xlabel("Active digits")
        plt.ylabel("Exact-match accuracy")
        plt.title(title)
        plt.ylim(0.0, 1.01)
        plt.legend()
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        saved_paths.append(path)

    plt.figure(figsize=(8, 4.5))
    for method in methods:
        stages = sorted(int(stage) for stage in aggregate[method]["stage_progression"].keys())
        means = [aggregate[method]["stage_progression"][str(stage)]["validation_digit_accuracy"]["mean"] for stage in stages]
        stds = [aggregate[method]["stage_progression"][str(stage)]["validation_digit_accuracy"]["std"] for stage in stages]
        plt.plot(stages, means, marker="o", label=method)
        plt.fill_between(
            stages,
            [max(0.0, mean - std) for mean, std in zip(means, stds)],
            [min(1.0, mean + std) for mean, std in zip(means, stds)],
            alpha=0.15,
        )
    plt.xlabel("Curriculum stage")
    plt.ylabel("Best validation digit accuracy")
    plt.ylim(0.0, 1.01)
    plt.title("Validation digit accuracy vs stage")
    plt.legend()
    plt.tight_layout()
    stage_curve_path = output_dir / "validation_digit_accuracy_by_stage.png"
    plt.savefig(stage_curve_path, dpi=160)
    plt.close()
    saved_paths.append(stage_curve_path)
    return saved_paths


def plot_single_run_results(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = plot_training_history(summary.get("history", []), output_dir)
    uniform = summary["final_results"]["test_uniform"]
    carry_heavy = summary["final_results"]["test_carry_heavy"]
    lengths = sorted(int(length) for length in uniform.keys())
    uniform_exact = [uniform[str(length)]["exact_match"] for length in lengths]
    carry_exact = [carry_heavy[str(length)]["exact_match"] for length in lengths]
    plt.figure(figsize=(8, 4.5))
    plt.plot(lengths, uniform_exact, marker="o", label="Uniform")
    plt.plot(lengths, carry_exact, marker="o", label="Carry-heavy")
    plt.xlabel("Active digits")
    plt.ylabel("Exact-match accuracy")
    plt.ylim(0.0, 1.01)
    plt.legend()
    plt.tight_layout()
    final_curve_path = output_dir / "final_exact_match_by_length.png"
    plt.savefig(final_curve_path, dpi=160)
    plt.close()
    saved_paths.append(final_curve_path)
    return saved_paths
