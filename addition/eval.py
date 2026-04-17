from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from addition.config import ExperimentConfig
from addition.data import (
    AdditionProblem,
    EvaluationSuite,
    build_batch,
    carry_density,
    count_carry_chain,
    exact_sum_matches,
    maybe_trim_examples,
)
from addition.model import AdditionTransformer


@dataclass
class LengthMetrics:
    digit_accuracy: float
    final_carry_accuracy: float
    exact_match: float
    avg_carry_chain: float
    avg_carry_density: float
    example_count: int
    per_position_digit_accuracy: list[float]


def _chunked(sequence: list[AdditionProblem], chunk_size: int) -> Iterable[list[AdditionProblem]]:
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start : start + chunk_size]


@torch.no_grad()
def evaluate_problem_set(
    model: AdditionTransformer,
    config: ExperimentConfig,
    problems: list[AdditionProblem],
    active_digits: int,
    *,
    device: str,
    return_attention: bool = False,
) -> tuple[LengthMetrics, dict[str, float] | None]:
    model.eval()
    latent_steps = config.latent_steps_for_stage(active_digits)
    num_examples = len(problems)
    if num_examples == 0:
        empty = LengthMetrics(
            digit_accuracy=0.0,
            final_carry_accuracy=0.0,
            exact_match=0.0,
            avg_carry_chain=0.0,
            avg_carry_density=0.0,
            example_count=0,
            per_position_digit_accuracy=[0.0] * active_digits,
        )
        return empty, None

    predicted_digits = torch.zeros(num_examples, active_digits, dtype=torch.long)
    predicted_final_carry = torch.zeros(num_examples, dtype=torch.long)
    truth_digits = torch.tensor([[problem.sum_digits[position] for position in range(active_digits)] for problem in problems], dtype=torch.long)
    truth_final_carry = torch.tensor([problem.carry_out[active_digits - 1] for problem in problems], dtype=torch.long)
    attention_stats: dict[str, float] | None = None

    offset = 0
    for problem_chunk in _chunked(problems, config.eval_batch_size):
        batch = build_batch(
            problems=problem_chunk,
            radix=config.radix,
            device=device,
        )
        outputs = model(batch.input_ids, latent_steps=latent_steps, return_attention=return_attention)
        chunk_size = len(problem_chunk)
        predicted_digits[offset : offset + chunk_size] = outputs.digit_logits.argmax(dim=-1)[:, :active_digits].cpu()
        predicted_final_carry[offset : offset + chunk_size] = outputs.final_carry_logits.argmax(dim=-1).cpu()
        if return_attention and attention_stats is None:
            attention_stats = summarize_attention(
                attention_weights=outputs.attention_weights,
                active_digits=active_digits,
                input_sequence_length=batch.input_ids.shape[1],
                output_sequence_length=outputs.output_hidden.shape[1],
            )
        offset += chunk_size

    exact_matches = []
    for example_index, problem in enumerate(problems):
        exact_matches.append(
            exact_sum_matches(
                predicted_digits=predicted_digits[example_index].tolist(),
                predicted_final_carry=int(predicted_final_carry[example_index].item()),
                truth_digits=problem.sum_digits[:active_digits],
                truth_final_carry=problem.carry_out[active_digits - 1],
            )
        )

    per_position_digit = (predicted_digits == truth_digits).float().mean(dim=0).tolist()
    metrics = LengthMetrics(
        digit_accuracy=float((predicted_digits == truth_digits).float().mean().item()),
        final_carry_accuracy=float((predicted_final_carry == truth_final_carry).float().mean().item()),
        exact_match=float(torch.tensor(exact_matches, dtype=torch.float32).mean().item()),
        avg_carry_chain=float(sum(count_carry_chain(problem) for problem in problems) / len(problems)),
        avg_carry_density=float(sum(carry_density(problem) for problem in problems) / len(problems)),
        example_count=len(problems),
        per_position_digit_accuracy=[float(value) for value in per_position_digit],
    )
    return metrics, attention_stats


def summarize_attention(
    attention_weights: torch.Tensor | None,
    *,
    active_digits: int,
    input_sequence_length: int,
    output_sequence_length: int,
) -> dict[str, float]:
    if attention_weights is None:
        return {}
    # Shape: [batch, heads, target_len, source_len]
    final_attention = attention_weights[:, :, -1, :]
    attention_mean = final_attention.mean(dim=(0, 1))
    active_last_a_index = active_digits
    active_last_b_index = input_sequence_length // 2 + active_digits
    latent_slice = attention_mean[input_sequence_length : -output_sequence_length]
    output_slice = attention_mean[-output_sequence_length:-1]
    entropy = -torch.sum(attention_mean * torch.log(attention_mean.clamp_min(1e-9))).item()
    summary = {
        "lsd_a_attention": float(attention_mean[1].item()),
        "msd_a_attention": float(attention_mean[active_last_a_index].item()),
        "lsd_b_attention": float(attention_mean[(input_sequence_length // 2) + 1].item()),
        "msd_b_attention": float(attention_mean[active_last_b_index].item()),
        "attention_entropy": float(entropy),
        "all_latent_attention": float(latent_slice.sum().item()) if latent_slice.numel() else 0.0,
        "previous_output_attention": float(output_slice.sum().item()) if output_slice.numel() else 0.0,
    }
    return summary


@torch.no_grad()
def evaluate_length_dict(
    model: AdditionTransformer,
    config: ExperimentConfig,
    problems_by_length: dict[int, list[AdditionProblem]],
    *,
    device: str,
    attention_length: int | None = None,
) -> dict[str, dict]:
    structured: dict[str, dict] = {}
    for length, problems in sorted(problems_by_length.items()):
        length_metrics, attention = evaluate_problem_set(
            model=model,
            config=config,
            problems=problems,
            active_digits=length,
            device=device,
            return_attention=attention_length is not None and attention_length == length,
        )
        structured[str(length)] = {
            "digit_accuracy": length_metrics.digit_accuracy,
            "final_carry_accuracy": length_metrics.final_carry_accuracy,
            "exact_match": length_metrics.exact_match,
            "avg_carry_chain": length_metrics.avg_carry_chain,
            "avg_carry_density": length_metrics.avg_carry_density,
            "example_count": length_metrics.example_count,
            "per_position_digit_accuracy": length_metrics.per_position_digit_accuracy,
        }
        if attention is not None:
            structured[str(length)]["attention_summary"] = attention
    return structured


def collect_hidden_dataset(
    model: AdditionTransformer,
    config: ExperimentConfig,
    problems: list[AdditionProblem],
    *,
    active_digits: int,
    device: str,
    limit_examples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    latent_steps = config.latent_steps_for_stage(active_digits)
    selected = maybe_trim_examples(problems, limit_examples)
    hidden_states: list[torch.Tensor] = []
    carry_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for problem_chunk in _chunked(selected, config.eval_batch_size):
            batch = build_batch(
                problems=problem_chunk,
                radix=config.radix,
                device=device,
            )
            outputs = model(batch.input_ids, latent_steps=latent_steps, return_attention=False)
            slot_hidden = outputs.output_hidden[:, :active_digits, :]
            slot_mask = batch.target_digit_mask
            hidden_states.append(slot_hidden[slot_mask].detach().cpu())
            carry_targets.append(batch.target_carry[slot_mask].detach().cpu())
    return torch.cat(hidden_states, dim=0), torch.cat(carry_targets, dim=0)


def fit_linear_probe(
    hidden_states: torch.Tensor,
    carry_targets: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    if hidden_states.numel() == 0:
        return {"probe_accuracy": 0.0}
    indices = torch.randperm(hidden_states.shape[0])
    hidden_states = hidden_states[indices]
    carry_targets = carry_targets[indices]
    split_index = max(1, int(0.8 * hidden_states.shape[0]))
    train_hidden = hidden_states[:split_index]
    train_targets = carry_targets[:split_index]
    test_hidden = hidden_states[split_index:]
    test_targets = carry_targets[split_index:]
    if test_hidden.numel() == 0:
        test_hidden = train_hidden
        test_targets = train_targets

    probe = nn.Linear(hidden_states.shape[-1], 2)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = probe(train_hidden)
        loss = loss_fn(logits, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = probe(test_hidden).argmax(dim=-1)
        accuracy = float((predictions == test_targets).float().mean().item())
    return {"probe_accuracy": accuracy}


def evaluate_suite(
    model: AdditionTransformer,
    config: ExperimentConfig,
    suite: EvaluationSuite,
    *,
    device: str,
) -> dict[str, dict]:
    id_lengths = list(range(1, config.train_max_digits + 1))
    ood_lengths = list(config.ood_lengths)
    max_attention_length = max(ood_lengths) if ood_lengths else config.train_max_digits

    validation = evaluate_length_dict(
        model=model,
        config=config,
        problems_by_length={length: suite.validation_uniform[length] for length in id_lengths},
        device=device,
    )
    uniform_all = evaluate_length_dict(
        model=model,
        config=config,
        problems_by_length={length: suite.test_uniform[length] for length in sorted(set(id_lengths + ood_lengths))},
        device=device,
        attention_length=max_attention_length,
    )
    carry_heavy_all = evaluate_length_dict(
        model=model,
        config=config,
        problems_by_length={length: suite.test_carry_heavy[length] for length in sorted(set(id_lengths + ood_lengths))},
        device=device,
        attention_length=max_attention_length,
    )
    probe_hidden, probe_targets = collect_hidden_dataset(
        model=model,
        config=config,
        problems=suite.test_carry_heavy[max_attention_length],
        active_digits=max_attention_length,
        device=device,
        limit_examples=config.attention_probe_examples,
    )
    diagnostics = fit_linear_probe(
        hidden_states=probe_hidden,
        carry_targets=probe_targets,
        epochs=config.linear_probe_epochs,
        learning_rate=config.linear_probe_lr,
    )
    diagnostics["attention_uniform"] = uniform_all[str(max_attention_length)].get("attention_summary", {})
    diagnostics["attention_carry_heavy"] = carry_heavy_all[str(max_attention_length)].get("attention_summary", {})
    return {
        "validation_uniform": validation,
        "test_uniform": uniform_all,
        "test_carry_heavy": carry_heavy_all,
        "diagnostics": diagnostics,
    }


def stage_validation_metric(results: dict[str, dict], stage: int) -> float:
    stage_metrics = results["validation_uniform"][str(stage)]
    return float(stage_metrics["digit_accuracy"])


def flatten_nested_metrics(prefix: str, nested: dict[str, dict]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for split_name, split_metrics in nested.items():
        if split_name == "diagnostics":
            for key, value in split_metrics.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        flat[f"{prefix}{split_name}/{key}/{inner_key}"] = float(inner_value)
                else:
                    flat[f"{prefix}{split_name}/{key}"] = float(value)
            continue
        for length, length_metrics in split_metrics.items():
            if not isinstance(length_metrics, dict):
                continue
            for metric_name, metric_value in length_metrics.items():
                if isinstance(metric_value, list):
                    if metric_value:
                        flat[f"{prefix}{split_name}/length_{length}/{metric_name}_mean"] = float(sum(metric_value) / len(metric_value))
                    continue
                if isinstance(metric_value, dict):
                    for inner_key, inner_value in metric_value.items():
                        flat[f"{prefix}{split_name}/length_{length}/{metric_name}/{inner_key}"] = float(inner_value)
                    continue
                flat[f"{prefix}{split_name}/length_{length}/{metric_name}"] = float(metric_value)
    return flat
