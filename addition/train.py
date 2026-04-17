from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from addition.config import ExperimentConfig, ensure_output_dirs, parse_config, save_config
from addition.data import build_batch, build_evaluation_suite, digits_to_string, exact_sum_matches, sample_training_batch, seed_everything
from addition.eval import evaluate_problem_set, evaluate_suite, flatten_nested_metrics
from addition.model import build_model, describe_model
from addition.plots import plot_single_run_results


def _maybe_init_wandb(config: ExperimentConfig, output_dir: Path):
    if not config.use_wandb or config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing with local logging only.")
        return None
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=config.effective_run_name,
        mode=config.wandb_mode,
        config={"experiment": config.__dict__},
        dir=str(output_dir),
        reinit=True,
    )
    return run


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def _stage_checkpoint_path(stage_directory: Path, stage: int) -> Path:
    return stage_directory / f"stage_{stage:02d}_passed.pt"


def _evaluate_current_stage(
    model: nn.Module,
    config: ExperimentConfig,
    suite,
    stage: int,
    device: str,
) -> dict[str, float]:
    stage_metrics, _ = evaluate_problem_set(
        model=model,
        config=config,
        problems=suite.validation_uniform[stage],
        active_digits=stage,
        device=device,
        return_attention=False,
    )
    return {
        "digit_accuracy": stage_metrics.digit_accuracy,
        "final_carry_accuracy": stage_metrics.final_carry_accuracy,
        "exact_match": stage_metrics.exact_match,
    }


def _masked_digit_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    masked_logits = logits[mask]
    masked_targets = targets[mask]
    if masked_logits.numel() == 0:
        return logits.new_zeros(())
    return loss_fn(masked_logits, masked_targets)


@torch.no_grad()
def _print_model_debug_format(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    stage: int,
    rng,
    device: str,
) -> None:
    debug_batch = sample_training_batch(config=config, stage=stage, rng=rng, device=device)
    outputs = model(debug_batch.input_ids, latent_steps=config.latent_steps_for_stage(stage), return_attention=False)
    print("[addition debug] model_architecture", flush=True)
    print(model, flush=True)
    print(
        "[addition debug] batch_format "
        f"stage={stage} input_shape={tuple(debug_batch.input_ids.shape)} "
        f"target_digits_shape={tuple(debug_batch.target_digits.shape)} "
        f"target_mask_shape={tuple(debug_batch.target_digit_mask.shape)} "
        f"target_final_carry_shape={tuple(debug_batch.target_final_carry.shape)} "
        f"digit_logits_shape={tuple(outputs.digit_logits.shape)} "
        f"final_carry_logits_shape={tuple(outputs.final_carry_logits.shape)} "
        f"output_hidden_shape={tuple(outputs.output_hidden.shape)}",
        flush=True,
    )


@torch.no_grad()
def _print_validation_samples(
    model: nn.Module,
    config: ExperimentConfig,
    problems,
    *,
    stage: int,
    device: str,
    limit: int = 3,
) -> None:
    sample_problems = list(problems[:limit])
    if not sample_problems:
        return
    batch = build_batch(problems=sample_problems, radix=config.radix, device=device)
    outputs = model(batch.input_ids, latent_steps=config.latent_steps_for_stage(stage), return_attention=False)
    predicted_digits = outputs.digit_logits.argmax(dim=-1).cpu().tolist()
    predicted_final_carry = outputs.final_carry_logits.argmax(dim=-1).cpu().tolist()

    for example_index, problem in enumerate(sample_problems):
        truth_digits = problem.sum_digits[:stage]
        truth_final_carry = problem.carry_out[stage - 1]
        pred_digits = predicted_digits[example_index][:stage]
        pred_final_carry = int(predicted_final_carry[example_index])
        exact = exact_sum_matches(
            predicted_digits=pred_digits,
            predicted_final_carry=pred_final_carry,
            truth_digits=truth_digits,
            truth_final_carry=truth_final_carry,
        )
        a_text = digits_to_string(problem.a_digits[:stage], final_carry=0, radix=config.radix)
        b_text = digits_to_string(problem.b_digits[:stage], final_carry=0, radix=config.radix)
        pred_text = digits_to_string(pred_digits, final_carry=pred_final_carry, radix=config.radix)
        truth_text = digits_to_string(truth_digits, final_carry=truth_final_carry, radix=config.radix)
        print(
            f"[addition sample] stage={stage} idx={example_index} "
            f"a={a_text} b={b_text} pred={pred_text} true={truth_text} "
            f"pred_digits={pred_digits} pred_carry={pred_final_carry} "
            f"true_digits={truth_digits} true_carry={truth_final_carry} exact={int(exact)}",
            flush=True,
        )


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    directories = ensure_output_dirs(config)
    save_config(config, directories["root"])
    seed_everything(config.seed)
    device = config.device
    model = build_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    digit_loss_fn = nn.CrossEntropyLoss()
    final_carry_loss_fn = nn.CrossEntropyLoss()
    suite = build_evaluation_suite(config)
    rng = __import__("random").Random(config.seed + 12345)
    history: list[dict[str, Any]] = []
    best_validation = -1.0
    best_checkpoint_path = directories["checkpoints"] / "best.pt"
    last_checkpoint_path = directories["checkpoints"] / "last.pt"
    stage = config.initial_stage if config.uses_curriculum else config.train_max_digits
    stage_steps = 0
    global_step = 0
    stop_reason = "train_steps_exhausted"
    wandb_run = _maybe_init_wandb(config, directories["root"])
    started_at = time.time()
    param_counts = describe_model(config)
    print(
        f"[addition train] model={config.model} seed={config.seed} device={device} "
        f"params={param_counts['total_params']} stage={stage}",
        flush=True,
    )
    _print_model_debug_format(model=model, config=config, stage=stage, rng=rng, device=device)

    while global_step < config.train_steps:
        model.train()
        batch = sample_training_batch(config=config, stage=stage, rng=rng, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.input_ids, latent_steps=config.latent_steps_for_stage(stage), return_attention=False)
        digit_loss = _masked_digit_loss(
            logits=outputs.digit_logits,
            targets=batch.target_digits,
            mask=batch.target_digit_mask,
            loss_fn=digit_loss_fn,
        )
        final_carry_loss = final_carry_loss_fn(outputs.final_carry_logits, batch.target_final_carry)
        loss = digit_loss + final_carry_loss
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        global_step += 1
        stage_steps += 1

        if global_step % max(1, config.validation_interval // 2) == 0:
            train_message = (
                f"[addition train] step={global_step} stage={stage} "
                f"loss={loss.item():.4f} digit_loss={digit_loss.item():.4f} "
                f"final_carry_loss={final_carry_loss.item():.4f}"
            )
            print(train_message, flush=True)

        should_validate = (
            global_step % config.validation_interval == 0
            or global_step == config.train_steps
            or (config.uses_curriculum and stage_steps == config.max_steps_per_stage)
        )
        if not should_validate:
            continue

        validation = _evaluate_current_stage(model=model, config=config, suite=suite, stage=stage, device=device)
        history_entry = {
            "global_step": global_step,
            "stage": stage,
            "stage_steps": stage_steps,
            "loss": float(loss.item()),
            "digit_loss": float(digit_loss.item()),
            "final_carry_loss": float(final_carry_loss.item()),
            "validation_digit_accuracy": validation["digit_accuracy"],
            "validation_final_carry_accuracy": validation["final_carry_accuracy"],
            "validation_exact_match": validation["exact_match"],
            "latent_steps": config.latent_steps_for_stage(stage),
        }
        history.append(history_entry)
        print(
            f"[addition val] step={global_step} stage={stage} "
            f"digit_acc={validation['digit_accuracy']:.4f} final_carry_acc={validation['final_carry_accuracy']:.4f} "
            f"exact={validation['exact_match']:.4f}",
            flush=True,
        )
        _print_validation_samples(
            model=model,
            config=config,
            problems=suite.validation_uniform[stage],
            stage=stage,
            device=device,
        )
        if wandb_run is not None:
            payload = {
                "train/loss": float(loss.item()),
                "train/digit_loss": float(digit_loss.item()),
                "train/final_carry_loss": float(final_carry_loss.item()),
                "train/stage": stage,
                "train/latent_steps": config.latent_steps_for_stage(stage),
                "validation/digit_accuracy": validation["digit_accuracy"],
                "validation/final_carry_accuracy": validation["final_carry_accuracy"],
                "validation/exact_match": validation["exact_match"],
                "step": global_step,
            }
            wandb_run.log(payload)

        if validation["exact_match"] >= best_validation:
            best_validation = validation["exact_match"]
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                metadata={
                    "global_step": global_step,
                    "stage": stage,
                    "best_validation_exact_match": best_validation,
                },
            )

        reached_threshold = validation["exact_match"] >= config.stage_accuracy_threshold
        reached_cap = stage_steps >= config.max_steps_per_stage

        if config.uses_curriculum:
            if stage < config.train_max_digits and reached_threshold:
                _save_checkpoint(
                    _stage_checkpoint_path(directories["stage_checkpoints"], stage),
                    model,
                    optimizer,
                    metadata={
                        "global_step": global_step,
                        "stage": stage,
                        "validation_exact_match": validation["exact_match"],
                        "validation_digit_accuracy": validation["digit_accuracy"],
                        "validation_final_carry_accuracy": validation["final_carry_accuracy"],
                    },
                )
                print(
                    f"[addition curriculum] advance {stage} -> {stage + 1} "
                    f"(exact_match={validation['exact_match']:.4f})",
                    flush=True,
                )
                stage += 1
                stage_steps = 0
                continue
            if reached_cap and not reached_threshold:
                print(
                    f"[addition curriculum] hold stage={stage} after {stage_steps} steps "
                    f"(exact_match={validation['exact_match']:.4f} < threshold={config.stage_accuracy_threshold:.2f})",
                    flush=True,
                )
            if stage == config.train_max_digits and reached_threshold:
                stop_reason = "final_stage_threshold"
                break

    _save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        metadata={
            "global_step": global_step,
            "stage": stage,
            "stop_reason": stop_reason,
        },
    )

    best_payload = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state"])
    final_results = evaluate_suite(model=model, config=config, suite=suite, device=device)
    flat_final_metrics = flatten_nested_metrics("", final_results)
    summary = {
        "config": config.__dict__,
        "param_counts": param_counts,
        "best_validation_exact_match": best_validation,
        "global_step": global_step,
        "final_stage": stage,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.time() - started_at,
        "history": history,
        "final_results": final_results,
        "flat_final_metrics": flat_final_metrics,
    }
    _save_json(directories["artifacts"] / "summary.json", summary)
    with (directories["artifacts"] / "history.jsonl").open("w", encoding="utf-8") as handle:
        for entry in history:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    plot_single_run_results(summary, directories["plots"])

    if wandb_run is not None:
        wandb_run.log(flat_final_metrics | {"step": global_step})
        wandb_run.summary.update(
            {
                "best_validation_exact_match": best_validation,
                "final_stage": stage,
                "stop_reason": stop_reason,
            }
        )
        wandb_run.finish()

    return summary


def main() -> None:
    config = parse_config("Train the addition carry experiment.")
    run_experiment(config)


if __name__ == "__main__":
    main()
