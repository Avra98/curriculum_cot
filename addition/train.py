from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from addition.config import ExperimentConfig, ensure_output_dirs, parse_config, save_config
from addition.data import build_evaluation_suite, sample_training_batch, seed_everything
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
        "carry_accuracy": stage_metrics.carry_accuracy,
        "exact_match": stage_metrics.exact_match,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    directories = ensure_output_dirs(config)
    save_config(config, directories["root"])
    seed_everything(config.seed)
    device = config.device
    model = build_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    digit_loss_fn = nn.CrossEntropyLoss()
    carry_loss_fn = nn.CrossEntropyLoss()
    suite = build_evaluation_suite(config)
    rng = __import__("random").Random(config.seed + 12345)
    history: list[dict[str, Any]] = []
    best_validation = -1.0
    best_checkpoint_path = directories["checkpoints"] / "best.pt"
    last_checkpoint_path = directories["checkpoints"] / "last.pt"
    stage = 1 if config.uses_curriculum else config.train_max_digits
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

    while global_step < config.train_steps:
        model.train()
        batch = sample_training_batch(config=config, stage=stage, rng=rng, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.input_ids, latent_steps=config.latent_steps_for_stage(stage), return_attention=False)
        digit_loss = digit_loss_fn(outputs.digit_logits, batch.target_digits)
        carry_loss = carry_loss_fn(outputs.carry_logits, batch.target_carry)
        loss = digit_loss + (config.carry_loss_weight * carry_loss)
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        global_step += 1
        stage_steps += 1

        if global_step % max(1, config.validation_interval // 2) == 0:
            print(
                f"[addition train] step={global_step} stage={stage} "
                f"loss={loss.item():.4f} digit_loss={digit_loss.item():.4f} carry_loss={carry_loss.item():.4f}",
                flush=True,
            )

        should_validate = (
            global_step % config.validation_interval == 0
            or global_step == config.train_steps
            or stage_steps >= config.max_steps_per_stage
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
            "carry_loss": float(carry_loss.item()),
            "validation_digit_accuracy": validation["digit_accuracy"],
            "validation_carry_accuracy": validation["carry_accuracy"],
            "validation_exact_match": validation["exact_match"],
            "latent_steps": config.latent_steps_for_stage(stage),
        }
        history.append(history_entry)
        print(
            f"[addition val] step={global_step} stage={stage} "
            f"digit_acc={validation['digit_accuracy']:.4f} carry_acc={validation['carry_accuracy']:.4f} "
            f"exact={validation['exact_match']:.4f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": float(loss.item()),
                    "train/digit_loss": float(digit_loss.item()),
                    "train/carry_loss": float(carry_loss.item()),
                    "train/stage": stage,
                    "train/latent_steps": config.latent_steps_for_stage(stage),
                    "validation/digit_accuracy": validation["digit_accuracy"],
                    "validation/carry_accuracy": validation["carry_accuracy"],
                    "validation/exact_match": validation["exact_match"],
                    "step": global_step,
                }
            )

        if validation["digit_accuracy"] >= best_validation:
            best_validation = validation["digit_accuracy"]
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                metadata={
                    "global_step": global_step,
                    "stage": stage,
                    "best_validation_digit_accuracy": best_validation,
                },
            )

        reached_threshold = validation["digit_accuracy"] >= config.stage_accuracy_threshold
        reached_cap = stage_steps >= config.max_steps_per_stage

        if config.uses_curriculum:
            if stage < config.train_max_digits and (reached_threshold or reached_cap):
                print(
                    f"[addition curriculum] advance {stage} -> {stage + 1} "
                    f"(threshold={reached_threshold} cap={reached_cap})",
                    flush=True,
                )
                stage += 1
                stage_steps = 0
                continue
            if stage == config.train_max_digits and (reached_threshold or reached_cap):
                stop_reason = "final_stage_threshold" if reached_threshold else "final_stage_cap"
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
        "best_validation_digit_accuracy": best_validation,
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
                "best_validation_digit_accuracy": best_validation,
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
