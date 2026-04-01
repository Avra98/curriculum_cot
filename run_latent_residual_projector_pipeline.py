from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkpoint_utils import final_checkpoint_root, normalize_to_final_checkpoint_root


CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
DEFAULT_CHECKPOINT_ROOT = Path(final_checkpoint_root("latent_multi_output_cell_policy"))
DEFAULT_CACHE_DIR = Path("/home/ubuntu/curriculum-CoT/.hf_cache")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SFT_SCRIPT = CURRENT_DIR / "latent_multi_output_cell_policy" / "residual_projector_warmstart_sft_latent_multi_output_train.py"
GRPO_SCRIPT = CURRENT_DIR / "latent_multi_output_cell_policy" / "grpo_residual_projector_latent_train.py"
STAGE_COMPLETE_MARKER = "_stage_complete.json"


@dataclass
class Artifact:
    path: str
    stage: int
    phase: str
    step: int
    mtime: float
    source_dir: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--python_executable", type=str, default=sys.executable)
    p.add_argument("--checkpoint_root", type=str, default=str(DEFAULT_CHECKPOINT_ROOT))
    p.add_argument("--output_root", type=str, default="")
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--train_jsonl", type=str, default="")
    p.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_empties_hint", type=int, default=20)
    p.add_argument("--min_stage", type=int, default=1)
    p.add_argument("--max_stage", type=int, default=3)
    p.add_argument("--sft_gpu_id", type=int, default=0)
    p.add_argument("--grpo_gpu_id", type=int, default=1)
    p.add_argument("--stage1_init_adapter_dir", type=str, default="")
    p.add_argument("--bootstrap_adapter_dir", type=str, default="")
    p.add_argument("--distributed_gpu_ids", type=str, default="")
    p.add_argument("--sft_num_processes", type=int, default=1)
    p.add_argument("--grpo_num_processes", type=int, default=1)
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="latent_residual_projector_pipeline")
    p.add_argument("--sft_num_epochs", type=float, default=1.0)
    p.add_argument("--sft_learning_rate_stage1", type=float, default=1e-6)
    p.add_argument("--sft_learning_rate_later", type=float, default=1e-6)
    p.add_argument("--sft_gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--sft_enable_gradient_checkpointing", action="store_true")
    p.add_argument("--sft_logging_steps", type=int, default=10)
    p.add_argument("--sft_eval_steps", type=int, default=100)
    p.add_argument("--sft_save_steps", type=int, default=100)
    p.add_argument("--sft_eval_rows", type=int, default=20)
    p.add_argument("--sft_max_completion_length", type=int, default=24)
    p.add_argument("--grpo_num_train_epochs", type=float, default=1.0)
    p.add_argument("--grpo_learning_rate", type=float, default=1e-6)
    p.add_argument("--grpo_per_device_train_batch_size", type=int, default=4)
    p.add_argument("--grpo_gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--grpo_enable_gradient_checkpointing", action="store_true")
    p.add_argument("--grpo_logging_steps", type=int, default=5)
    p.add_argument("--grpo_eval_steps", type=int, default=25)
    p.add_argument("--grpo_save_steps", type=int, default=25)
    p.add_argument("--grpo_eval_rows", type=int, default=20)
    p.add_argument("--grpo_num_generations", type=int, default=2)
    p.add_argument("--grpo_max_prompt_length", type=int, default=1024)
    p.add_argument("--grpo_max_completion_length", type=int, default=24)
    p.add_argument("--grpo_beta", type=float, default=0.0)
    p.add_argument("--phase_max_wall_clock_seconds", type=int, default=21600)
    p.add_argument("--limit_train_rows", type=int, default=0)
    p.add_argument("--sft_stage_max_steps", type=str, default="")
    p.add_argument("--grpo_stage_max_steps", type=str, default="")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def stage_dir_pattern(stage: int, phase: str, empties: int) -> str:
    return f"stage{stage:02d}_{phase}_i{stage}_{empties}empty*"


def extract_numeric_suffix(name: str, prefix: str) -> Optional[int]:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
    return int(match.group(1)) if match else None


def stage_complete_path(stage_dir: Path) -> Path:
    return stage_dir / STAGE_COMPLETE_MARKER


def is_stage_complete(stage_dir: Path) -> bool:
    return stage_complete_path(stage_dir).is_file()


def output_root_has_stage_artifacts(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "pipeline_state.json").exists():
        return True
    return any(path.glob("stage[0-9][0-9]_*"))


def latest_sft_checkpoint(stage_dir: Path) -> Optional[Artifact]:
    best: Optional[Artifact] = None
    for child in stage_dir.iterdir():
        if not child.is_dir():
            continue
        step = extract_numeric_suffix(child.name, "checkpoint-step-")
        if step is None:
            continue
        artifact = Artifact(
            path=str(child),
            stage=-1,
            phase="sft",
            step=step,
            mtime=child.stat().st_mtime,
            source_dir=str(stage_dir),
        )
        if best is None or (artifact.mtime, artifact.step) > (best.mtime, best.step):
            best = artifact
    return best


def latest_grpo_artifact(stage_dir: Path) -> Optional[Artifact]:
    best: Optional[Artifact] = None
    root_adapter = stage_dir / "adapter_model.safetensors"
    if root_adapter.exists():
        best = Artifact(
            path=str(stage_dir),
            stage=-1,
            phase="grpo",
            step=10**9,
            mtime=stage_dir.stat().st_mtime,
            source_dir=str(stage_dir),
        )
    for child in stage_dir.iterdir():
        if not child.is_dir():
            continue
        step = extract_numeric_suffix(child.name, "checkpoint-")
        if step is None:
            continue
        adapter = child / "adapter_model.safetensors"
        latent_state = child / "latent_cot_state.pt"
        if not adapter.exists() or not latent_state.exists():
            continue
        artifact = Artifact(
            path=str(child),
            stage=-1,
            phase="grpo",
            step=step,
            mtime=child.stat().st_mtime,
            source_dir=str(stage_dir),
        )
        if best is None or (artifact.mtime, artifact.step) > (best.mtime, best.step):
            best = artifact
    return best


def discover_latest_artifact(
    search_root: Path,
    *,
    stage: int,
    phase: str,
    empties: int,
    require_complete: bool = True,
) -> Optional[Artifact]:
    if not search_root.exists():
        return None
    best: Optional[Artifact] = None
    for stage_dir in search_root.rglob(stage_dir_pattern(stage, phase, empties)):
        if not stage_dir.is_dir():
            continue
        if require_complete and not is_stage_complete(stage_dir):
            continue
        artifact = latest_sft_checkpoint(stage_dir) if phase == "sft" else latest_grpo_artifact(stage_dir)
        if artifact is None:
            continue
        artifact.stage = stage
        artifact.phase = phase
        if best is None or (artifact.mtime, artifact.step) > (best.mtime, best.step):
            best = artifact
    return best


def choose_output_root(args: argparse.Namespace, checkpoint_root: Path) -> Path:
    if str(args.output_root).strip():
        requested_root = Path(normalize_to_final_checkpoint_root(args.output_root, "latent_multi_output_cell_policy")).resolve()
        if output_root_has_stage_artifacts(requested_root):
            run_tag = str(args.run_tag).strip() or time.strftime("%Y%m%d_%H%M%S")
            return requested_root / run_tag
        return requested_root
    run_tag = str(args.run_tag).strip() or time.strftime("%Y%m%d_%H%M%S")
    return checkpoint_root / run_tag / f"latent_pipeline_{args.total_empties_hint}empty_{args.max_stage}stage"


def default_train_jsonl(args: argparse.Namespace) -> Path:
    if str(args.train_jsonl).strip():
        return Path(args.train_jsonl).resolve()
    return (PARENT_DIR / "data" / f"sudoku_t3_{int(args.total_empties_hint)}empty_value_qwen_text.jsonl").resolve()


def phase_output_dir(output_root: Path, *, stage: int, phase: str, empties: int) -> Path:
    return output_root / f"stage{stage:02d}_{phase}_i{stage}_{empties}empty_residual_projector"


def run_command(command: List[str], *, env: Dict[str, str], dry_run: bool) -> None:
    print("")
    print("Running command:")
    print(" ".join(subprocess.list2cmdline([part]) for part in command))
    if dry_run:
        print("Dry run enabled; command not executed.")
        return
    subprocess.run(command, env=env, check=True)


def parse_stage_int_map(raw: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    text = str(raw or "").strip()
    if not text:
        return mapping
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        stage_text, value_text = item.split(":", 1)
        mapping[int(stage_text.strip())] = int(value_text.strip())
    return mapping


def resolve_stage_value(mapping: Dict[int, int], stage: int) -> int:
    return int(mapping.get(int(stage), 0))


def make_env(*, gpu_id: int, wandb_mode: str, gpu_ids: str, num_processes: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    requested = [part.strip() for part in str(gpu_ids or "").split(",") if part.strip()]
    if int(num_processes) > 1:
        if requested:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(requested[: int(num_processes)])
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(requested[0] if requested else int(gpu_id))
    env["WANDB__SERVICE_WAIT"] = "300"
    env["WANDB_MODE"] = str(wandb_mode)
    return env


def build_sft_command(
    args: argparse.Namespace,
    *,
    train_jsonl: Path,
    output_dir: Path,
    stage: int,
    init_adapter_dir: str,
    stage_max_steps: int,
) -> List[str]:
    num_processes = max(1, int(args.sft_num_processes))
    if num_processes > 1:
        command = [
            args.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_processes),
            str(SFT_SCRIPT),
        ]
    else:
        command = [args.python_executable, "-u", str(SFT_SCRIPT)]
    command.extend(
        [
            "--model_name",
            args.model_name,
            "--train_jsonl",
            str(train_jsonl),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            args.cache_dir,
            "--init_adapter_dir",
            str(init_adapter_dir),
            "--seed",
            str(int(args.seed)),
            "--gpu_id",
            str(0 if num_processes > 1 else int(args.sft_gpu_id)),
            "--stage_i",
            str(int(stage)),
            "--num_cot_tokens",
            str(int(stage)),
            "--total_empties_hint",
            str(int(args.total_empties_hint)),
            "--num_epochs",
            str(float(args.sft_num_epochs)),
            "--learning_rate",
            str(float(args.sft_learning_rate_stage1 if stage <= 1 else args.sft_learning_rate_later)),
            "--gradient_accumulation_steps",
            str(int(args.sft_gradient_accumulation_steps)),
            "--enable_gradient_checkpointing" if args.sft_enable_gradient_checkpointing else "",
            "--logging_steps",
            str(int(args.sft_logging_steps)),
            "--save_steps",
            str(int(args.sft_save_steps)),
            "--eval_steps",
            str(int(args.sft_eval_steps)),
            "--eval_rows",
            str(int(args.sft_eval_rows)),
            "--max_completion_length",
            str(int(args.sft_max_completion_length)),
            "--max_wall_clock_seconds",
            str(int(args.phase_max_wall_clock_seconds)),
        ]
    )
    command = [part for part in command if part != ""]
    if int(args.limit_train_rows) > 0:
        command.extend(["--limit_train_rows", str(int(args.limit_train_rows))])
    if int(stage_max_steps) > 0:
        command.extend(["--max_steps", str(int(stage_max_steps))])
    if args.use_wandb:
        command.extend(
            [
                "--use_wandb",
                "--wandb_entity",
                args.wandb_entity,
                "--wandb_project",
                "sudoku-latent-multi-output-sft-residual-projector",
                "--wandb_run_name",
                f"latent_stage{stage:02d}_sft_i{stage}_{args.total_empties_hint}empty_residual_projector",
                "--wandb_mode",
                args.wandb_mode,
            ]
        )
    return command


def build_grpo_command(
    args: argparse.Namespace,
    *,
    train_jsonl: Path,
    output_dir: Path,
    stage: int,
    init_adapter_dir: str,
    stage_max_steps: int,
) -> List[str]:
    num_processes = max(1, int(args.grpo_num_processes))
    if num_processes > 1:
        command = [
            args.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_processes),
            str(GRPO_SCRIPT),
        ]
    else:
        command = [args.python_executable, "-u", str(GRPO_SCRIPT)]
    command.extend(
        [
            "--model_name",
            args.model_name,
            "--train_jsonl",
            str(train_jsonl),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            args.cache_dir,
            "--init_adapter_dir",
            str(init_adapter_dir),
            "--seed",
            str(int(args.seed)),
            "--gpu_id",
            str(0 if num_processes > 1 else int(args.grpo_gpu_id)),
            "--stage_i",
            str(int(stage)),
            "--num_cot_tokens",
            str(int(stage)),
            "--total_empties_hint",
            str(int(args.total_empties_hint)),
            "--per_device_train_batch_size",
            str(int(args.grpo_per_device_train_batch_size)),
            "--gradient_accumulation_steps",
            str(int(args.grpo_gradient_accumulation_steps)),
            "--enable_gradient_checkpointing" if args.grpo_enable_gradient_checkpointing else "",
            "--num_train_epochs",
            str(float(args.grpo_num_train_epochs)),
            "--learning_rate",
            str(float(args.grpo_learning_rate)),
            "--logging_steps",
            str(int(args.grpo_logging_steps)),
            "--save_steps",
            str(int(args.grpo_save_steps)),
            "--eval_steps",
            str(int(args.grpo_eval_steps)),
            "--eval_rows",
            str(int(args.grpo_eval_rows)),
            "--num_generations",
            str(int(args.grpo_num_generations)),
            "--max_prompt_length",
            str(int(args.grpo_max_prompt_length)),
            "--max_completion_length",
            str(int(args.grpo_max_completion_length)),
            "--beta",
            str(float(args.grpo_beta)),
            "--max_wall_clock_seconds",
            str(int(args.phase_max_wall_clock_seconds)),
            "--wandb_group",
            args.wandb_group,
        ]
    )
    command = [part for part in command if part != ""]
    if int(args.limit_train_rows) > 0:
        command.extend(["--limit_train_rows", str(int(args.limit_train_rows))])
    if int(stage_max_steps) > 0:
        command.extend(["--max_steps", str(int(stage_max_steps))])
    if args.use_wandb:
        command.extend(
            [
                "--use_wandb",
                "--wandb_entity",
                args.wandb_entity,
                "--wandb_project",
                "sudoku-latent-multi-output-grpo-residual-projector",
                "--wandb_run_name",
                f"latent_stage{stage:02d}_grpo_i{stage}_{args.total_empties_hint}empty_residual_projector",
                "--wandb_mode",
                args.wandb_mode,
            ]
        )
    return command


def write_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def mark_stage_complete(stage_dir: Path, artifact: Artifact) -> None:
    write_state(
        stage_complete_path(stage_dir),
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "artifact": asdict(artifact),
        },
    )


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(normalize_to_final_checkpoint_root(args.checkpoint_root, "latent_multi_output_cell_policy")).resolve()
    output_root = choose_output_root(args, checkpoint_root)
    train_jsonl = default_train_jsonl(args)
    state_path = output_root / "pipeline_state.json"
    sft_stage_max_steps = parse_stage_int_map(args.sft_stage_max_steps)
    grpo_stage_max_steps = parse_stage_int_map(args.grpo_stage_max_steps)

    output_root.mkdir(parents=True, exist_ok=True)
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Missing train_jsonl: {train_jsonl}")

    state: Dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_jsonl": str(train_jsonl),
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "min_stage": int(args.min_stage),
        "max_stage": int(args.max_stage),
        "total_empties_hint": int(args.total_empties_hint),
        "stages": [],
    }

    previous_grpo: Optional[Artifact] = None
    for stage in range(int(args.min_stage), int(args.max_stage) + 1):
        stage_record: Dict[str, Any] = {"stage": stage}
        existing_sft = discover_latest_artifact(output_root, stage=stage, phase="sft", empties=int(args.total_empties_hint))
        existing_grpo = discover_latest_artifact(
            output_root, stage=stage, phase="grpo", empties=int(args.total_empties_hint)
        )

        if existing_grpo is not None:
            previous_grpo = existing_grpo
            stage_record["status"] = "using_existing_grpo"
            stage_record["grpo_artifact"] = asdict(existing_grpo)
            if existing_sft is not None:
                stage_record["sft_artifact"] = asdict(existing_sft)
            state["stages"].append(stage_record)
            write_state(state_path, state)
            print(f"Stage {stage}: using existing GRPO artifact {existing_grpo.path}")
            continue

        if existing_sft is None:
            sft_output_dir = phase_output_dir(output_root, stage=stage, phase="sft", empties=int(args.total_empties_hint))
            if stage == int(args.min_stage) and str(args.bootstrap_adapter_dir).strip():
                init_adapter_dir = str(args.bootstrap_adapter_dir).strip()
            elif stage == 1:
                init_adapter_dir = str(args.stage1_init_adapter_dir).strip()
            else:
                if previous_grpo is None:
                    raise RuntimeError(f"Missing previous GRPO artifact needed to launch latent stage {stage} SFT.")
                init_adapter_dir = previous_grpo.path
            print(f"Stage {stage}: launching latent SFT into {sft_output_dir}")
            run_command(
                build_sft_command(
                    args,
                    train_jsonl=train_jsonl,
                    output_dir=sft_output_dir,
                    stage=stage,
                    init_adapter_dir=init_adapter_dir,
                    stage_max_steps=resolve_stage_value(sft_stage_max_steps, stage),
                ),
                env=make_env(
                    gpu_id=int(args.sft_gpu_id),
                    wandb_mode=args.wandb_mode,
                    gpu_ids=args.distributed_gpu_ids,
                    num_processes=int(args.sft_num_processes),
                ),
                dry_run=bool(args.dry_run),
            )
            existing_sft = discover_latest_artifact(
                output_root,
                stage=stage,
                phase="sft",
                empties=int(args.total_empties_hint),
                require_complete=False,
            )
            if existing_sft is None and not args.dry_run:
                raise RuntimeError(f"Could not locate latent SFT checkpoint for stage {stage} after running SFT.")
            if existing_sft is not None:
                mark_stage_complete(Path(existing_sft.source_dir), existing_sft)
                stage_record["sft_artifact"] = asdict(existing_sft)
        else:
            stage_record["sft_artifact"] = asdict(existing_sft)
            print(f"Stage {stage}: using existing latent SFT artifact {existing_sft.path}")

        if existing_sft is None:
            stage_record["status"] = "dry_run_pending_grpo"
            state["stages"].append(stage_record)
            write_state(state_path, state)
            break

        grpo_output_dir = phase_output_dir(output_root, stage=stage, phase="grpo", empties=int(args.total_empties_hint))
        print(f"Stage {stage}: launching latent GRPO into {grpo_output_dir}")
        run_command(
            build_grpo_command(
                args,
                train_jsonl=train_jsonl,
                output_dir=grpo_output_dir,
                stage=stage,
                init_adapter_dir=existing_sft.path,
                stage_max_steps=resolve_stage_value(grpo_stage_max_steps, stage),
            ),
            env=make_env(
                gpu_id=int(args.grpo_gpu_id),
                wandb_mode=args.wandb_mode,
                gpu_ids=args.distributed_gpu_ids,
                num_processes=int(args.grpo_num_processes),
            ),
            dry_run=bool(args.dry_run),
        )
        existing_grpo = discover_latest_artifact(
            output_root,
            stage=stage,
            phase="grpo",
            empties=int(args.total_empties_hint),
            require_complete=False,
        )
        if existing_grpo is None and not args.dry_run:
            raise RuntimeError(f"Could not locate latent GRPO artifact for stage {stage} after running GRPO.")
        if existing_grpo is not None:
            mark_stage_complete(Path(existing_grpo.source_dir), existing_grpo)
            previous_grpo = existing_grpo
            stage_record["grpo_artifact"] = asdict(existing_grpo)
        stage_record["status"] = "launched"
        state["stages"].append(stage_record)
        write_state(state_path, state)

    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_state(state_path, state)
    print("")
    print(f"Pipeline state written to: {state_path}")


if __name__ == "__main__":
    main()
