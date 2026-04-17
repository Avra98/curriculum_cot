from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


VALID_MODELS = ("nocurr_nocot", "curr_nocot", "curr_cot")
VALID_PRESETS = ("default", "smoke")


@dataclass
class ExperimentConfig:
    model: str = "nocurr_nocot"
    output_dir: str = "addition_runs/default"
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    preset: str = "default"
    run_name: str = ""
    notes: str = ""
    use_wandb: bool = True
    wandb_project: str = "addition-carry"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    radix: int = 10
    train_max_digits: int = 12
    eval_max_digits: int = 20
    ood_lengths: tuple[int, ...] = (14, 16, 20)
    train_batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    carry_loss_weight: float = 0.0
    train_steps: int = 3600
    max_steps_per_stage: int = 300
    validation_interval: int = 100
    stage_accuracy_threshold: float = 0.99
    initial_stage: int = 1
    eval_examples_per_length: int = 256
    carry_heavy_examples_per_length: int = 256
    train_carry_heavy_prob: float = 0.15
    d_model: int = 512
    n_heads: int = 1
    ff_dim: int = 2048
    dropout: float = 0.0
    max_latent_steps: int = 12
    attention_probe_examples: int = 256
    linear_probe_epochs: int = 150
    linear_probe_lr: float = 1e-2
    comparison_num_seeds: int = 5

    def __post_init__(self) -> None:
        if self.model not in VALID_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
        if self.preset not in VALID_PRESETS:
            raise ValueError(f"Unsupported preset: {self.preset}")
        if self.train_max_digits > self.eval_max_digits:
            raise ValueError("train_max_digits must be <= eval_max_digits")
        if self.max_latent_steps < 0:
            raise ValueError("max_latent_steps must be non-negative")
        if self.radix < 2 or self.radix > 16:
            raise ValueError("radix must be between 2 and 16")
        if self.initial_stage < 1 or self.initial_stage > self.train_max_digits:
            raise ValueError("initial_stage must be between 1 and train_max_digits")
        self.ood_lengths = tuple(int(v) for v in self.ood_lengths if int(v) > self.train_max_digits)
        if not self.ood_lengths:
            self.ood_lengths = (self.eval_max_digits,)

    @property
    def uses_curriculum(self) -> bool:
        return self.model in {"curr_nocot", "curr_cot"}

    @property
    def uses_latent_cot(self) -> bool:
        return self.model == "curr_cot"

    @property
    def discrete_vocab_size(self) -> int:
        return self.radix + 2

    @property
    def digit_vocab_size(self) -> int:
        return self.radix

    @property
    def input_sequence_length(self) -> int:
        return self.input_sequence_length_for_digits(self.eval_max_digits)

    @property
    def output_sequence_length(self) -> int:
        return self.output_sequence_length_for_digits(self.eval_max_digits)

    @property
    def base_sequence_length(self) -> int:
        return self.base_sequence_length_for_digits(self.eval_max_digits)

    @property
    def max_sequence_length(self) -> int:
        return self.base_sequence_length + self.max_latent_steps

    @property
    def effective_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.model}_base{self.radix}_seed{self.seed}"

    def input_sequence_length_for_digits(self, active_digits: int) -> int:
        return (int(active_digits) * 2) + 2

    def output_sequence_length_for_digits(self, active_digits: int) -> int:
        return int(active_digits) + 1

    def base_sequence_length_for_digits(self, active_digits: int) -> int:
        return self.input_sequence_length_for_digits(active_digits) + self.output_sequence_length_for_digits(active_digits)

    def latent_steps_for_stage(self, stage: int) -> int:
        if not self.uses_latent_cot:
            return 0
        return max(0, min(int(stage), int(self.max_latent_steps)))


def default_output_root() -> Path:
    return Path("addition_runs")


def apply_preset(config: ExperimentConfig) -> ExperimentConfig:
    config = dataclasses.replace(config)
    if config.preset == "smoke":
        config.output_dir = config.output_dir or str(default_output_root() / "smoke")
        config.train_batch_size = 64
        config.eval_batch_size = 128
        config.d_model = 128
        config.n_heads = 1
        config.ff_dim = 512
        config.train_steps = 180
        config.max_steps_per_stage = 40
        config.validation_interval = 20
        config.eval_examples_per_length = 64
        config.carry_heavy_examples_per_length = 64
        config.attention_probe_examples = 64
        config.linear_probe_epochs = 60
        config.comparison_num_seeds = 2
    return config


def config_to_dict(config: ExperimentConfig) -> dict:
    data = dataclasses.asdict(config)
    data["ood_lengths"] = list(config.ood_lengths)
    data["uses_curriculum"] = config.uses_curriculum
    data["uses_latent_cot"] = config.uses_latent_cot
    data["discrete_vocab_size"] = config.discrete_vocab_size
    data["input_sequence_length"] = config.input_sequence_length
    data["output_sequence_length"] = config.output_sequence_length
    data["base_sequence_length"] = config.base_sequence_length
    data["max_sequence_length"] = config.max_sequence_length
    data["effective_run_name"] = config.effective_run_name
    return data


def save_config(config: ExperimentConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2, sort_keys=True)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=VALID_MODELS, default="nocurr_nocot")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=VALID_PRESETS, default="default")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="addition-carry")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--radix", type=int, default=10)
    parser.add_argument("--train_max_digits", type=int, default=12)
    parser.add_argument("--eval_max_digits", type=int, default=20)
    parser.add_argument("--ood_lengths", type=int, nargs="*", default=[14, 16, 20])
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--carry_loss_weight", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=3600)
    parser.add_argument("--max_steps_per_stage", type=int, default=300)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--stage_accuracy_threshold", type=float, default=0.99)
    parser.add_argument("--initial_stage", type=int, default=1)
    parser.add_argument("--eval_examples_per_length", type=int, default=256)
    parser.add_argument("--carry_heavy_examples_per_length", type=int, default=256)
    parser.add_argument("--train_carry_heavy_prob", type=float, default=0.15)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_latent_steps", type=int, default=12)
    parser.add_argument("--attention_probe_examples", type=int, default=256)
    parser.add_argument("--linear_probe_epochs", type=int, default=150)
    parser.add_argument("--linear_probe_lr", type=float, default=1e-2)
    parser.add_argument("--comparison_num_seeds", type=int, default=5)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    use_wandb = bool(args.use_wandb or not args.no_wandb)
    if args.wandb_mode == "disabled":
        use_wandb = False
    output_dir = args.output_dir or str(default_output_root() / f"{args.model}_base{args.radix}_seed{args.seed}")
    config = ExperimentConfig(
        model=args.model,
        output_dir=output_dir,
        seed=args.seed,
        device=args.device,
        preset=args.preset,
        run_name=args.run_name,
        notes=args.notes,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        radix=args.radix,
        train_max_digits=args.train_max_digits,
        eval_max_digits=args.eval_max_digits,
        ood_lengths=tuple(args.ood_lengths),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        carry_loss_weight=args.carry_loss_weight,
        train_steps=args.train_steps,
        max_steps_per_stage=args.max_steps_per_stage,
        validation_interval=args.validation_interval,
        stage_accuracy_threshold=args.stage_accuracy_threshold,
        initial_stage=args.initial_stage,
        eval_examples_per_length=args.eval_examples_per_length,
        carry_heavy_examples_per_length=args.carry_heavy_examples_per_length,
        train_carry_heavy_prob=args.train_carry_heavy_prob,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_latent_steps=args.max_latent_steps,
        attention_probe_examples=args.attention_probe_examples,
        linear_probe_epochs=args.linear_probe_epochs,
        linear_probe_lr=args.linear_probe_lr,
        comparison_num_seeds=args.comparison_num_seeds,
    )
    return apply_preset(config)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    return parser


def parse_config(description: str) -> ExperimentConfig:
    parser = build_arg_parser(description)
    args = parser.parse_args()
    return build_config_from_args(args)


def ensure_output_dirs(config: ExperimentConfig) -> dict[str, Path]:
    root = Path(config.output_dir)
    directories = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "stage_checkpoints": root / "checkpoints" / "stages",
        "plots": root / "plots",
        "artifacts": root / "artifacts",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def flatten_metric_dict(prefix: str, metrics: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def iter_stage_lengths(config: ExperimentConfig) -> Iterable[int]:
    for stage in range(1, config.train_max_digits + 1):
        yield stage
