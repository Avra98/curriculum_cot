from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch

from addition.config import ExperimentConfig


DIGIT_OFFSET = 0
DEFAULT_SYMBOLS = "0123456789ABCDEF"


@dataclass
class AdditionProblem:
    a_digits: list[int]
    b_digits: list[int]
    sum_digits: list[int]
    carry_out: list[int]
    active_digits: int
    is_carry_heavy: bool


@dataclass
class Batch:
    input_ids: torch.Tensor
    target_digits: torch.Tensor
    target_digit_mask: torch.Tensor
    target_carry: torch.Tensor
    target_final_carry: torch.Tensor
    active_digits: torch.Tensor
    is_carry_heavy: torch.Tensor


@dataclass
class EvaluationSuite:
    validation_uniform: dict[int, list[AdditionProblem]]
    test_uniform: dict[int, list[AdditionProblem]]
    test_carry_heavy: dict[int, list[AdditionProblem]]


def a_token_id(radix: int) -> int:
    return radix


def b_token_id(radix: int) -> int:
    return radix + 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sum_and_carry(a_digits: list[int], b_digits: list[int], radix: int) -> tuple[list[int], list[int]]:
    sum_digits: list[int] = []
    carry_out: list[int] = []
    carry = 0
    for a_digit, b_digit in zip(a_digits, b_digits):
        total = int(a_digit) + int(b_digit) + carry
        sum_digits.append(total % radix)
        carry = total // radix
        carry_out.append(carry)
    return sum_digits, carry_out


def sample_uniform_problem(max_digits: int, active_digits: int, radix: int, rng: random.Random) -> AdditionProblem:
    a_digits = [0] * max_digits
    b_digits = [0] * max_digits
    for index in range(active_digits):
        a_digits[index] = rng.randint(0, radix - 1)
        b_digits[index] = rng.randint(0, radix - 1)
    sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits, radix=radix)
    return AdditionProblem(
        a_digits=a_digits,
        b_digits=b_digits,
        sum_digits=sum_digits,
        carry_out=carry_out,
        active_digits=active_digits,
        is_carry_heavy=False,
    )


def sample_carry_heavy_problem(max_digits: int, active_digits: int, radix: int, rng: random.Random) -> AdditionProblem:
    a_digits = [0] * max_digits
    b_digits = [0] * max_digits
    carry = 0
    for index in range(active_digits):
        high_floor = max(0, radix // 2)
        a_digit = rng.randint(high_floor, radix - 1)
        if carry == 0:
            min_b = max(0, radix - a_digit)
        else:
            min_b = max(0, (radix - 1) - a_digit)
        b_digit = rng.randint(min_b, radix - 1)
        a_digits[index] = a_digit
        b_digits[index] = b_digit
        total = a_digit + b_digit + carry
        carry = total // radix
    sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits, radix=radix)
    return AdditionProblem(
        a_digits=a_digits,
        b_digits=b_digits,
        sum_digits=sum_digits,
        carry_out=carry_out,
        active_digits=active_digits,
        is_carry_heavy=True,
    )


def sample_problem(
    max_digits: int,
    active_digits: int,
    radix: int,
    rng: random.Random,
    carry_heavy: bool = False,
) -> AdditionProblem:
    if carry_heavy:
        return sample_carry_heavy_problem(max_digits=max_digits, active_digits=active_digits, radix=radix, rng=rng)
    return sample_uniform_problem(max_digits=max_digits, active_digits=active_digits, radix=radix, rng=rng)


def encode_problem_tokens(problem: AdditionProblem, radix: int) -> list[int]:
    return (
        [a_token_id(radix)]
        + [DIGIT_OFFSET + digit for digit in problem.a_digits[: problem.active_digits]]
        + [b_token_id(radix)]
        + [DIGIT_OFFSET + digit for digit in problem.b_digits[: problem.active_digits]]
    )


def build_batch(
    problems: list[AdditionProblem],
    radix: int,
    device: str,
) -> Batch:
    active_digits = problems[0].active_digits if problems else 0
    input_ids = torch.tensor(
        [
            encode_problem_tokens(problem=problem, radix=radix)
            for problem in problems
        ],
        dtype=torch.long,
        device=device,
    )
    target_digits = torch.tensor(
        [problem.sum_digits[:active_digits] for problem in problems],
        dtype=torch.long,
        device=device,
    )
    target_digit_mask = torch.tensor(
        [[1] * active_digits for _ in problems],
        dtype=torch.bool,
        device=device,
    )
    target_carry = torch.tensor(
        [problem.carry_out[:active_digits] for problem in problems],
        dtype=torch.long,
        device=device,
    )
    target_final_carry = torch.tensor(
        [problem.carry_out[problem.active_digits - 1] for problem in problems],
        dtype=torch.long,
        device=device,
    )
    return Batch(
        input_ids=input_ids,
        target_digits=target_digits,
        target_digit_mask=target_digit_mask,
        target_carry=target_carry,
        target_final_carry=target_final_carry,
        active_digits=torch.tensor([problem.active_digits for problem in problems], dtype=torch.long, device=device),
        is_carry_heavy=torch.tensor([int(problem.is_carry_heavy) for problem in problems], dtype=torch.bool, device=device),
    )


def sample_training_batch(
    config: ExperimentConfig,
    stage: int,
    rng: random.Random,
    device: str,
) -> Batch:
    problems: list[AdditionProblem] = []
    for _ in range(config.train_batch_size):
        carry_heavy = rng.random() < config.train_carry_heavy_prob
        problem = sample_problem(
            max_digits=stage,
            active_digits=stage,
            radix=config.radix,
            rng=rng,
            carry_heavy=carry_heavy,
        )
        problems.append(problem)
    return build_batch(
        problems=problems,
        radix=config.radix,
        device=device,
    )


def build_problem_set(
    *,
    max_digits: int,
    active_digits: int,
    radix: int,
    count: int,
    seed: int,
    carry_heavy: bool,
) -> list[AdditionProblem]:
    rng = random.Random(seed)
    return [
        sample_problem(max_digits=max_digits, active_digits=active_digits, radix=radix, rng=rng, carry_heavy=carry_heavy)
        for _ in range(count)
    ]


def build_evaluation_suite(config: ExperimentConfig) -> EvaluationSuite:
    validation_uniform: dict[int, list[AdditionProblem]] = {}
    test_uniform: dict[int, list[AdditionProblem]] = {}
    test_carry_heavy: dict[int, list[AdditionProblem]] = {}
    all_lengths = sorted(set(range(1, config.train_max_digits + 1)).union(config.ood_lengths))
    for length in all_lengths:
        validation_uniform[length] = build_problem_set(
            max_digits=length,
            active_digits=length,
            radix=config.radix,
            count=config.eval_examples_per_length,
            seed=10_000 + length,
            carry_heavy=False,
        )
        test_uniform[length] = build_problem_set(
            max_digits=length,
            active_digits=length,
            radix=config.radix,
            count=config.eval_examples_per_length,
            seed=20_000 + length,
            carry_heavy=False,
        )
        test_carry_heavy[length] = build_problem_set(
            max_digits=length,
            active_digits=length,
            radix=config.radix,
            count=config.carry_heavy_examples_per_length,
            seed=30_000 + length,
            carry_heavy=True,
        )
    return EvaluationSuite(
        validation_uniform=validation_uniform,
        test_uniform=test_uniform,
        test_carry_heavy=test_carry_heavy,
    )


def digits_to_string(digits: Iterable[int], final_carry: int, radix: int) -> str:
    digits = list(digits)
    significant_digits = list(digits)
    if final_carry:
        significant_digits.append(final_carry)
    while len(significant_digits) > 1 and significant_digits[-1] == 0:
        significant_digits.pop()
    symbols = DEFAULT_SYMBOLS[:radix]
    return "".join(symbols[digit] for digit in reversed(significant_digits))


def value_from_digits(digits: Iterable[int], final_carry: int, radix: int) -> int:
    value = 0
    place = 1
    for digit in digits:
        value += int(digit) * place
        place *= radix
    if final_carry:
        value += int(final_carry) * place
    return value


def exact_sum_matches(
    predicted_digits: list[int],
    predicted_final_carry: int,
    truth_digits: list[int],
    truth_final_carry: int,
) -> bool:
    return predicted_digits == truth_digits and int(predicted_final_carry) == int(truth_final_carry)


def summarize_problem(problem: AdditionProblem, radix: int) -> dict[str, int | str]:
    final_carry = problem.carry_out[problem.active_digits - 1]
    return {
        "a": digits_to_string(problem.a_digits[: problem.active_digits], final_carry=0, radix=radix),
        "b": digits_to_string(problem.b_digits[: problem.active_digits], final_carry=0, radix=radix),
        "sum": digits_to_string(problem.sum_digits[: problem.active_digits], final_carry=final_carry, radix=radix),
        "radix": radix,
        "active_digits": problem.active_digits,
        "carry_heavy": int(problem.is_carry_heavy),
    }


def count_carry_chain(problem: AdditionProblem) -> int:
    longest = 0
    current = 0
    for index in range(problem.active_digits):
        if problem.carry_out[index]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def carry_density(problem: AdditionProblem) -> float:
    if problem.active_digits <= 0:
        return 0.0
    return float(sum(problem.carry_out[: problem.active_digits])) / float(problem.active_digits)


def curriculum_stage_lengths(config: ExperimentConfig) -> list[int]:
    if config.uses_curriculum:
        return list(range(1, config.train_max_digits + 1))
    return [config.train_max_digits]


def infer_eval_lengths(config: ExperimentConfig) -> list[int]:
    return sorted(set(range(1, config.train_max_digits + 1)).union(config.ood_lengths))


def estimate_train_tokens_per_step(config: ExperimentConfig, stage: int) -> int:
    latent_steps = config.latent_steps_for_stage(stage)
    return config.train_batch_size * (config.base_sequence_length_for_digits(stage) + latent_steps)


def stage_fraction(stage: int, max_stage: int) -> float:
    if max_stage <= 1:
        return 1.0
    return float(stage - 1) / float(max_stage - 1)


def maybe_trim_examples(problems: list[AdditionProblem], limit: int) -> list[AdditionProblem]:
    if limit <= 0 or len(problems) <= limit:
        return list(problems)
    return list(problems[:limit])


def stage_display_name(stage: int) -> str:
    suffix = "th"
    if stage % 10 == 1 and stage % 100 != 11:
        suffix = "st"
    elif stage % 10 == 2 and stage % 100 != 12:
        suffix = "nd"
    elif stage % 10 == 3 and stage % 100 != 13:
        suffix = "rd"
    return f"{stage}{suffix}-digit"


def ideal_carry_chain_examples(config: ExperimentConfig, active_digits: int) -> list[AdditionProblem]:
    examples: list[AdditionProblem] = []
    for base_digit in (max(0, config.radix - 2), config.radix - 1):
        a_digits = [base_digit] * active_digits
        b_digits = [1] * active_digits
        sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits, radix=config.radix)
        examples.append(
            AdditionProblem(
                a_digits=a_digits,
                b_digits=b_digits,
                sum_digits=sum_digits,
                carry_out=carry_out,
                active_digits=active_digits,
                is_carry_heavy=True,
            )
        )
    return examples


def expected_sum_length(problem: AdditionProblem) -> int:
    final_carry = problem.carry_out[problem.active_digits - 1]
    return problem.active_digits + int(final_carry > 0)


def average_query_count(config: ExperimentConfig) -> float:
    lengths = curriculum_stage_lengths(config)
    return sum(lengths) / float(len(lengths))


def token_budget(config: ExperimentConfig) -> int:
    avg_stage = int(math.ceil(average_query_count(config)))
    return config.base_sequence_length_for_digits(avg_stage) + config.latent_steps_for_stage(avg_stage)
