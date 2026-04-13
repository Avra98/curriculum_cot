from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch

from addition.config import ExperimentConfig


DIGIT_OFFSET = 0
A_TOKEN_ID = 10
B_TOKEN_ID = 11
QUERY_TOKEN_ID = 12
POSITION_TOKEN_OFFSET = 13


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
    target_carry: torch.Tensor
    query_positions: torch.Tensor
    active_digits: torch.Tensor
    is_carry_heavy: torch.Tensor


@dataclass
class EvaluationSuite:
    validation_uniform: dict[int, list[AdditionProblem]]
    test_uniform: dict[int, list[AdditionProblem]]
    test_carry_heavy: dict[int, list[AdditionProblem]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sum_and_carry(a_digits: list[int], b_digits: list[int]) -> tuple[list[int], list[int]]:
    sum_digits: list[int] = []
    carry_out: list[int] = []
    carry = 0
    for a_digit, b_digit in zip(a_digits, b_digits):
        total = int(a_digit) + int(b_digit) + carry
        sum_digits.append(total % 10)
        carry = total // 10
        carry_out.append(carry)
    return sum_digits, carry_out


def sample_uniform_problem(max_digits: int, active_digits: int, rng: random.Random) -> AdditionProblem:
    a_digits = [0] * max_digits
    b_digits = [0] * max_digits
    for index in range(active_digits):
        a_digits[index] = rng.randint(0, 9)
        b_digits[index] = rng.randint(0, 9)
    sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits)
    return AdditionProblem(
        a_digits=a_digits,
        b_digits=b_digits,
        sum_digits=sum_digits,
        carry_out=carry_out,
        active_digits=active_digits,
        is_carry_heavy=False,
    )


def sample_carry_heavy_problem(max_digits: int, active_digits: int, rng: random.Random) -> AdditionProblem:
    a_digits = [0] * max_digits
    b_digits = [0] * max_digits
    carry = 0
    for index in range(active_digits):
        if index == 0:
            a_digit = rng.randint(5, 9)
            b_digit = rng.randint(max(10 - a_digit, 1), 9)
        else:
            a_digit = rng.randint(4, 9)
            b_digit = rng.randint(max(9 - a_digit, 0), 9)
        if carry == 0 and a_digit + b_digit < 10:
            b_digit = min(9, 10 - a_digit)
        elif carry == 1 and a_digit + b_digit < 9:
            b_digit = min(9, 9 - a_digit)
        a_digits[index] = a_digit
        b_digits[index] = b_digit
        total = a_digit + b_digit + carry
        carry = total // 10
    sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits)
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
    rng: random.Random,
    carry_heavy: bool = False,
) -> AdditionProblem:
    if carry_heavy:
        return sample_carry_heavy_problem(max_digits=max_digits, active_digits=active_digits, rng=rng)
    return sample_uniform_problem(max_digits=max_digits, active_digits=active_digits, rng=rng)


def encode_problem_tokens(problem: AdditionProblem, query_position: int, max_digits: int) -> list[int]:
    if query_position < 0 or query_position >= max_digits:
        raise ValueError(f"query_position {query_position} out of range for max_digits={max_digits}")
    return (
        [A_TOKEN_ID]
        + [DIGIT_OFFSET + digit for digit in problem.a_digits]
        + [B_TOKEN_ID]
        + [DIGIT_OFFSET + digit for digit in problem.b_digits]
        + [QUERY_TOKEN_ID, POSITION_TOKEN_OFFSET + query_position]
    )


def build_batch(
    problems: list[AdditionProblem],
    query_positions: list[int],
    max_digits: int,
    device: str,
) -> Batch:
    input_ids = torch.tensor(
        [encode_problem_tokens(problem=problem, query_position=query_pos, max_digits=max_digits) for problem, query_pos in zip(problems, query_positions)],
        dtype=torch.long,
        device=device,
    )
    target_digits = torch.tensor(
        [problem.sum_digits[query_pos] for problem, query_pos in zip(problems, query_positions)],
        dtype=torch.long,
        device=device,
    )
    target_carry = torch.tensor(
        [problem.carry_out[query_pos] for problem, query_pos in zip(problems, query_positions)],
        dtype=torch.long,
        device=device,
    )
    return Batch(
        input_ids=input_ids,
        target_digits=target_digits,
        target_carry=target_carry,
        query_positions=torch.tensor(query_positions, dtype=torch.long, device=device),
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
    query_positions: list[int] = []
    for _ in range(config.train_batch_size):
        carry_heavy = rng.random() < config.train_carry_heavy_prob
        problem = sample_problem(
            max_digits=config.eval_max_digits,
            active_digits=stage,
            rng=rng,
            carry_heavy=carry_heavy,
        )
        problems.append(problem)
        query_positions.append(rng.randint(0, stage - 1))
    return build_batch(problems=problems, query_positions=query_positions, max_digits=config.eval_max_digits, device=device)


def build_problem_set(
    *,
    max_digits: int,
    active_digits: int,
    count: int,
    seed: int,
    carry_heavy: bool,
) -> list[AdditionProblem]:
    rng = random.Random(seed)
    return [
        sample_problem(max_digits=max_digits, active_digits=active_digits, rng=rng, carry_heavy=carry_heavy)
        for _ in range(count)
    ]


def build_evaluation_suite(config: ExperimentConfig) -> EvaluationSuite:
    validation_uniform: dict[int, list[AdditionProblem]] = {}
    test_uniform: dict[int, list[AdditionProblem]] = {}
    test_carry_heavy: dict[int, list[AdditionProblem]] = {}
    all_lengths = sorted(set(range(1, config.train_max_digits + 1)).union(config.ood_lengths))
    for length in all_lengths:
        validation_uniform[length] = build_problem_set(
            max_digits=config.eval_max_digits,
            active_digits=length,
            count=config.eval_examples_per_length,
            seed=10_000 + length,
            carry_heavy=False,
        )
        test_uniform[length] = build_problem_set(
            max_digits=config.eval_max_digits,
            active_digits=length,
            count=config.eval_examples_per_length,
            seed=20_000 + length,
            carry_heavy=False,
        )
        test_carry_heavy[length] = build_problem_set(
            max_digits=config.eval_max_digits,
            active_digits=length,
            count=config.carry_heavy_examples_per_length,
            seed=30_000 + length,
            carry_heavy=True,
        )
    return EvaluationSuite(
        validation_uniform=validation_uniform,
        test_uniform=test_uniform,
        test_carry_heavy=test_carry_heavy,
    )


def digits_to_string(digits: Iterable[int], final_carry: int) -> str:
    digits = list(digits)
    significant_digits = list(digits)
    if final_carry:
        significant_digits.append(final_carry)
    while len(significant_digits) > 1 and significant_digits[-1] == 0:
        significant_digits.pop()
    return "".join(str(digit) for digit in reversed(significant_digits))


def int_from_digits(digits: Iterable[int], final_carry: int) -> int:
    return int(digits_to_string(digits=digits, final_carry=final_carry))


def exact_sum_matches(
    predicted_digits: list[int],
    predicted_final_carry: int,
    truth_digits: list[int],
    truth_final_carry: int,
) -> bool:
    return int_from_digits(predicted_digits, predicted_final_carry) == int_from_digits(truth_digits, truth_final_carry)


def summarize_problem(problem: AdditionProblem) -> dict[str, int | str]:
    return {
        "a": int_from_digits(problem.a_digits[: problem.active_digits], final_carry=0),
        "b": int_from_digits(problem.b_digits[: problem.active_digits], final_carry=0),
        "sum": int_from_digits(problem.sum_digits[: problem.active_digits], final_carry=problem.carry_out[problem.active_digits - 1]),
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
    return config.train_batch_size * (config.base_sequence_length + latent_steps)


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
    for base_digit in (8, 9):
        a_digits = [base_digit] * active_digits + [0] * (config.eval_max_digits - active_digits)
        b_digits = [1] * active_digits + [0] * (config.eval_max_digits - active_digits)
        sum_digits, carry_out = compute_sum_and_carry(a_digits, b_digits)
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
    base = config.base_sequence_length
    avg_stage = int(math.ceil(average_query_count(config)))
    return base + config.latent_steps_for_stage(avg_stage)
