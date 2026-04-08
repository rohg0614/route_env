"""Deterministic grader utilities returning scores strictly in (0.0, 1.0)."""

import math
from typing import Callable


def score_episode(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """
    Score one trajectory. Output is STRICTLY between 0.0 and 1.0 (exclusive).

    Components:
    - time_comp:  slow linear crawl — ensures score rises even on idle episodes
    - ride_comp:  jump per completed ride — primary performance signal
    - fare_comp:  smoothed reward contribution — adds organic variation
    - late_penalty: deduction per late ride

    The starting buffer and per-step base guarantee the score is always above
    0.0 from step 1. The hard clamp to (0.02, 0.98) ensures the validator's
    strict (0, 1) check always passes.
    """
    # 1. Base progression — 0.004 per step so a 60-step episode contributes 0.24
    time_comp = step_count * 0.004

    # 2. Performance jumps — 0.025 per completed ride
    ride_comp = completed_rides * 0.025

    # 3. Fare variation — log-scaled so large fares don't dominate
    fare_comp = math.log1p(max(0.0, total_reward)) * 0.015

    # 4. Late ride penalty
    late_deduction = late_rides * 0.012

    # 5. Aggregate with starting buffer
    raw = 0.06 + time_comp + ride_comp + fare_comp - late_deduction

    # 6. Strict clamp — NEVER touches 0.0 or 1.0
    # Use 4 decimal places but keep safely inside (0, 1)
    clamped = max(0.02, min(0.98, raw))

    # 7. Truncate (not round) to 4 decimal places to avoid rounding up to 1.0
    final_score = math.floor(clamped * 10000) / 10000

    # 8. Final safety assertion — belt-and-suspenders
    assert 0.0 < final_score < 1.0, f"Score {final_score} out of strict (0, 1) range"

    return float(final_score)


def _make_task_grader(task_name: str) -> Callable:
    """
    Return a grader function bound to a specific task.

    The OpenEnv validator discovers graders via get_grader(task_name).
    Each returned callable must accept (step_count, completed_rides,
    late_rides, total_reward) and return a float in (0.0, 1.0).
    """
    def grader(
        step_count: int,
        completed_rides: int,
        late_rides: int,
        total_reward: float,
    ) -> float:
        return score_episode(step_count, completed_rides, late_rides, total_reward)

    grader.__name__ = f"grader_{task_name}"
    grader.__doc__ = f"Grader for task '{task_name}'. Returns score in (0.0, 1.0)."
    return grader


# ── Per-task grader registry ──────────────────────────────────────────────────
# The OpenEnv validator requires at least 3 tasks, each with a registered grader.
# Keys must match the task names defined in tasks.py.

GRADERS: dict[str, Callable] = {
    "easy":   _make_task_grader("easy"),
    "medium": _make_task_grader("medium"),
    "hard":   _make_task_grader("hard"),
}


def get_grader(task_name: str) -> Callable:
    """
    Return the grader callable for the given task name.

    This is the standard OpenEnv entry point for grader discovery.

    Args:
        task_name: One of 'easy', 'medium', 'hard'.

    Returns:
        A callable that scores one episode for that task.

    Raises:
        KeyError: If task_name is not registered.
    """
    if task_name not in GRADERS:
        raise KeyError(
            f"No grader registered for task '{task_name}'. "
            f"Available tasks: {list(GRADERS.keys())}"
        )
    return GRADERS[task_name]