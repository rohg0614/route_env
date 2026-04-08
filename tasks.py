"""Task presets for the route dispatch benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    horizon_steps: int
    node_count: int
    max_shift_hours: float
    base_lambda: float
    lateness_budget: float
    distance_scale: float


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig("easy", 60, 6, 8.0, 1.2, 0.45, 8.0),
    "medium": TaskConfig("medium", 84, 8, 9.0, 1.8, 0.32, 10.0),
    # Hard: more nodes, tighter lateness budget, higher demand variance,
    # longer horizon — designed so a greedy heuristic scores below 0.5.
    "hard": TaskConfig("hard", 120, 12, 10.0, 3.2, 0.15, 14.0),
}

TASK_ORDER = ["easy", "medium", "hard"]